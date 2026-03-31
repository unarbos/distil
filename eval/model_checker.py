"""
Model architecture checker with MoE-aware param counting and identity verification.

Checks:
1. Total parameter count ≤ max allowed (prevents huge MoE uploads)
2. Active parameter count for MoE models (logged for transparency)
3. Vocab size matches teacher (same tokenizer required)
4. SHA256 hash of first safetensors shard (copy detection)
5. Tokenizer encoding verification (spot-check)
"""
import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, model_info

logger = logging.getLogger("distillation.model_checker")

# Qwen3.5-35B-A3B: vocab_size=248320 in config (text_config.vocab_size)
# Note: tokenizer.vocab_size reports 248044 but model config uses 248320 (padded)
BASELINE_VOCAB_SIZE = 248320
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
STATE_DIR = Path("state")


def compute_moe_params(config: dict) -> dict:
    """
    Compute total and active parameters for MoE models.

    Returns dict with:
        - total_params: all parameters including all experts
        - active_params: parameters active during a single forward pass
        - is_moe: whether model uses MoE
        - num_experts: total experts per layer
        - num_active_experts: experts active per token
    """
    # Support nested configs (e.g. text_config for multimodal models)
    # Merge text_config into a flat lookup: text_config values are used as fallback
    text_cfg = config.get("text_config", {})
    def _get(key, default=0):
        """Get from top-level config first, then text_config, then default."""
        v = config.get(key)
        if v is None or v == 0:
            v = text_cfg.get(key)
        return v if v is not None else default

    hidden = _get("hidden_size", 0)
    layers = _get("num_hidden_layers", 0)
    vocab = _get("vocab_size", 0)
    intermediate = _get("intermediate_size", hidden * 4)
    num_heads = _get("num_attention_heads", 0)
    kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)

    if not all([hidden, layers, vocab]):
        return {"total_params": 0, "active_params": 0, "is_moe": False}

    # Attention params per layer: Q + K + V + O
    attn_per_layer = (
        hidden * num_heads * head_dim  # Q
        + hidden * kv_heads * head_dim  # K
        + hidden * kv_heads * head_dim  # V
        + num_heads * head_dim * hidden  # O
    )
    total_attn = layers * attn_per_layer

    # Embeddings: input + output (may share weights)
    tie_word = config.get("tie_word_embeddings", False)
    embed_params = vocab * hidden * (1 if tie_word else 2)

    # Layer norms, biases, etc. (rough estimate)
    norm_params = layers * hidden * 4  # 2 norms per layer, 2 params each

    # MoE detection — check both top-level and text_config
    num_experts = _get("num_local_experts", 0) or _get("num_experts", 1)
    num_active = _get("num_experts_per_tok", 0) or _get("num_active_experts", num_experts)
    is_moe = num_experts > 1

    # FFN params per expert (SwiGLU: gate + up + down)
    # Some models use moe_intermediate_size for expert FFN
    expert_intermediate = _get("moe_intermediate_size", intermediate)
    ffn_per_expert = hidden * expert_intermediate * 2 + expert_intermediate * hidden

    if is_moe:
        # Some layers may be dense (shared experts)
        num_shared = _get("num_shared_experts", 0)
        shared_intermediate = _get("shared_expert_intermediate_size", intermediate)
        shared_ffn = hidden * shared_intermediate * 2 + shared_intermediate * hidden if num_shared else 0

        router_per_layer = hidden * num_experts
        total_ffn = layers * (num_experts * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
        active_ffn = layers * (num_active * ffn_per_expert + router_per_layer + num_shared * shared_ffn)
    else:
        total_ffn = layers * ffn_per_expert
        active_ffn = total_ffn

    total_params = total_attn + total_ffn + embed_params + norm_params
    active_params = total_attn + active_ffn + embed_params + norm_params

    return {
        "total_params": total_params,
        "active_params": active_params,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_active_experts": num_active,
    }


def get_safetensors_param_count(model_repo: str, revision: str = None) -> float:
    """Get verified param count from safetensors metadata (billions). Returns -1 if unavailable."""
    try:
        info = model_info(model_repo, revision=revision)
        if info.safetensors and hasattr(info.safetensors, "total"):
            return info.safetensors.total / 1e9
    except Exception:
        pass
    return -1.0


def compute_model_hash(model_repo: str, revision: str = None) -> Optional[str]:
    """
    Get a stable identity hash for a model using HuggingFace API metadata.
    Uses the SHA256 from safetensors file info (no download needed).
    Returns hex digest or None if unavailable.
    """
    try:
        info = model_info(model_repo, revision=revision, files_metadata=True)
        # Find first safetensors shard and use its SHA from HF API
        for sibling in sorted(info.siblings or [], key=lambda s: s.rfilename):
            if sibling.rfilename.endswith(".safetensors"):
                # HF provides lfs sha256 for each file
                if hasattr(sibling, "lfs") and sibling.lfs:
                    return sibling.lfs.get("sha256", sibling.lfs.get("oid", None))
                # Fallback: use the blob_id (git SHA) as identity
                if hasattr(sibling, "blob_id") and sibling.blob_id:
                    return sibling.blob_id
        # No safetensors found
        return None
    except Exception as e:
        logger.warning(f"Hash computation failed for {model_repo}: {e}")
        return None


def check_duplicate_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
) -> Optional[int]:
    """
    Check if this model hash was already submitted by a different miner.
    Returns the UID of the original submitter, or None if unique.
    """
    hash_file = state_dir / "model_hashes.json"
    if not hash_file.exists():
        return None
    try:
        hashes = json.loads(hash_file.read_text())
        for uid_str, stored_hash in hashes.items():
            if stored_hash == model_hash and int(uid_str) != miner_uid:
                return int(uid_str)
    except Exception:
        pass
    return None


def register_model_hash(
    model_hash: str, miner_uid: int, state_dir: Path = STATE_DIR,
):
    """Register a model hash for a miner UID."""
    state_dir.mkdir(parents=True, exist_ok=True)
    hash_file = state_dir / "model_hashes.json"
    hashes = {}
    if hash_file.exists():
        try:
            hashes = json.loads(hash_file.read_text())
        except Exception:
            pass
    hashes[str(miner_uid)] = model_hash
    hash_file.write_text(json.dumps(hashes, indent=2))


def verify_tokenizer(teacher_model: str, student_model: str) -> tuple[bool, str]:
    """Verify student uses exact same tokenizer as teacher."""
    from transformers import AutoTokenizer

    t_tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    s_tok = AutoTokenizer.from_pretrained(student_model, trust_remote_code=False)

    if t_tok.vocab_size != s_tok.vocab_size:
        return False, f"Vocab size mismatch: {s_tok.vocab_size} ≠ {t_tok.vocab_size} (teacher)"

    test_strings = [
        "def fibonacci(n):\n    if n <= 1: return n",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nclass Model(nn.Module):",
        "class MyClass(BaseClass):\n    def __init__(self):\n        pass",
    ]
    for s in test_strings:
        if t_tok.encode(s) != s_tok.encode(s):
            return False, f"Tokenizer encoding mismatch on test string: '{s[:40]}...'"

    return True, "ok"


def verify_model_integrity(
    model_repo: str,
    revision: str = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """
    Pre-weight-setting integrity check:
    1. Model is still publicly accessible on HuggingFace
    2. Model weights haven't changed since commitment (SHA256 match)

    Returns dict with:
      pass: bool
      reason: str
      current_hash: str or None
    """
    try:
        # 1. Check model is still public (HEAD request to repo)
        info = model_info(model_repo, revision=revision)
        if info.private:
            return {
                "pass": False,
                "reason": f"Model {model_repo} is now private — must be public for transparency",
                "current_hash": None,
            }
        if info.disabled:
            return {
                "pass": False,
                "reason": f"Model {model_repo} has been disabled on HuggingFace",
                "current_hash": None,
            }
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} no longer exists on HuggingFace (404)",
                "current_hash": None,
            }
        if "403" in err or "restricted" in err.lower():
            return {
                "pass": False,
                "reason": f"Model {model_repo} is restricted/gated — must be publicly accessible",
                "current_hash": None,
            }
        # Transient errors should not DQ
        err_lower = err.lower()
        if any(k in err_lower for k in ["429", "rate limit", "too many", "timeout", "503", "502", "connection"]):
            return {
                "pass": True,
                "reason": f"transient_error: {err}",
                "current_hash": None,
                "transient": True,
            }
        return {
            "pass": False,
            "reason": f"Cannot verify model accessibility: {err}",
            "current_hash": None,
        }

    # 2. Verify weights haven't changed (SHA256 of first shard)
    current_hash = compute_model_hash(model_repo, revision)
    if not current_hash:
        return {
            "pass": False,
            "reason": f"Cannot compute model hash — safetensors may have been removed",
            "current_hash": None,
        }

    if expected_hash and current_hash != expected_hash:
        return {
            "pass": False,
            "reason": f"Model weights changed since commitment! hash {current_hash[:16]}... ≠ expected {expected_hash[:16]}...",
            "current_hash": current_hash,
        }

    return {
        "pass": True,
        "reason": "ok",
        "current_hash": current_hash,
    }


# Fixed test strings for tokenizer verification — diverse enough to catch mismatches
TOKENIZER_TEST_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "日本語のテスト文字列です。Unicode handling matters.",
    "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) for all x in vocabulary",
]
_teacher_tokenizer = None


def _get_teacher_tokenizer():
    """Lazily load and cache the teacher tokenizer."""
    global _teacher_tokenizer
    if _teacher_tokenizer is None:
        from transformers import AutoTokenizer
        _teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    return _teacher_tokenizer


def verify_tokenizer_match(model_repo: str, revision: str = None) -> dict:
    """
    Verify that a model's tokenizer produces identical token IDs as the teacher.
    
    Downloads the student tokenizer and encodes fixed test strings.
    If any encoding differs, the tokenizer is incompatible.
    """
    from transformers import AutoTokenizer

    teacher_tok = _get_teacher_tokenizer()
    # NEVER trust_remote_code for students — blocks custom tokenizer.py code execution exploits
    student_tok = AutoTokenizer.from_pretrained(model_repo, revision=revision, trust_remote_code=False)

    for test_str in TOKENIZER_TEST_STRINGS:
        teacher_ids = teacher_tok.encode(test_str)
        student_ids = student_tok.encode(test_str)
        if teacher_ids != student_ids:
            return {
                "match": False,
                "reason": (
                    f"Encoding mismatch on test string: "
                    f"teacher produced {len(teacher_ids)} tokens, "
                    f"student produced {len(student_ids)} tokens"
                ),
            }

    return {"match": True}


def check_model_architecture(
    model_repo: str,
    revision: str = None,
    max_total_params_b: float = 3.5,
) -> dict:
    """
    Check if a model meets distillation subnet requirements.

    Checks:
    - Total params ≤ max_total_params_b (prevents huge MoE uploads)
    - Vocab size matches baseline (same tokenizer)
    - Reports active params for MoE transparency

    Returns dict with pass, reason, params_b, active_params_b, vocab_size
    """
    try:
        # 0. SECURITY: Reject repos with custom Python code files
        # This blocks exploits like tokenizer.py that monkey-patch json.dump
        try:
            info = model_info(model_repo, revision=revision)
            dangerous_files = []
            for sibling in (info.siblings or []):
                fname = sibling.rfilename
                if fname.endswith('.py') and fname != '__init__.py':
                    dangerous_files.append(fname)
            if dangerous_files:
                return {
                    "pass": False,
                    "reason": f"SECURITY: Repo contains custom code files ({', '.join(dangerous_files)}). "
                              f"Custom code is not allowed — students must use standard architectures only.",
                    "params_b": 0,
                }
        except Exception as e:
            logger.warning(f"Could not check repo files for {model_repo}: {e}")

        # 0b. SECURITY: Check minimum safetensors file size.
        # A real 1B+ param model in bf16 is at least ~2GB. Tiny files are fake.
        MIN_SAFETENSORS_BYTES = 500_000_000  # 500MB — even a 0.5B model is bigger than this
        try:
            total_st_bytes = 0
            for sibling in (info.siblings or []):
                if sibling.rfilename.endswith('.safetensors'):
                    if hasattr(sibling, 'size') and sibling.size is not None:
                        total_st_bytes += sibling.size
                    elif hasattr(sibling, 'lfs') and sibling.lfs:
                        total_st_bytes += sibling.lfs.get('size', 0)
            if 0 < total_st_bytes < MIN_SAFETENSORS_BYTES:
                return {
                    "pass": False,
                    "reason": f"FRAUD: Safetensors total size {total_st_bytes:,} bytes is impossibly small for a real model (min {MIN_SAFETENSORS_BYTES:,})",
                    "params_b": 0,
                }
        except Exception as e:
            logger.warning(f"Could not check safetensors file sizes for {model_repo}: {e}")

        # 1. Get safetensors-verified param count
        safetensors_params_b = get_safetensors_param_count(model_repo, revision)

        # 2. Download config.json
        config_path = hf_hub_download(
            repo_id=model_repo, filename="config.json", revision=revision,
        )
        with open(config_path) as f:
            config = json.load(f)

        # 3. MoE-aware param counting from config
        moe_info = compute_moe_params(config)
        config_total_b = moe_info["total_params"] / 1e9
        config_active_b = moe_info["active_params"] / 1e9

        # Use safetensors count if available (most accurate), else config estimate
        total_params_b = safetensors_params_b if safetensors_params_b > 0 else config_total_b

        if total_params_b <= 0:
            return {
                "pass": False,
                "reason": "Cannot determine parameter count — model may be missing safetensors metadata and config",
                "params_b": 0,
            }

        # 4. Check TOTAL param count (not active — prevents gaming with huge MoE)
        if total_params_b > max_total_params_b:
            return {
                "pass": False,
                "reason": f"Model too large: {total_params_b:.2f}B > {max_total_params_b:.1f}B max",
                "params_b": total_params_b,
                "active_params_b": config_active_b,
            }

        # 5. Reject quantized models (GPTQ, AWQ, GGUF, etc.)
        quant_config = config.get("quantization_config", {})
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown")
            return {
                "pass": False,
                "reason": f"Quantized model detected ({quant_method}) — subnet requires bf16/fp16 architecture distillation",
                "params_b": total_params_b,
            }

        # 6. Check vocab size (may be in text_config for multimodal/nested configs)
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)
        if vocab_size != BASELINE_VOCAB_SIZE:
            return {
                "pass": False,
                "reason": f"Vocab size mismatch: {vocab_size} ≠ {BASELINE_VOCAB_SIZE} (teacher)",
                "params_b": total_params_b,
                "vocab_size": vocab_size,
            }

        # 7. Verify tokenizer produces identical encodings as teacher
        try:
            tokenizer_match = verify_tokenizer_match(model_repo, revision)
            if not tokenizer_match["match"]:
                return {
                    "pass": False,
                    "reason": f"Tokenizer mismatch: {tokenizer_match['reason']}",
                    "params_b": total_params_b,
                    "vocab_size": vocab_size,
                }
        except Exception as tok_err:
            logger.warning(f"Tokenizer check failed for {model_repo}: {tok_err} (allowing)")
            # Don't block on tokenizer download failure — vocab_size check is the primary gate

        # Log MoE info for transparency
        if moe_info["is_moe"]:
            logger.info(
                f"  MoE model: {moe_info['num_experts']} experts, "
                f"{moe_info['num_active_experts']} active/token, "
                f"total={total_params_b:.2f}B, active={config_active_b:.2f}B"
            )

        return {
            "pass": True,
            "reason": "ok",
            "params_b": total_params_b,
            "active_params_b": config_active_b,
            "vocab_size": vocab_size,
            "is_moe": moe_info["is_moe"],
        }

    except Exception as e:
        err_str = str(e).lower()
        # Transient errors (rate limits, network issues) should NOT disqualify
        is_transient = any(k in err_str for k in [
            "429", "rate limit", "too many requests",
            "connection", "timeout", "503", "502",
            "temporary", "unavailable",
        ])
        if is_transient:
            return {"pass": True, "reason": f"transient_error:{e}", "transient": True}
        return {"pass": False, "reason": f"check_failed:{e}"}
