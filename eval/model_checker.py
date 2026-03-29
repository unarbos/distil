"""
Model Architecture Checker (adapted from Affine Cortex pattern).

Downloads config.json from HuggingFace, verifies:
1. Total parameter count ≤ max allowed
2. Uses compatible tokenizer (same vocab_size as GLM-5)

Quantization-proof: checks architecture fields, not file sizes.
"""
import json, logging, os
from huggingface_hub import hf_hub_download, model_info

logger = logging.getLogger("distillation.model_checker")

# Baseline model reference values
# Qwen3.5-35B-A3B: vocab_size=248320 (padded), 35B total, 3B active
# ALL student models MUST use the same tokenizer (verified by exact encoding match)
BASELINE_VOCAB_SIZE = 248320


def estimate_params_from_config(config: dict) -> float:
    """Estimate total parameters in billions from config.json fields."""
    # Try direct field first
    if "num_parameters" in config:
        return config["num_parameters"] / 1e9

    # For transformer models, estimate from architecture
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    intermediate = config.get("intermediate_size", hidden * 4)
    num_heads = config.get("num_attention_heads", 0)
    kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads if num_heads else 0

    if not all([hidden, layers, vocab]):
        return 0.0

    # Attention: Q + K + V + O projections
    attn_params = layers * (
        hidden * num_heads * head_dim
        + 2 * hidden * kv_heads * head_dim
        + num_heads * head_dim * hidden
    )
    # FFN: gate + up + down for SwiGLU
    ffn_params = layers * (hidden * intermediate * 2 + intermediate * hidden)
    # Embeddings: input + output
    embed_params = vocab * hidden * 2

    # For MoE models, multiply FFN by num_experts
    num_experts = config.get("num_local_experts", config.get("num_experts", 1))
    if num_experts > 1:
        router_params = layers * hidden * num_experts
        ffn_params = ffn_params * num_experts + router_params

    total = attn_params + ffn_params + embed_params
    return total / 1e9


def verify_same_tokenizer(teacher_model: str, student_model: str) -> tuple:
    """Verify student uses EXACT same tokenizer as teacher. Returns (ok, reason)."""
    from transformers import AutoTokenizer
    t_tok = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
    s_tok = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)

    if t_tok.vocab_size != s_tok.vocab_size:
        return False, f"vocab_size mismatch: {s_tok.vocab_size} vs {t_tok.vocab_size}"

    test_strings = [
        "def fibonacci(n):\n    if n <= 1: return n",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nclass Model(nn.Module):",
        "class MyClass(BaseClass):\n    def __init__(self):\n        pass",
    ]
    for s in test_strings:
        if t_tok.encode(s) != s_tok.encode(s):
            return False, f"encoding mismatch on: {s[:40]}..."

    return True, "ok"


def get_verified_param_count(model_repo: str, revision: str = None) -> float:
    """Get cheat-proof param count from safetensors metadata (billions). Returns -1 if can't verify."""
    info = model_info(model_repo, revision=revision)
    if info.safetensors and hasattr(info.safetensors, 'total'):
        return info.safetensors.total / 1e9
    return -1.0


def check_model_architecture(
    model_repo: str, revision: str = None, max_params_b: float = 74.4
) -> dict:
    """
    Check if a model meets the distillation subnet requirements.

    Returns dict with:
        - pass: bool
        - reason: str
        - params_b: float (estimated total params in billions)
        - vocab_size: int
    """
    try:
        # Try safetensors metadata first (most accurate)
        info = model_info(model_repo, revision=revision)
        params_b = None
        if info.safetensors and hasattr(info.safetensors, "total"):
            params_b = info.safetensors.total / 1e9

        # Download config.json
        config_path = hf_hub_download(
            repo_id=model_repo,
            filename="config.json",
            revision=revision,
        )
        with open(config_path) as f:
            config = json.load(f)

        # If no safetensors metadata, estimate from config
        if params_b is None:
            params_b = estimate_params_from_config(config)

        if params_b <= 0:
            return {"pass": False, "reason": "cannot_determine_param_count", "params_b": 0}

        # Check param count
        if params_b > max_params_b:
            return {
                "pass": False,
                "reason": f"too_large:{params_b:.1f}B > {max_params_b:.1f}B",
                "params_b": params_b,
            }

        # Check tokenizer compatibility (vocab_size must match GLM-5)
        vocab_size = config.get("vocab_size", 0)
        if vocab_size != BASELINE_VOCAB_SIZE:
            return {
                "pass": False,
                "reason": f"vocab_mismatch:{vocab_size} != {BASELINE_VOCAB_SIZE}",
                "params_b": params_b,
                "vocab_size": vocab_size,
            }

        return {"pass": True, "reason": "ok", "params_b": params_b, "vocab_size": vocab_size}

    except Exception as e:
        return {"pass": False, "reason": f"check_failed:{e}"}
