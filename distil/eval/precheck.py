"""Pre-GPU model checks (architecture / size / vocab / integrity / hash).

Run on the validator host before any GPU time is allocated. A model
that fails the precheck is DQ'd immediately (and the miner gets a
clear error message via ``distil check``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from distil.settings import settings

logger = logging.getLogger("distil.eval.precheck")

# Conservative caps; tuned to leave headroom for B200 + 32K context.
MAX_TOTAL_PARAMS = 1_500_000_000_000
MAX_ACTIVE_PARAMS = 200_000_000_000
MAX_VOCAB_SIZE = 256_000
ALLOWED_DTYPES = ("float16", "bfloat16", "float32")
DISALLOWED_QUANT = ("gptq", "awq", "bitsandbytes", "compressed_tensors", "fp8")


@dataclass
class PrecheckResult:
    """One precheck verdict."""

    ok: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_total_active(config: dict) -> tuple[int | None, int | None, bool]:
    """Best-effort total/active param + is-MoE inference from the config."""
    total = config.get("num_params") or config.get("total_params")
    is_moe = bool(
        config.get("num_local_experts")
        or config.get("num_experts")
        or "MoE" in (str(config.get("model_type", "")))
    )
    active = config.get("num_active_params")
    return total, active, is_moe


def _allow_dtype(config: dict) -> tuple[bool, str]:
    dt = config.get("torch_dtype") or ""
    if dt and dt not in ALLOWED_DTYPES:
        return False, f"disallowed_dtype:{dt}"
    qc = config.get("quantization_config") or {}
    method = (qc.get("quant_method") or qc.get("config_format") or "").lower()
    if method and any(b in method for b in DISALLOWED_QUANT):
        return False, f"disallowed_quant:{method}"
    return True, ""


def _hf_metadata(repo: str, revision: str = "") -> dict[str, Any]:
    """Fetch ``config.json`` + ``tokenizer_config.json`` + commit sha via the HF API."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=settings.teacher_hf_token or None)
    info = api.model_info(repo, revision=revision or None)
    sha = info.sha or revision or "main"
    cfg_path = hf_hub_download(
        repo_id=repo, filename="config.json", revision=sha, token=settings.teacher_hf_token or None
    )
    config = json.loads(open(cfg_path).read())
    try:
        tok_path = hf_hub_download(
            repo_id=repo,
            filename="tokenizer_config.json",
            revision=sha,
            token=settings.teacher_hf_token or None,
        )
        tok_cfg = json.loads(open(tok_path).read())
    except Exception:
        tok_cfg = {}
    return {
        "config": config,
        "tokenizer_config": tok_cfg,
        "hf_sha": sha,
        "siblings": [s.rfilename for s in (info.siblings or [])],
    }


def _weight_files(siblings: list[str]) -> list[str]:
    return [s for s in siblings if re.search(r"\.(safetensors|bin)$", s)]


def precheck(model_repo: str, revision: str = "") -> PrecheckResult:
    """Run the full precheck battery; first failure short-circuits with a reason."""
    try:
        meta = _hf_metadata(model_repo, revision)
    except Exception as exc:
        return PrecheckResult(False, f"hf_metadata_fetch_failed: {exc}")

    config = meta["config"]
    arches = config.get("architectures") or []
    if not arches or not isinstance(arches, list):
        return PrecheckResult(False, "no_architectures_in_config")
    vocab = int(config.get("vocab_size") or 0)
    if vocab and vocab > MAX_VOCAB_SIZE:
        return PrecheckResult(False, f"vocab_too_large:{vocab}>{MAX_VOCAB_SIZE}")
    total, active, is_moe = _extract_total_active(config)
    if total and total > MAX_TOTAL_PARAMS:
        return PrecheckResult(False, f"total_params_too_large:{total}>{MAX_TOTAL_PARAMS}")
    if active and active > MAX_ACTIVE_PARAMS:
        return PrecheckResult(False, f"active_params_too_large:{active}>{MAX_ACTIVE_PARAMS}")
    ok, why = _allow_dtype(config)
    if not ok:
        return PrecheckResult(False, why)
    weight_files = _weight_files(meta["siblings"])
    if not weight_files:
        return PrecheckResult(False, "no_weight_files_found")
    fingerprint = hashlib.sha256(
        ("\n".join(sorted(weight_files)) + "|" + meta["hf_sha"]).encode()
    ).hexdigest()
    return PrecheckResult(
        True,
        "ok",
        metadata={
            "config": config,
            "hf_sha": meta["hf_sha"],
            "weight_files": weight_files,
            "fingerprint_sha256": fingerprint,
            "total_params": total,
            "active_params": active,
            "is_moe": is_moe,
        },
    )
