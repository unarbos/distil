#!/usr/bin/env python3
"""
King Model Integrity Check
==========================
Reusable script to verify the integrity of the current king model on the
Distillation subnet. Downloads only metadata (config.json, tokenizer_config.json,
chat_template.jinja) plus a 100KB header fingerprint from the first safetensors
shard — no full model download required for the metadata checks.

Usage:
    python scripts/check_king_integrity.py <repo_id> [--full-fingerprint]
    python scripts/check_king_integrity.py slowsnake/qwen35-4b-kd-v10

Output: structured integrity report to stdout (and optionally JSON).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    sys.exit("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")

# ---------------------------------------------------------------------------
# Known watermarks — add new ones as they're discovered
# ---------------------------------------------------------------------------
KNOWN_WATERMARKS = [
    "{# model distilled by caseus #}",
    "trained by caseus",
    "distilled by caseus",
    # Add more known watermarks here as they appear
]

# Suspicious patterns (case-insensitive search)
SUSPICIOUS_PATTERNS = [
    "distilled by",
    "trained by",
    "made by",
    "watermark",
    "model distilled",
    "fine-tuned by",
    "finetuned by",
]

# Expected architecture families for the subnet (Qwen 4B class)
VALID_ARCHITECTURES = [
    "Qwen2ForCausalLM",
    "Qwen2_5ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3_5ForCausalLM",
]

MAX_PARAM_COUNT = 5.25e9  # 5.25 billion parameters

# Known fingerprints: sha256(first 100KB) -> description
# This acts as a dedup database — if two models share a fingerprint, they're
# likely the same weights (or a trivial rename).
KNOWN_FINGERPRINTS = {
    # Format: "sha256_hex": "repo_id or description"
    "1e6fa78cf5237f466b8a8cef84e1135c77ff8294c69c9c0d5d2757778f564a52": "slowsnake/qwen35-4b-kd-v10 (king as of 2026-04-02)",
}


def download_metadata(repo_id: str) -> dict:
    """Download config files without pulling the full model."""
    result = {"repo_id": repo_id, "files": []}

    files = list(list_repo_files(repo_id))
    result["files"] = files

    # config.json
    try:
        path = hf_hub_download(repo_id, "config.json")
        result["config"] = json.load(open(path))
    except Exception as e:
        result["config"] = None
        result["config_error"] = str(e)

    # tokenizer_config.json
    try:
        path = hf_hub_download(repo_id, "tokenizer_config.json")
        result["tokenizer_config"] = json.load(open(path))
    except Exception as e:
        result["tokenizer_config"] = None
        result["tokenizer_config_error"] = str(e)

    # chat_template.jinja (standalone file, common in Qwen models)
    try:
        path = hf_hub_download(repo_id, "chat_template.jinja")
        result["chat_template_file"] = open(path).read()
    except Exception:
        result["chat_template_file"] = None

    return result


def check_watermarks(metadata: dict) -> list:
    """Check all template sources for known watermarks."""
    findings = []

    # Gather all template text
    templates = {}
    if metadata.get("chat_template_file"):
        templates["chat_template.jinja"] = metadata["chat_template_file"]
    tc = metadata.get("tokenizer_config") or {}
    if tc.get("chat_template"):
        templates["tokenizer_config.chat_template"] = str(tc["chat_template"])

    for source, text in templates.items():
        text_lower = text.lower()

        # Exact known watermarks
        for wm in KNOWN_WATERMARKS:
            if wm.lower() in text_lower:
                findings.append({
                    "type": "known_watermark",
                    "severity": "HIGH",
                    "source": source,
                    "watermark": wm,
                })

        # Suspicious patterns
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in text_lower:
                findings.append({
                    "type": "suspicious_pattern",
                    "severity": "MEDIUM",
                    "source": source,
                    "pattern": pattern,
                })

    return findings


def check_architecture(metadata: dict) -> dict:
    """Verify model architecture and parameter count."""
    config = metadata.get("config") or {}
    result = {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type", "unknown"),
        "valid_architecture": False,
        "estimated_params": None,
        "under_param_limit": None,
        "trust_remote_code_needed": config.get("auto_map") is not None,
    }

    # Check architecture
    archs = config.get("architectures", [])
    result["valid_architecture"] = any(a in VALID_ARCHITECTURES for a in archs)

    # Estimate parameter count
    h = config.get("hidden_size", 0)
    L = config.get("num_hidden_layers", 0)
    V = config.get("vocab_size", 0)
    i = config.get("intermediate_size", 0)
    n_heads = config.get("num_attention_heads", 0)
    kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", h // n_heads if n_heads else 0)

    if h and L and V:
        # Attention: Q + K + V + O projections
        attn = L * (h * n_heads * head_dim + 2 * h * kv_heads * head_dim + n_heads * head_dim * h)
        # FFN: gate + up + down
        ffn = L * (3 * h * i) if i else 0
        # Embeddings (may be tied)
        tie = config.get("tie_word_embeddings", False)
        embed = V * h * (1 if tie else 2)
        total = attn + ffn + embed
        result["estimated_params"] = total
        result["estimated_params_b"] = round(total / 1e9, 3)
        result["under_param_limit"] = total < MAX_PARAM_COUNT

    return result


def compute_fingerprint(repo_id: str, header_bytes: int = 102400) -> dict:
    """Compute SHA256 of first N bytes of the first safetensors shard."""
    result = {"sha256_100kb": None, "file_size_gb": None, "known_match": None}

    # Find safetensors files
    files = list(list_repo_files(repo_id))
    safetensors = sorted([f for f in files if f.endswith(".safetensors")])

    if not safetensors:
        result["error"] = "No safetensors files found"
        return result

    target = safetensors[0]
    try:
        path = hf_hub_download(repo_id, target)
        import os
        result["file_size_gb"] = round(os.path.getsize(path) / 1e9, 3)

        with open(path, "rb") as f:
            header = f.read(header_bytes)
        sha = hashlib.sha256(header).hexdigest()
        result["sha256_100kb"] = sha
        result["shard_file"] = target

        # Check against known fingerprints
        if sha in KNOWN_FINGERPRINTS:
            result["known_match"] = KNOWN_FINGERPRINTS[sha]

    except Exception as e:
        result["error"] = str(e)

    return result


def generate_report(repo_id: str, skip_fingerprint: bool = False) -> dict:
    """Run all checks and return structured report."""
    print(f"[*] Downloading metadata for {repo_id}...")
    metadata = download_metadata(repo_id)

    print("[*] Checking watermarks...")
    watermarks = check_watermarks(metadata)

    print("[*] Checking architecture...")
    arch = check_architecture(metadata)

    fingerprint = {}
    if not skip_fingerprint:
        print("[*] Computing fingerprint (downloads first safetensors shard)...")
        fingerprint = compute_fingerprint(repo_id)
    else:
        print("[*] Skipping fingerprint (--skip-fingerprint)")

    # Overall verdict
    issues = []
    if watermarks:
        issues.append(f"{len(watermarks)} watermark/pattern findings")
    if not arch["valid_architecture"]:
        issues.append(f"Invalid architecture: {arch['architectures']}")
    if arch["under_param_limit"] is False:
        issues.append(f"Over param limit: {arch.get('estimated_params_b', '?')}B > 5.25B")
    if arch["trust_remote_code_needed"]:
        issues.append("Requires trust_remote_code (potential code execution)")

    verdict = "FAIL" if issues else "PASS"

    report = {
        "repo_id": repo_id,
        "verdict": verdict,
        "issues": issues,
        "watermark_findings": watermarks,
        "architecture": arch,
        "fingerprint": fingerprint,
        "files_in_repo": metadata.get("files", []),
    }

    return report


def print_report(report: dict):
    """Pretty-print the integrity report."""
    v = report["verdict"]
    icon = "❌" if v == "FAIL" else "✅"
    print(f"\n{'=' * 60}")
    print(f"  MODEL INTEGRITY REPORT: {report['repo_id']}")
    print(f"  Verdict: {icon} {v}")
    print(f"{'=' * 60}")

    if report["issues"]:
        print("\n⚠️  ISSUES:")
        for issue in report["issues"]:
            print(f"  - {issue}")

    # Watermarks
    if report["watermark_findings"]:
        print("\n🔍 WATERMARK / PATTERN FINDINGS:")
        for f in report["watermark_findings"]:
            sev = f["severity"]
            src = f["source"]
            if f["type"] == "known_watermark":
                print(f"  [{sev}] Known watermark in {src}: {f['watermark']}")
            else:
                print(f"  [{sev}] Pattern in {src}: \"{f['pattern']}\"")

    # Architecture
    arch = report["architecture"]
    print(f"\n🏗️  ARCHITECTURE:")
    print(f"  Architectures: {arch['architectures']}")
    print(f"  Model type: {arch['model_type']}")
    print(f"  Valid architecture: {'✅' if arch['valid_architecture'] else '❌'}")
    if arch["estimated_params_b"]:
        limit_ok = "✅" if arch["under_param_limit"] else "❌"
        print(f"  Estimated params: {arch['estimated_params_b']}B {limit_ok} (limit: 5.25B)")
    print(f"  Needs trust_remote_code: {'⚠️ YES' if arch['trust_remote_code_needed'] else '✅ No'}")

    # Fingerprint
    fp = report.get("fingerprint", {})
    if fp.get("sha256_100kb"):
        print(f"\n🔑 FINGERPRINT:")
        print(f"  Shard: {fp.get('shard_file', 'N/A')}")
        print(f"  Size: {fp.get('file_size_gb', 'N/A')} GB")
        print(f"  SHA256 (100KB): {fp['sha256_100kb']}")
        if fp.get("known_match"):
            print(f"  Known match: {fp['known_match']}")

    print(f"\n📁 Repo files: {', '.join(report.get('files_in_repo', []))}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Check king model integrity")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. slowsnake/qwen35-4b-kd-v10)")
    parser.add_argument("--skip-fingerprint", action="store_true",
                        help="Skip downloading safetensors for fingerprinting")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = generate_report(args.repo_id, skip_fingerprint=args.skip_fingerprint)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
