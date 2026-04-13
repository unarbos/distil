#!/usr/bin/env python3
"""
Cosine similarity check between two HuggingFace models.

Computes layer-wise cosine similarity on weight tensors to detect
near-copies that bypass hash checks (slightly fine-tuned copies).

Usage:
    python cosine_similarity_check.py model_a model_b [--threshold 0.999]

Returns exit code 0 if models are sufficiently different, 1 if near-copy detected.
Outputs JSON with per-layer and aggregate similarity scores.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download, model_info


def download_model_weights(model_repo: str, cache_dir: str = None) -> Path:
    """Download only safetensors files from a HuggingFace model."""
    path = snapshot_download(
        model_repo,
        allow_patterns=["*.safetensors"],
        cache_dir=cache_dir,
    )
    return Path(path)


def load_state_dict_lazy(model_dir: Path) -> dict:
    """Load all safetensors files into a combined state dict (on CPU)."""
    state = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    return state


def compute_cosine_similarities(state_a: dict, state_b: dict) -> dict:
    """Compute per-layer cosine similarity between two state dicts."""
    common_keys = sorted(set(state_a.keys()) & set(state_b.keys()))
    if not common_keys:
        return {"error": "No common keys found", "layers": {}, "aggregate": 0.0}

    similarities = {}
    weighted_sum = 0.0
    total_params = 0

    for key in common_keys:
        a = state_a[key].float().flatten()
        b = state_b[key].float().flatten()

        if a.shape != b.shape:
            similarities[key] = {"cosine": None, "shape_mismatch": True}
            continue

        cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        n_params = a.numel()
        similarities[key] = {"cosine": round(cos_sim, 8), "params": n_params}

        weighted_sum += cos_sim * n_params
        total_params += n_params

    aggregate = weighted_sum / total_params if total_params > 0 else 0.0

    # Also compute a simple mean (unweighted)
    cosines = [v["cosine"] for v in similarities.values() if v.get("cosine") is not None]
    mean_cosine = sum(cosines) / len(cosines) if cosines else 0.0

    return {
        "layers": similarities,
        "aggregate_weighted": round(aggregate, 8),
        "aggregate_mean": round(mean_cosine, 8),
        "n_common_layers": len(common_keys),
        "n_total_params": total_params,
        "only_in_a": sorted(set(state_a.keys()) - set(state_b.keys())),
        "only_in_b": sorted(set(state_b.keys()) - set(state_a.keys())),
    }


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity check between HF models")
    parser.add_argument("model_a", help="First model (HF repo id)")
    parser.add_argument("model_b", help="Second model (HF repo id)")
    parser.add_argument("--threshold", type=float, default=0.999,
                        help="Similarity threshold for near-copy detection (default: 0.999)")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only output aggregate scores, not per-layer")
    args = parser.parse_args()

    print(f"Downloading {args.model_a}...", flush=True)
    path_a = download_model_weights(args.model_a, args.cache_dir)
    print(f"Downloading {args.model_b}...", flush=True)
    path_b = download_model_weights(args.model_b, args.cache_dir)

    print("Loading weights...", flush=True)
    state_a = load_state_dict_lazy(path_a)
    state_b = load_state_dict_lazy(path_b)

    print(f"Comparing {len(state_a)} vs {len(state_b)} tensors...", flush=True)
    result = compute_cosine_similarities(state_a, state_b)

    result["model_a"] = args.model_a
    result["model_b"] = args.model_b
    result["threshold"] = args.threshold
    result["is_near_copy"] = result["aggregate_weighted"] >= args.threshold

    if args.summary_only:
        result.pop("layers", None)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}", flush=True)

    print(f"\n{'='*60}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Weighted cosine similarity: {result['aggregate_weighted']:.6f}")
    print(f"Mean cosine similarity:     {result['aggregate_mean']:.6f}")
    print(f"Common layers: {result['n_common_layers']}")
    print(f"Total params compared: {result['n_total_params']:,}")
    print(f"Threshold: {args.threshold}")
    print(f"Near-copy detected: {'YES ⚠️' if result['is_near_copy'] else 'NO ✓'}")
    print(f"{'='*60}")

    sys.exit(1 if result["is_near_copy"] else 0)


if __name__ == "__main__":
    main()
