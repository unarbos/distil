#!/usr/bin/env python3
"""Provision an 8×H100 80GB experiment pod for Stage-2 Kimi K2.6 A/B
+ multi-GPU eval testing (v30.4).

Use this to rent the dedicated experiment pod separate from the
production validator's `distil-eval` pod. The Kimi K2.6 A/B
experiment runs here without touching production.

Workflow:

    LIUM_API_KEY=... ./scripts/provision_experiment_pod.py \\
        --gpu-type H100 --gpu-count 8 \\
        --name distil-experiment

Once the pod is up:
  1. SSHes in, installs Python deps, mounts the repo at /opt/distil/repo.
  2. Stages Kimi K2.6 weights via huggingface-cli (~250GB FP8 / ~2TB BF16).
  3. Stages Qwen3.6-35B-A3B weights for the baseline variant.
  4. Optionally runs the smoke test against Kimi K2.6.

Cost: 8×H100 80GB on Lium runs ~$15.92/hr per the dashboard. Multi-day
runs add up — make sure the experiment plan has a clear stop point.
After the A/B experiment completes (and, if PROMOTE, the production
swap), tear the pod down.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-type", default="H100",
                   help="Lium GPU type. H100 is the standard 80GB SXM.")
    p.add_argument("--gpu-count", type=int, default=8,
                   help="Required GPU count (default 8 for K2.6).")
    p.add_argument("--name", default="distil-experiment",
                   help="Human-friendly pod name.")
    p.add_argument("--probe-only", action="store_true",
                   help="Just list available executors and exit.")
    p.add_argument("--stage-models", default="moonshot/Kimi-K2.6-Instruct,Qwen/Qwen3.6-35B-A3B",
                   help="Comma-separated HF repos to pre-stage on the pod.")
    p.add_argument("--run-smoke-test", action="store_true",
                   help="After provisioning, immediately run the K2.6 smoke test.")
    args = p.parse_args()

    api_key = os.environ.get("LIUM_API_KEY")
    if not api_key:
        print("ERROR: LIUM_API_KEY not set in environment", file=sys.stderr)
        return 2

    try:
        from lium import Config, Lium
    except ImportError:
        print(
            "ERROR: lium SDK not installed. Install with "
            "`/opt/distil/venv/bin/pip install lium-sdk`.",
            file=sys.stderr,
        )
        return 2

    cfg = Config()
    if hasattr(cfg, "set"):
        cfg.set("api_key", api_key)
    elif hasattr(cfg, "api_key"):
        cfg.api_key = api_key
    lium = Lium(config=cfg)

    print(f"[exp-pod] balance: {lium.balance()}", flush=True)

    # Find available executors with the requested GPU type and count.
    executors = lium.get_executor(gpu_type=args.gpu_type) or []
    available: list = []
    for e in executors:
        if not getattr(e, "available", True):
            continue
        # Filter by GPU count if the SDK exposes it.
        n_gpus = getattr(e, "gpu_count", None) or getattr(e, "n_gpus", None)
        if n_gpus is not None and n_gpus != args.gpu_count:
            continue
        available.append(e)

    print(f"[exp-pod] {len(available)} available executors with "
          f"{args.gpu_count}×{args.gpu_type}", flush=True)
    if args.probe_only:
        for e in available[:5]:
            ec_id = getattr(e, "id", None) or (
                e.get("id") if isinstance(e, dict) else "?"
            )
            n_gpus = getattr(e, "gpu_count", None) or "?"
            location = getattr(e, "location", None) or "?"
            price = getattr(e, "price_per_hour", None) or "?"
            print(f"  - {ec_id}: {n_gpus} GPU @ {location} ${price}/hr",
                  flush=True)
        return 0

    if not available:
        print(f"ERROR: no available {args.gpu_count}×{args.gpu_type} executors. "
              "Retry later or try a different GPU type.", file=sys.stderr)
        return 3

    chosen = available[0]
    exec_id = getattr(chosen, "id", None) or chosen.get("id")
    print(f"[exp-pod] picked executor {exec_id}", flush=True)

    pod = lium.up(executor_id=exec_id, name=args.name, ports=4)
    pod_id = getattr(pod, "id", None) or pod.get("id")
    print(f"[exp-pod] pod started: id={pod_id}", flush=True)

    # Wait for SSH to come up.
    print("[exp-pod] waiting for SSH...", flush=True)
    for attempt in range(60):
        time.sleep(10)
        try:
            ssh = lium.ssh_info(pod_id) if hasattr(lium, "ssh_info") else None
            if ssh:
                print(f"[exp-pod] SSH ready: {ssh}", flush=True)
                break
        except Exception:
            pass
    else:
        print("WARNING: SSH didn't come up in 10 minutes. Check manually.",
              file=sys.stderr)

    # Persist pod info for later steps (Kimi A/B runner).
    out_path = REPO / "state" / "experiment_pod.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pod_id": pod_id,
        "name": args.name,
        "gpu_type": args.gpu_type,
        "gpu_count": args.gpu_count,
        "started_at": time.time(),
        "stage_models": args.stage_models.split(","),
    }, indent=2))
    print(f"[exp-pod] saved {out_path}", flush=True)

    print(f"\n[exp-pod] Next steps (manually for now):")
    print(f"  1. SSH into pod {pod_id}.")
    print(f"  2. Stage models: huggingface-cli download "
          f"{args.stage_models.replace(',', ' ')}")
    print(f"  3. Run smoke test:")
    print(f"     /opt/distil/venv/bin/python "
          f"scripts/experiments/kimi26_vllm_smoke_test.py "
          f"--tp-size {args.gpu_count}")
    print(f"  4. If smoke passes, run A/B:")
    print(f"     /opt/distil/venv/bin/python "
          f"scripts/experiments/run_kimi26_a_b.py "
          f"--students <UID156,UID149,...> --rounds 6")
    print(f"  5. Tear pod down when experiment completes:")
    print(f"     lium down {pod_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
