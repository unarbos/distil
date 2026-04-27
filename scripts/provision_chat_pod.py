#!/usr/bin/env python3
"""Provision a dedicated Lium pod for the chat-king vLLM.

Use this to decouple chat.arbos.life from the eval pod so eval can use
the full GPU. The eval pod stays on its current H200/H200-NVL host;
this script rents a separate (smaller) GPU just for the chat king.

Workflow:

    LIUM_API_KEY=... ./scripts/provision_chat_pod.py \
        --gpu-type RTX_4090 \
        --hours 24 \
        --target-model eugene141759/distil-m20

Once the pod is up:

  1. The script SSH-installs the chat_server.py and starts vLLM with the
     target model on port 8100.
  2. It updates ``state/chat_pod.json`` so the validator's chat-tunnel
     watcher rebinds to the new host.
  3. It prints the SSH coordinates so the operator can verify.

Cost (rough): an RTX 4090 is ~$0.20-0.40/hr on Lium. A 4B model on a
4090 fits comfortably with 16-20 GB of headroom. Running 24h costs
~$5-10. Worth it once the eval pod is running rounds back-to-back.

After verifying the new pod works, the operator can bump
``VLLM_EVAL_GPU_UTIL`` from 0.65 → 0.92 in distil.env, restart the
validator, and unlock the full eval-pod GPU for teacher inference.
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
    p.add_argument(
        "--gpu-type",
        default="RTX_4090",
        help="Lium GPU type. RTX_4090 is plenty for a 4B chat model. "
        "Heavier chat models can use H100_PCIE.",
    )
    p.add_argument(
        "--target-model",
        required=True,
        help="HuggingFace repo of the model to serve as the king "
        "(e.g. eugene141759/distil-m20).",
    )
    p.add_argument(
        "--name", default="distil-chat-king", help="Human pod name.",
    )
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU util on the chat pod. Higher than the 0.15 we "
        "used when chat was co-located with eval — the chat pod has "
        "the GPU to itself.",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Just list available executors and exit. Doesn't rent.",
    )
    args = p.parse_args()

    api_key = os.environ.get("LIUM_API_KEY")
    if not api_key:
        print("ERROR: LIUM_API_KEY not set in environment", file=sys.stderr)
        return 2

    try:
        from lium import Config, Lium
    except ImportError:
        print(
            "ERROR: lium SDK not installed. "
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

    print(f"[chat-pod] balance: {lium.balance()}", flush=True)
    print(f"[chat-pod] gpu_types available:", flush=True)
    for g in lium.gpu_types() or []:
        print(f"  - {g}", flush=True)

    if args.probe_only:
        return 0

    # Find an available executor with the requested GPU type.
    executors = lium.get_executor(gpu_type=args.gpu_type) or []
    available = [e for e in executors if getattr(e, "available", True)]
    if not available:
        print(
            f"ERROR: no available {args.gpu_type} executors. "
            "Try a different GPU type or retry later.",
            file=sys.stderr,
        )
        return 3
    chosen = available[0]
    exec_id = getattr(chosen, "id", None) or chosen.get("id")
    print(f"[chat-pod] picked executor {exec_id} ({args.gpu_type})", flush=True)

    # Rent the pod.
    pod = lium.up(executor_id=exec_id, name=args.name, ports=2)
    pod_id = pod.get("id")
    ssh_cmd = pod.get("ssh_command") or pod.get("ssh") or ""
    print(f"[chat-pod] rented pod_id={pod_id}", flush=True)
    print(f"[chat-pod] ssh: {ssh_cmd}", flush=True)

    # Wait for SSH to come up.
    print(f"[chat-pod] waiting for SSH (~60s)...", flush=True)
    time.sleep(60)

    # Parse host + port from SSH command (e.g. `ssh -p 12345 root@1.2.3.4`).
    host = None
    port = None
    if "@" in ssh_cmd:
        try:
            tokens = ssh_cmd.split()
            for i, tok in enumerate(tokens):
                if tok == "-p" and i + 1 < len(tokens):
                    port = int(tokens[i + 1])
                if "@" in tok:
                    host = tok.split("@")[-1]
        except Exception:
            pass

    if not host or not port:
        print(
            "WARN: could not parse SSH host/port from rent response. "
            "You'll need to update state/chat_pod.json manually.",
            file=sys.stderr,
        )
        print(json.dumps(pod, indent=2, default=str), file=sys.stderr)
        return 4

    print(
        f"[chat-pod] host={host} port={port} — installing chat_server...",
        flush=True,
    )

    # Copy chat_server.py + start vLLM. We use scripts/validator/chat_pod_admin
    # rather than reinventing here; it knows how to update state and start.
    state_dir = os.environ.get("DISTIL_STATE_DIR", str(REPO / "state"))
    cmd = [
        "/opt/distil/venv/bin/python",
        "-m", "scripts.validator.chat_pod_admin", "set",
        "--host", host,
        "--ssh-port", str(port),
        "--app-port", "8100",
        "--ssh-key", os.path.expanduser("~/.ssh/id_ed25519"),
        "--model", args.target_model,
        "--note", f"Provisioned by provision_chat_pod.py at {int(time.time())}; "
                  f"dedicated chat GPU ({args.gpu_type})",
    ]
    env = dict(os.environ)
    env["DISTIL_STATE_DIR"] = state_dir
    res = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return 5

    print(f"[chat-pod] healing chat_server.py on the new pod...", flush=True)
    res = subprocess.run(
        [
            "/opt/distil/venv/bin/python",
            "-m", "scripts.validator.chat_pod_admin", "heal",
        ],
        env=env, cwd=str(REPO), capture_output=True, text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return 6

    print(f"[chat-pod] probing /v1/models...", flush=True)
    res = subprocess.run(
        [
            "/opt/distil/venv/bin/python",
            "-m", "scripts.validator.chat_pod_admin", "probe",
        ],
        env=env, cwd=str(REPO), capture_output=True, text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(
            "WARN: probe failed. Pod is rented and chat_server is starting; "
            "the chat-tunnel watcher will rebind once vLLM is ready (~5 min).",
            file=sys.stderr,
        )

    print()
    print("[chat-pod] DONE.", flush=True)
    print()
    print("Next steps:")
    print(
        "  1. Verify chat.arbos.life works (it'll switch automatically "
        "once chat-tunnel.path picks up state/chat_pod.json).",
    )
    print(
        "  2. Once verified, set VLLM_EVAL_GPU_UTIL=0.92 in "
        "/home/distil/.secrets/distil.env to unlock the full eval-pod GPU.",
    )
    print(
        "  3. Optionally bump vllmConcurrency in subnet-config.json from "
        "48 → 96 for faster teacher generation.",
    )
    print(
        "  4. Restart distil-validator at the next round boundary.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
