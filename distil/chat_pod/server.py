"""Bootstrap the king chat server on a Lium chat pod.

Run as ``python -m distil.chat_pod.server <model-repo>``. Spawns
``vllm serve`` on port 8100 with ``--served-model-name sn97-king`` and
hooks the Kimi reasoning-content parser. Intended to be supervised by
``deploy/systemd/distil-chat.service`` on the chat pod.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from distil.settings import settings

logger = logging.getLogger("distil.chat_pod.server")

DEFAULT_PORT = 8100
SERVED_NAME = "sn97-king"


def _install_kimi_parser() -> None:
    """Make ``distil.chat_pod.kimi_parser`` importable as ``distil_kimi_reasoning_parser``."""
    src = Path(__file__).parent / "kimi_parser.py"
    if not src.exists():
        logger.warning("kimi_parser.py missing; reasoning_content will be raw")
        return
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dst = site_packages / "distil_kimi_reasoning_parser.py"
    try:
        shutil.copyfile(src, dst)
        logger.info(f"installed kimi parser -> {dst}")
    except Exception as exc:
        logger.warning(f"could not install kimi parser: {exc}")


def serve(model_repo: str, *, port: int = DEFAULT_PORT) -> int:
    _install_kimi_parser()
    cmd = [
        "vllm",
        "serve",
        model_repo,
        "--port",
        str(port),
        "--served-model-name",
        SERVED_NAME,
        "--max-model-len",
        str(settings.vllm_max_model_len),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        str(settings.vllm_gpu_memory_utilization),
        "--dtype",
        settings.vllm_dtype,
        "--trust-remote-code",
        "--reasoning-parser",
        "distil_kimi",
    ]
    env = os.environ.copy()
    env["HF_TOKEN"] = settings.hf_dl_token or settings.teacher_hf_token or env.get("HF_TOKEN", "")
    logger.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_repo")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    return serve(args.model_repo, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
