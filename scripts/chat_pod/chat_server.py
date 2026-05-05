#!/usr/bin/env python3
"""
King chat server bootstrapper — runs on the chat-bench pod.

Starts an OpenAI-compatible vLLM server for the current king model so
chat.arbos.life can talk to it via the SSH tunnel on port 8100.

Two supported student families, selected automatically from config.json:

  (A) Qwen3.5 / Qwen3.6 family (``Qwen3_5ForCausalLM`` or
      ``Qwen3_5ForConditionalGeneration``). Legacy path — was the only
      supported family before the Kimi K2.6 teacher swap. Requires the
      Qwen3_5ForConditionalGeneration VL wrapper and graft-in of base
      Qwen3.5-4B visual weights so vLLM's weight loader sees a complete
      checkpoint. Kept for backward compatibility so the current Qwen
      king keeps serving while miners migrate to Kimi-family arch.

  (B) Kimi K2.6 family — text-only ``DeepseekV3ForCausalLM`` (inner text
      model of the Kimi K2.6 wrapper) or the full
      ``KimiK25ForConditionalGeneration`` wrapper. vLLM 0.19+ supports
      both natively with ``--trust-remote-code``; no config rewrite or
      weight grafting needed. We just download-and-serve.

Usage:
    python3 chat_server.py <hf_repo>[:revision] [port]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAW_MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
if ":" in RAW_MODEL:
    MODEL_NAME, MODEL_REVISION = RAW_MODEL.split(":", 1)
else:
    MODEL_NAME, MODEL_REVISION = RAW_MODEL, None
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

MODEL_DIR = Path("/root/king-model")
# Legacy Qwen wrapper base — only used if the downloaded model is a
# Qwen3.5/3.6 family student that needs visual-weight grafting.
BASE_MODEL = "Qwen/Qwen3.5-4B"
SERVED_NAME = "sn97-king"


def _detect_arch_family() -> str:
    """Inspect the downloaded config.json and return the arch family.

    Returns one of ``"qwen35"``, ``"kimi_k2"`` (text-only), ``"kimi_k25"``
    (vision wrapper), or ``"unknown"``.
    """
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return "unknown"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return "unknown"
    archs = cfg.get("architectures") or []
    mt = cfg.get("model_type", "")
    if "KimiK25ForConditionalGeneration" in archs or mt == "kimi_k25":
        return "kimi_k25"
    if "DeepseekV3ForCausalLM" in archs and mt in ("kimi_k2", "deepseek_v3"):
        return "kimi_k2"
    if mt == "kimi_k2" or mt == "deepseek_v3":
        return "kimi_k2"
    if any(a.startswith("Qwen3_5") for a in archs) or mt.startswith("qwen3_5"):
        return "qwen35"
    return "unknown"


def log(msg: str):
    print(f"[chat-vllm] {msg}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kwargs)


def _hf_cli() -> str:
    """Return the HF CLI binary — `hf` replaced `huggingface-cli` in v1.10."""
    for candidate in ("hf", "huggingface-cli"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("no huggingface CLI found (need `hf` or `huggingface-cli`)")


_KING_MARKER = "/root/king-model/.king_marker.json"


def _read_marker() -> dict:
    """Return the persisted king-marker payload (model + revision + ts)."""
    try:
        with open(_KING_MARKER) as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _write_marker(model: str, revision: str | None):
    """Persist a marker file describing what's currently in MODEL_DIR.

    Lets the next chat_server startup skip the 30 GB re-download when the
    king hasn't changed. We persist BOTH the HF repo id and the revision
    so a same-repo king upgrade (e.g. UID promoting a new commit) still
    triggers a fresh pull.
    """
    try:
        with open(_KING_MARKER, "w") as f:
            json.dump(
                {"model": model, "revision": revision, "ts": time.time()}, f,
            )
    except OSError as exc:
        log(f"warning: could not write king marker: {exc}")


def _is_complete_download() -> bool:
    """Best-effort check that MODEL_DIR contains a usable model.

    Heuristic: ``config.json`` + ``model.safetensors.index.json`` present,
    AND every shard listed in the index actually exists on disk. We pick
    these two files because every transformers checkpoint we serve has
    them, and the index lets us verify the shards weren't truncated by
    a previous crashed download.
    """
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        return False
    index_path = MODEL_DIR / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path) as f:
                idx = json.load(f)
            shards = set((idx.get("weight_map") or {}).values())
            for shard in shards:
                if not (MODEL_DIR / shard).exists():
                    return False
            return True
        except (OSError, ValueError):
            return False
    # Single-file model — good enough if there's any *.safetensors.
    return any(MODEL_DIR.glob("*.safetensors"))


def download_model():
    """Download the king model — but skip the 30 GB pull if we already have it.

    Pre-2026-05-04 this function unconditionally ``shutil.rmtree``'d
    ``/root/king-model`` and re-downloaded. That's fine when chat_server
    only runs once per king crowning, but the API's
    ``_ensure_chat_server`` watchdog re-spawns the bootstrapper every
    time vLLM dies (eval-pod GPU contention, OOMs, restarts after
    network blips, etc.). On the chat-bench pod this means a fresh 30 GB
    DeepSeek-V3 / Kimi-K2 pull every restart even when the on-disk
    weights are already perfect — chat stays dark for 5-10 min instead
    of the ~60 s it takes vLLM to load + warm up. Worse, two concurrent
    downloads race-delete each other's tmp incomplete files (``shutil.move``
    blowups in ``hf_hub_download._chmod_and_move``), poisoning the
    download for both processes.

    The marker file (``/root/king-model/.king_marker.json``) records what
    HF repo + revision is in the dir. If the marker matches what we were
    asked to serve AND the shard files are still on disk, we skip the
    download entirely. Mismatch (new king, revision bump, partial
    download) → wipe + re-download fresh.

    Sebastian's report 2026-05-04 ("chat doesn't work with current king")
    was caused by exactly this race: the API spawned 2-3 chat_server.py
    processes during a single eval-end window, each tried to wipe + re-
    download, and the racing rmtree+download crashed all of them with
    ``FileNotFoundError`` on the same incomplete path.
    """
    marker = _read_marker()
    if (
        marker.get("model") == MODEL_NAME
        and marker.get("revision") == MODEL_REVISION
        and _is_complete_download()
    ):
        log(
            f"skipping download — marker matches "
            f"({MODEL_NAME}@{MODEL_REVISION or 'latest'}, "
            f"on disk since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(marker.get('ts', 0)))} UTC)"
        )
        return
    if MODEL_DIR.exists():
        log("wiping stale MODEL_DIR (marker mismatch or incomplete download)")
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [_hf_cli(), "download", MODEL_NAME, "--local-dir", str(MODEL_DIR)]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    run(cmd)
    _write_marker(MODEL_NAME, MODEL_REVISION)


def patch_config_and_tokenizer():
    """Normalize config.json to a Qwen3_5ForConditionalGeneration wrapper with
    text_config + vision_config + image/video token ids, so vLLM can resolve
    the architecture and so the visual weights we graft in line up.
    """
    from huggingface_hub import hf_hub_download

    config_path = MODEL_DIR / "config.json"
    tokenizer_path = MODEL_DIR / "tokenizer_config.json"

    with open(config_path) as f:
        cfg = json.load(f)

    ref_path = hf_hub_download(BASE_MODEL, "config.json")
    with open(ref_path) as f:
        ref = json.load(f)

    arch = cfg.get("architectures") or []

    if arch == ["Qwen3_5ForCausalLM"]:
        text_inner = dict(cfg)
        text_inner["architectures"] = ["Qwen3_5ForCausalLM"]
        text_inner["model_type"] = "qwen3_5_text"
        for k in (
            "vision_config", "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id",
            "tie_word_embeddings",
        ):
            text_inner.pop(k, None)
        cfg = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": text_inner,
            "torch_dtype": text_inner.get("torch_dtype", "bfloat16"),
            "transformers_version": ref.get("transformers_version"),
        }
        log("wrapped flat Qwen3_5ForCausalLM config → Qwen3_5ForConditionalGeneration")
    elif arch == ["Qwen3_5ForConditionalGeneration"]:
        if "text_config" not in cfg:
            raise RuntimeError(
                "Qwen3_5ForConditionalGeneration config is missing text_config"
            )
        text_inner = cfg["text_config"]
        text_inner.setdefault("architectures", ["Qwen3_5ForCausalLM"])
        text_inner.setdefault("model_type", "qwen3_5_text")
        log("using existing Qwen3_5ForConditionalGeneration wrapper")
    else:
        log(f"warning: unexpected architectures {arch!r}; leaving as-is")

    for key in (
        "vision_config", "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id",
    ):
        if key in ref:
            cfg[key] = ref[key]
    cfg.setdefault("tie_word_embeddings", ref.get("tie_word_embeddings", False))

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log("wrote patched config.json")

    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tok = json.load(f)
        tok["tokenizer_class"] = "Qwen2Tokenizer"
        tok.pop("auto_map", None)
        with open(tokenizer_path, "w") as f:
            json.dump(tok, f, indent=2)
        log("patched tokenizer_config.json")

    for fn in ("tokenizer.json", "tokenizer_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
            log(f"copied {fn} from base model")
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")

    for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, MODEL_DIR / fn)
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")

    for fn in ("chat_template.jinja",):
        dst = MODEL_DIR / fn
        if dst.exists():
            continue
        try:
            src = hf_hub_download(BASE_MODEL, fn)
            shutil.copy(src, dst)
            log(f"copied {fn} from base model")
        except Exception as e:
            log(f"warning: could not copy {fn}: {e}")


def inject_visual_weights():
    """Graft base visual weights into the miner's shard set as visual.safetensors,
    and rebuild model.safetensors.index.json so vLLM finds every key.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_shard = hf_hub_download(
        BASE_MODEL, "model.safetensors-00002-of-00002.safetensors"
    )
    visual_out = MODEL_DIR / "visual.safetensors"
    index_file = MODEL_DIR / "model.safetensors.index.json"

    model_shards = sorted(MODEL_DIR.glob("model*.safetensors"))
    if not model_shards:
        raise RuntimeError(f"missing model weights under {MODEL_DIR}")

    visual_tensors = {}
    with safe_open(base_shard, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.visual."):
                new_key = "visual." + key[len("model.visual."):]
                visual_tensors[new_key] = f.get_tensor(key)

    save_file(visual_tensors, str(visual_out))
    log(f"wrote visual shard with {len(visual_tensors)} tensors")

    weight_map = {}
    total_size = visual_out.stat().st_size
    for shard in model_shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name
        total_size += shard.stat().st_size
    for key in visual_tensors.keys():
        weight_map[key] = "visual.safetensors"

    with open(index_file, "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f, indent=2,
        )
    log(f"wrote sharded index with {len(weight_map)} keys across {len(model_shards)+1} shards")


def write_health(status: str = "starting"):
    payload = {
        "status": status,
        "backend": "vllm",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "port": PORT,
        "ts": time.time(),
    }
    with open("/root/chat_health.json", "w") as f:
        json.dump(payload, f)
    with open("/root/model_name.txt", "w") as f:
        f.write(MODEL_NAME)


def _install_custom_reasoning_parser():
    """Copy ``distil_kimi_reasoning_parser.py`` into vLLM's parser dir
    AND patch vLLM's ``vllm/reasoning/__init__.py`` to import it.

    The parser self-registers via the
    ``@ReasoningParserManager.register_module("distil_kimi")`` decorator,
    but dropping the file alone isn't enough: vLLM's ``__init__.py``
    only auto-imports parsers listed in ``_REASONING_PARSERS_TO_REGISTER``
    (lazy registration); the file just sits there until something imports
    it. So we also append a single ``from . import …`` line to
    ``__init__.py`` (guarded by a sentinel so re-runs are no-ops),
    which forces the decorator to run at vLLM startup.

    We do BOTH actions from chat_server (NOT from a separate deployment
    step) so the chat-king pod self-heals when the underlying vLLM
    image is wiped or upgraded — every fresh bootstrap re-installs the
    parser before launching vLLM.

    Idempotent: re-copies the parser file on every boot so an in-place
    edit ships, but the ``__init__.py`` patch is sentinel-guarded so it
    only adds the import line once. No-op (logs a warning) if the
    source file is missing or the destination dir doesn't exist (e.g. a
    future vLLM re-org); chat falls back to the stock parsers and
    ``--reasoning-parser distil_kimi`` will fail loudly at startup.
    """
    src = Path(__file__).parent / "distil_kimi_reasoning_parser.py"
    if not src.is_file():
        log(f"warning: custom reasoning parser src missing: {src}")
        return
    candidates = list(
        Path("/usr/local/lib").glob("python3.*/dist-packages/vllm/reasoning")
    )
    if not candidates:
        candidates = list(
            Path("/usr/lib").glob("python3.*/dist-packages/vllm/reasoning")
        )
    if not candidates:
        log("warning: vLLM reasoning dir not found; skipping custom parser install")
        return
    dst_dir = candidates[0]
    dst = dst_dir / "distil_kimi_reasoning_parser.py"
    try:
        shutil.copy2(src, dst)
        log(f"installed custom reasoning parser: {dst}")
    except OSError as exc:
        log(f"warning: could not install custom reasoning parser: {exc}")
        return

    init_path = dst_dir / "__init__.py"
    sentinel = "# distil_kimi_reasoning_parser auto-registered"
    import_line = (
        f"\n{sentinel}\n"
        "try:\n"
        "    from . import distil_kimi_reasoning_parser  # noqa: F401\n"
        "except Exception as _e:  # pragma: no cover\n"
        "    import logging\n"
        "    logging.getLogger('vllm').warning(\n"
        "        'distil_kimi parser auto-import failed: %s', _e,\n"
        "    )\n"
    )
    try:
        existing = init_path.read_text(encoding="utf-8")
        if sentinel in existing:
            log(f"distil_kimi parser already wired into {init_path.name}")
            return
        with open(init_path, "a", encoding="utf-8") as f:
            f.write(import_line)
        log(f"wired distil_kimi import into {init_path}")
    except OSError as exc:
        log(f"warning: could not patch vLLM __init__.py: {exc}")


def exec_vllm():
    write_health()
    _install_custom_reasoning_parser()
    # Chat-king coexists with the validator's eval workload on the same GPU
    # most of the time. ``pod_eval.py`` claims ~0.90 of the H200 during
    # rounds, so a 0.90 chat slice would OOM the second vLLM to come up
    # (whichever loses the race). Default to a slim slice that comfortably
    # fits a 4B-class model + KV cache and tune via env if a future king is
    # bigger or the eval pod gets a smaller card.
    gpu_util = os.environ.get("CHAT_VLLM_GPU_UTIL", "0.30")
    max_model_len = os.environ.get("CHAT_VLLM_MAX_MODEL_LEN", "32768")
    family = _detect_arch_family()
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_DIR),
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--served-model-name", SERVED_NAME,
        "--gpu-memory-utilization", str(gpu_util),
        "--enforce-eager",
    ]
    # 2026-05-05 follow-up to Sebastian's 2026-05-04 "chat doesn't work with
    # current king" report. Earlier fix made the ``kimi_k2`` / ``kimi_k25``
    # branches omit ``--enable-auto-tool-choice`` entirely because we
    # believed vLLM didn't ship a Kimi tool parser. That kept vLLM from
    # crashing at boot, but it broke the chat differently: Open-WebUI
    # sends ``tools[]`` with implicit ``tool_choice: "auto"`` (the
    # ``sn97-king`` model row has ``params.function_calling = "native"``
    # plus the SN97 status toolkit attached by default), and vLLM rejects
    # those requests with HTTP 400:
    #
    #   "auto" tool choice requires --enable-auto-tool-choice and
    #   --tool-call-parser to be set
    #
    # Effect: every chat.arbos.life turn returned an opaque "tool calls"
    # error and the king never responded. vLLM 0.19.1 actually does ship
    # both ``kimi_k2`` and ``deepseek_v3`` tool parsers (registered via
    # ``vllm/tool_parsers/__init__.py``'s ``_TOOL_PARSERS_TO_REGISTER`` map)
    # plus matching reasoning parsers, so we can wire them up properly
    # for both Kimi families now.
    if family == "qwen35":
        # Qwen 3.5 / 3.6 emit ``<tool_call><function=name><parameter=k>v
        # </parameter></function></tool_call>`` XML — the ``qwen3_xml``
        # parser ships with vLLM 0.19+ and matches the family natively.
        cmd += [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_xml",
            "--reasoning-parser", "qwen3",
            "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
            "--skip-mm-profiling",
        ]
    elif family == "kimi_k25":
        # Kimi K2.5/K2.6 vision wrapper — disable vision path for text
        # chat, wire the Kimi tool parser so Open-WebUI's
        # ``function_calling=native`` requests succeed end-to-end, and
        # use the custom ``distil_kimi`` reasoning parser
        # (scripts/chat_pod/distil_kimi_reasoning_parser.py, installed
        # by ``_install_custom_reasoning_parser`` below) so the
        # Thinking pane lights up cleanly:
        #   * model emits ``thoughts</think>answer`` → split (thinking
        #     pane shows ``thoughts``, answer shows ``answer``)
        #   * model emits ``answer`` (no </think>) → all content (NO
        #     thinking pane that turn — accurate, the model didn't
        #     think — but the answer is visible, not buried in
        #     reasoning like the stock kimi_k2 parser does)
        # Stock vLLM ``kimi_k2`` reasoning parser was tried earlier and
        # rejected because it returned ``content=None`` for every turn
        # the king didn't close ``</think>`` — verified empty on
        # ``bodenmaurice/distil-new-v16``.
        cmd += [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "kimi_k2",
            "--reasoning-parser", "distil_kimi",
            "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
            "--skip-mm-profiling",
        ]
    elif family == "kimi_k2":
        # Text-only DeepSeek V3 inner of Kimi K2 — same tool-call +
        # reasoning template as the vision wrapper. See ``kimi_k25``
        # for the rationale on the custom ``distil_kimi`` parser.
        cmd += [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "kimi_k2",
            "--reasoning-parser", "distil_kimi",
        ]
    else:
        # Unknown architecture — pass no family-specific flags. We also
        # leave auto-tool-choice off because we can't guess the right
        # parser. vLLM may succeed on simple architectures (Llama-family)
        # without them.
        log(f"warning: unknown architecture family, falling back to minimal vLLM args")
    log(f"exec vLLM (family={family}, gpu_util={gpu_util}, max_model_len={max_model_len})")
    os.execvp(cmd[0], cmd)


def _acquire_startup_lock():
    """Serialise concurrent ``chat_server.py`` startups via ``flock``.

    The API watchdog (``api/routes/chat.py:_ensure_chat_server``) and the
    validator's ``ensure_chat_server_running`` BOTH spawn this script
    when chat is dark, and they don't coordinate with each other. We've
    seen up to 3 simultaneous chat_server processes during a single
    eval-end window, all racing on the same model dir and port. The
    second/third invocations crash on:

      * ``FileNotFoundError`` mid-download (rmtree races)
      * ``Address already in use`` when two vLLMs fight for port 8100
      * dueling weight loaders if both somehow get past download

    A non-blocking ``fcntl.flock(LOCK_EX | LOCK_NB)`` on a pid-stamped
    lock file at ``/tmp/chat_server.lock`` lets the first invocation win
    cleanly: subsequent invocations log + exit 0 (so the watchdog
    doesn't see a crash and re-spawn yet again). The OS releases the
    lock automatically when this process exits — no manual cleanup
    needed even on hard kill.
    """
    import fcntl
    lock_path = "/tmp/chat_server.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            with open(lock_path) as f:
                holder = f.read().strip()
        except OSError:
            holder = "?"
        log(
            f"another chat_server is already starting (pid={holder}); "
            f"exiting cleanly to avoid race."
        )
        sys.exit(0)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    # Intentionally leak the fd so the lock survives until process exit.
    return fd


if __name__ == "__main__":
    _acquire_startup_lock()
    rev_suffix = f"@{MODEL_REVISION}" if MODEL_REVISION else ""
    log(f"bootstrapping model={MODEL_NAME}{rev_suffix} port={PORT}")
    download_model()
    family = _detect_arch_family()
    log(f"detected arch family: {family}")
    if family == "qwen35":
        # Legacy Qwen students: run the VL-wrapper dance so vLLM sees the
        # full config + visual weights.
        patch_config_and_tokenizer()
        inject_visual_weights()
    else:
        # Kimi-family and unknown: download-and-serve. vLLM loads the
        # model's own config.json / tokenizer directly.
        log(f"skipping Qwen-specific patch + visual-weight grafting for family={family}")
    exec_vllm()
