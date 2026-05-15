"""Multi-GPU parallel student-eval orchestrator.

Strategy:

  Phase 1 (single GPU):
    - Run pod_eval.py with --students <king> on GPU 0.
    - Phase 1 (teacher API gen) populates teacher_cache.pt (300 prompts, top-K
      logprobs). King is also scored as the first student.
    - Output: king_result.json

  Phase 2 (N GPUs in parallel):
    - Split the remaining challengers into N roughly-equal shards.
    - Spawn N worker subprocesses, one per GPU, each:
      * CUDA_VISIBLE_DEVICES=k
      * --students <shard>
      * --teacher-logits=<shared teacher_cache.pt>   (cache hit, no API calls)
      * --output=shard_k.json
      * --use-vllm-students                          (vLLM batched bench gen)
    - Wait for all workers, abort the round if any wedge.

  Merge:
    - Combine king_result.json + shard_*.json into final_eval_results.json.

Why a separate orchestrator (not patching pod_eval_vllm.py):
  - pod_eval_vllm.py is 19K lines and is the production validator's main entry
    point. Adding multi-GPU plumbing inside it risks regressing prod.
  - Each worker is a clean, isolated process with its own CUDA context, vLLM
    engine, and HF student. No shared state to deadlock or corrupt.
  - The teacher cache is a plain pickle file on disk: read-only, shared via
    POSIX (no locking needed; workers only read).
  - Resume semantics (--resume + already-scored skip) carry over automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


_CHALLENGERS_SPAWNED: dict = {}


# 2026-05-15: secrets are NEVER hardcoded. The orchestrator reads them from
# its parent environment. The validator/pod_session.py forwards these via
# the same allow-list used for pod_eval.py. If we ever launch the
# orchestrator manually, source /home/distil/.secrets/distil.env first.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
OPENROUTER_KEY = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("DISTIL_TEACHER_API_KEY")
    or ""
)
TOK_PATH = os.environ.get(
    "DISTIL_STUDENT_VLLM_TOKENIZER",
    # Default to the locally-cached Kimi-K2.6 snapshot every distilled
    # student in the active pool tokenizes against. Overridable via env
    # when we cut over to a new teacher snapshot.
    "/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/"
    "snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58",
)

if not OPENROUTER_KEY:
    print(
        "[orch] FATAL: OPENROUTER_API_KEY / DISTIL_TEACHER_API_KEY not set in "
        "environment. Refusing to launch — Phase 1 would fail downstream with "
        "an unhelpful auth error. Source /home/distil/.secrets/distil.env or "
        "pass the secret via the validator's env allow-list.",
        flush=True,
    )
    sys.exit(2)


def _shared_env() -> dict:
    """Env vars common to every worker (and the king/Phase-1 run).

    HF_HOME points at the pod overlay (/home, ~700GB free) instead of the
    default /root (the 494GB attached volume which fills up after a handful
    of student weights). Existing complete downloads on /root are symlinked
    into the new path so we don't redownload them.
    """
    e = dict(os.environ)
    e.update({
        "HF_HOME": "/home/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/home/.cache/huggingface/hub",
        "HF_HUB_CACHE": "/home/.cache/huggingface/hub",
        "HF_TOKEN": HF_TOKEN,
        "DISTIL_TEACHER_MODE": "api",
        "OPENROUTER_API_KEY": OPENROUTER_KEY,
        "DISTIL_TEACHER_API_BASE": "https://openrouter.ai/api",
        "DISTIL_TEACHER_API_MODEL": "moonshotai/kimi-k2.6",
        "DISTIL_TEACHER_API_KEY": OPENROUTER_KEY,
        "DISTIL_TEACHER_API_ENDPOINT": "chat",
        "DISTIL_TEACHER_API_PROVIDERS": "Inceptron",
        "DISTIL_TEACHER_API_TOP_LOGPROBS": "20",
        "DISTIL_TEACHER_API_CONCURRENCY": "4",
        "DISTIL_TEACHER_API_DISABLE_REASONING": "1",
        "DISTIL_STUDENT_USE_VLLM": "1",
        "DISTIL_STUDENT_VLLM_TOKENIZER": TOK_PATH,
        "DISTIL_STUDENT_VLLM_MAX_LEN": "16384",
        "DISTIL_STUDENT_VLLM_GPU_UTIL": "0.55",
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_DEEP_GEMM_E8M0": "0",
        "DISTIL_EVAL_POLICY": "/home/dev_eval/eval_policy.json",
        "ACTIVATION_FP_VOCAB_SIZE": "163840",
        "TEACHER_CONFIG_VOCAB_SIZE": "163840",
        # Each worker gets its own progress file so they don't stomp on each other.
        # Set per-worker before spawn.
    })
    return e


def _common_args(workdir: str, prompts: str, teacher_cache: str) -> list[str]:
    """CLI args common to every pod_eval.py invocation.

    Both --teacher-logits (load path) and --save-teacher-logits (save path) point
    at the same shared file. The first invocation (Phase 1 + king) creates it;
    every subsequent worker hits the cache and skips Phase 1 entirely.
    """
    return [
        "/usr/bin/python3", "-u", "pod_eval.py",
        "--teacher", "moonshotai/Kimi-K2.6",
        "--prompts", prompts,
        "--teacher-logits", teacher_cache,
        "--save-teacher-logits", teacher_cache,
        "--no-vllm",                      # disable teacher vLLM (we use API)
        "--max-new-tokens", "512",
        "--concurrency", "4",
        "--teacher-mode", "api",
        "--tensor-parallel-size", "1",
        "--block-seed", "42",
        "--use-vllm-students",            # vLLM batched bench gen for students
    ]


def run_phase1_and_king(workdir: str, prompts: str, teacher_cache: str,
                        king: dict, out_path: str) -> int:
    """Run Phase 1 (teacher gen) + king scoring on GPU 0.

    The king is scored first because (a) the policy re-evals the king every
    round and (b) populating teacher_cache.pt is a Phase-1 side-effect we
    want done exactly once before workers fan out.
    """
    env = _shared_env()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = _common_args(workdir, prompts, teacher_cache) + [
        "--students", king["model"],
        "--revisions", king["revision"],
        "--king", king["model"],
        "--output", out_path,
    ]
    print(f"[orch] phase1+king cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    log = open(os.path.join(workdir, "phase1_king.log"), "w")
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[orch] phase1+king pid={proc.pid} (logging to phase1_king.log)", flush=True)
    return proc.pid


def spawn_worker(gpu_idx: int, workdir: str, prompts: str, teacher_cache: str,
                 shard: list[dict], shard_out: str, shard_log: str) -> subprocess.Popen:
    """Spawn one Phase-2 student worker pinned to a single GPU.

    All workers share cwd=/home/dev_eval (so pod_eval's imports + scripts/v31/
    resolve correctly). Per-worker output paths sit in a per-GPU subdir so
    eval_progress.json doesn't collide (pod_eval writes it to dirname(--output)).
    """
    env = _shared_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    students_csv = ",".join(s["model"] for s in shard)
    revisions_csv = ",".join(s["revision"] for s in shard)
    cmd = _common_args(workdir, prompts, teacher_cache) + [
        "--students", students_csv,
        "--revisions", revisions_csv,
        "--output", shard_out,
        "--resume",
    ]
    log = open(shard_log, "w")
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[orch] gpu{gpu_idx} worker pid={proc.pid} students={students_csv}", flush=True)
    return proc


def _spawn_challenger_workers(workdir: str, prompts: str, teacher_cache: str,
                              chals: list[dict], n_gpus: int) -> list:
    """Fan out N-1 challenger workers (skipping GPU 0 which is held by king).

    Used when teacher_cache lands mid-king so we can start challengers in
    parallel with king's bench battery instead of waiting.
    """
    n_workers = max(1, n_gpus - 1)
    shards: list[list[dict]] = [[] for _ in range(n_workers)]
    for i, c in enumerate(chals):
        shards[i % n_workers].append(c)
    procs = []
    for k in range(n_workers):
        if not shards[k]:
            continue
        gpu_idx = k + 1
        out_dir = os.path.join(workdir, f"gpu{gpu_idx}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, "shard_result.json")
        log_path = os.path.join(workdir, f"shard_gpu{gpu_idx}.log")
        proc = spawn_worker(gpu_idx, workdir, prompts, teacher_cache,
                            shards[k], out_path, log_path)
        procs.append((gpu_idx, proc, out_path, log_path))
        time.sleep(5)
    return procs


def merge_results(king_path: str, shard_paths: list[str], out_path: str) -> dict:
    """Merge king_result.json + shard_*.json into a single eval_results.json."""
    merged = {"students": {}}
    for p in [king_path, *shard_paths]:
        if not os.path.exists(p):
            print(f"[orch] WARN missing result file: {p}", flush=True)
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[orch] WARN cannot parse {p}: {e}", flush=True)
            continue
        for k, v in d.items():
            if k == "students":
                merged["students"].update(v or {})
            elif k not in merged:
                merged[k] = v
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/home/dev_eval")
    ap.add_argument("--prompts", default="/home/dev_eval/prompts.json")
    ap.add_argument("--teacher-cache", default="/home/dev_eval/teacher_cache.pt")
    ap.add_argument("--queue", default=None,
                    help="Path to JSON queue file with king + challengers. "
                         "Default: pull from https://api.arbos.life/api/queue.")
    ap.add_argument("--gpus", type=int, default=8,
                    help="Total GPUs to use (1 for king, N-1 for challengers).")
    ap.add_argument("--out", default="/home/dev_eval/full_round_results.json")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Assume teacher_cache.pt + king result already exist; "
                         "go straight to challenger fan-out.")
    args = ap.parse_args()

    workdir = args.workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # ── Resolve king + challengers ──────────────────────────────────────
    if args.queue and os.path.exists(args.queue):
        with open(args.queue) as f:
            queue = json.load(f)
    else:
        import urllib.request
        with urllib.request.urlopen("https://api.arbos.life/api/queue", timeout=30) as r:
            queue = json.loads(r.read().decode())

    slots = queue.get("slots", []) or []
    king = next((s for s in slots if s.get("role") == "king"), None)
    challengers = [s for s in slots if s.get("role") == "challenger"]
    if not king:
        print("[orch] FATAL: no king in queue", flush=True)
        sys.exit(1)
    print(f"[orch] king: uid={king.get('uid')} model={king.get('model')} "
          f"rev={(king.get('revision') or '')[:12]}", flush=True)
    print(f"[orch] challengers: {len(challengers)}", flush=True)

    # Normalize records: model + revision (default "main" if missing).
    def _norm(s):
        return {"model": s["model"], "revision": s.get("revision") or "main", "uid": s.get("uid")}
    king_n = _norm(king)
    chals_n = [_norm(c) for c in challengers]

    # 2026-05-15: tell every worker which models the *whole round* needs so the
    # per-worker pre-round HF cache sweep in pod_eval.py preserves sibling-shard
    # models. Without this, worker 1 (shard = [a,b]) would delete worker 2's
    # cached student (shard = [c,d]) at startup, then worker 2 would have to
    # re-download 30 GB.
    os.environ["DISTIL_CACHE_KEEP_MODELS"] = ",".join(
        [king_n["model"]] + [c["model"] for c in chals_n]
    )

    n_gpus = max(1, args.gpus)
    king_path = os.path.join(workdir, "king_result.json")

    if args.skip_phase1 and not os.path.exists(args.teacher_cache):
        print(f"[orch] FATAL: --skip-phase1 but no teacher_cache at {args.teacher_cache}", flush=True)
        sys.exit(2)

    # ── Phase 1 + king (only if cache missing) ─────────────────────────
    if not args.skip_phase1 and not os.path.exists(args.teacher_cache):
        if os.path.exists(king_path):
            os.remove(king_path)
        pid = run_phase1_and_king(workdir, args.prompts, args.teacher_cache, king_n, king_path)
        # Wait until teacher_cache hits disk (Phase 1 done), then move on
        # to fan-out IMMEDIATELY — don't wait for king's bench battery.
        # King keeps running on GPU 0; challengers fan across the remaining GPUs.
        while True:
            try:
                ret = os.waitpid(pid, os.WNOHANG)
                if ret != (0, 0):
                    exit_code = os.WEXITSTATUS(ret[1])
                    print(f"[orch] phase1+king exit_code={exit_code}", flush=True)
                    break
            except ChildProcessError:
                print("[orch] phase1+king child gone", flush=True)
                break
            time.sleep(30)
            tc_size = os.path.getsize(args.teacher_cache) if os.path.exists(args.teacher_cache) else 0
            kr_size = os.path.getsize(king_path) if os.path.exists(king_path) else 0
            if tc_size > 1_000_000 and not _CHALLENGERS_SPAWNED.get("done"):
                print(f"[orch] teacher_cache landed ({tc_size/1024/1024:.0f}MB) — "
                      f"fanning out challengers in parallel while king finishes",
                      flush=True)
                _CHALLENGERS_SPAWNED["procs"] = _spawn_challenger_workers(
                    workdir, args.prompts, args.teacher_cache, chals_n, n_gpus,
                )
                _CHALLENGERS_SPAWNED["done"] = True
            print(f"[orch] phase1+king alive… teacher_cache={tc_size/1024/1024:.0f}MB "
                  f"king_result={kr_size}B", flush=True)
        if not os.path.exists(args.teacher_cache):
            print("[orch] FATAL: teacher_cache.pt was not produced", flush=True)
            sys.exit(2)

    # ── Phase 2: if we already had a cached teacher and skipped Phase 1,
    #            spawn king + challengers all in parallel right now ─────
    if args.skip_phase1 or not _CHALLENGERS_SPAWNED.get("done"):
        # King + all challengers, every one on its own GPU.
        all_students = [king_n] + chals_n
        # GPU 0 = king (always); GPUs 1..n_gpus-1 split the challengers.
        n_chal_gpus = max(1, n_gpus - 1)
        shards: list[list[dict]] = [[king_n]] + [[] for _ in range(n_chal_gpus)]
        for i, c in enumerate(chals_n):
            shards[1 + (i % n_chal_gpus)].append(c)
        workers = []
        shard_paths = []
        for k, shard in enumerate(shards):
            if not shard:
                continue
            gpu_idx = k
            out_dir = os.path.join(workdir, f"gpu{gpu_idx}")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(out_dir, "shard_result.json")
            log_path = os.path.join(workdir, f"shard_gpu{gpu_idx}.log")
            shard_paths.append(out_path)
            proc = spawn_worker(gpu_idx, workdir, args.prompts, args.teacher_cache,
                                shard, out_path, log_path)
            workers.append((gpu_idx, proc, out_path, log_path))
            time.sleep(5)
    else:
        workers = _CHALLENGERS_SPAWNED["procs"]
        shard_paths = [w[2] for w in workers] + [king_path]

    # ── Wait + monitor ─────────────────────────────────────────────────
    # 2026-05-15: per-worker watchdog. The vLLM v1 EngineCore subprocess
    # occasionally hangs on shutdown (leaks a multiprocessing semaphore and
    # gets stuck in cleanup), pinning the parent pod_eval.py at 0% CPU /
    # 65GB VRAM held indefinitely. Without a watchdog the round can sit on
    # 1-2 wedged workers for 30+ min while the other shards stream done.
    # We watch each worker's log file mtime; if it stops growing for
    # WATCHDOG_STALE_S seconds while proc.poll() still reports alive, we
    # SIGKILL the process *and its EngineCore children* and let the merge
    # proceed with whatever the worker had already written.
    WATCHDOG_STALE_S = int(os.environ.get("DISTIL_ORCH_WATCHDOG_S", "360"))
    print(f"[orch] {len(workers)} workers spawned; monitoring "
          f"(watchdog={WATCHDOG_STALE_S}s, Ctrl-C to abort)…",
          flush=True)
    pending = list(workers)
    last_progress: dict[int, float] = {}
    last_log_size: dict[int, int] = {}
    now0 = time.time()
    for gpu_idx, _proc, _out, log_path in pending:
        last_progress[gpu_idx] = now0
        last_log_size[gpu_idx] = (
            os.path.getsize(log_path) if os.path.exists(log_path) else 0
        )
    while pending:
        time.sleep(60)
        still = []
        for gpu_idx, proc, out_path, log_path in pending:
            ret = proc.poll()
            if ret is None:
                sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                cur_log_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
                if cur_log_size > last_log_size.get(gpu_idx, 0):
                    last_log_size[gpu_idx] = cur_log_size
                    last_progress[gpu_idx] = time.time()
                stale_for = time.time() - last_progress.get(gpu_idx, time.time())
                if stale_for > WATCHDOG_STALE_S:
                    print(
                        f"[orch] WATCHDOG: gpu{gpu_idx} pid={proc.pid} silent "
                        f"for {int(stale_for)}s (limit={WATCHDOG_STALE_S}s) — "
                        f"SIGKILL + reaping EngineCore children. "
                        f"shard_result.json on disk = {sz}B.",
                        flush=True,
                    )
                    try:
                        # SIGKILL the parent first so it stops respawning helpers.
                        proc.kill()
                    except Exception:
                        pass
                    # Now reap any orphaned EngineCore / vLLM workers tied to
                    # this GPU. pkill-by-cwd is too coarse (would hit other
                    # shards); -P limits to children of the dead parent.
                    try:
                        subprocess.run(
                            ["pkill", "-9", "-P", str(proc.pid)],
                            check=False, capture_output=True, timeout=10,
                        )
                    except Exception:
                        pass
                    # Belt+suspenders: kill any python holding this GPU.
                    try:
                        subprocess.run(
                            ["bash", "-lc",
                             "for p in $(nvidia-smi --query-compute-apps=pid "
                             f"--format=csv,noheader -i {gpu_idx} | tr -d ' '); "
                             "do kill -9 $p 2>/dev/null; done"],
                            check=False, capture_output=True, timeout=10,
                        )
                    except Exception:
                        pass
                    continue  # drop from `still` — treat as exited
                still.append((gpu_idx, proc, out_path, log_path))
                # Tail the log to grab the latest stage line for visibility.
                tail = ""
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "rb") as f:
                            f.seek(0, os.SEEK_END)
                            end = f.tell()
                            f.seek(max(0, end - 4096))
                            chunk = f.read().decode("utf-8", errors="ignore").splitlines()
                            tail = next(
                                (ln for ln in reversed(chunk)
                                 if "[bench" in ln or "[eval]" in ln
                                 or "Student" in ln or "stage" in ln),
                                "",
                            )[:140]
                    except Exception:
                        pass
                print(
                    f"[orch] gpu{gpu_idx} pid={proc.pid} alive  out={sz}B  "
                    f"stale={int(stale_for)}s  {tail}",
                    flush=True,
                )
            else:
                print(f"[orch] gpu{gpu_idx} pid={proc.pid} EXITED rc={ret}  out_exists={os.path.exists(out_path)}",
                      flush=True)
        pending = still

    # ── Merge ──────────────────────────────────────────────────────────
    print("[orch] all workers finished; merging…", flush=True)
    merged = merge_results(king_path, shard_paths, args.out)
    n_students = len(merged.get("students", {}))
    print(f"[orch] merged {n_students} students into {args.out}", flush=True)
    return


if __name__ == "__main__":
    main()
