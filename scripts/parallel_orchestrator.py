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


def _read_shard_progress(shard_out: str) -> dict:
    """Pull the live ``eval_progress.json`` a worker writes alongside
    its shard_result.json. pod_eval.py writes this every few seconds
    with the current student / stage / prompts_done — exactly what the
    dashboard needs to render per-GPU progress for the parallel run.
    """
    # pod_eval.py writes eval_progress.json into the same directory as
    # its --output (shard_result.json). See ``scripts/eval_progress_io.py``.
    if not shard_out:
        return {}
    prog_path = os.path.join(os.path.dirname(shard_out), "eval_progress.json")
    if not os.path.exists(prog_path):
        return {}
    try:
        with open(prog_path) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_unified_progress(
    workdir: str,
    unified_path: str,
    shard_progress: dict,
    n_students_target: int,
    finished: bool = False,
) -> None:
    """Aggregate per-shard ``eval_progress.json`` into a single
    pod-level ``eval_progress.json`` the validator can scrape.

    Schema mirrors the single-shard format pod_eval.py writes so the
    validator's ``poll_pod_progress`` loop sees a familiar shape, with
    an extra ``shards`` array carrying per-GPU detail that the dashboard
    can fan out into "N students scoring simultaneously" tiles.
    """
    if not unified_path:
        return
    shards: list[dict] = []
    completed: list[dict] = []
    teacher_done = 0
    teacher_total = 0
    n_prompts_total = 0
    run_started_at = None
    current_student = None
    current_stage = None
    current_prompts_done = 0
    for gpu_idx in sorted(shard_progress.keys()):
        meta = dict(shard_progress[gpu_idx])
        shard_out = meta.pop("shard_out", "")
        live = _read_shard_progress(shard_out)
        cur = (live.get("current") or {}) if live else {}
        if cur:
            meta["current_student"] = cur.get("student_name")
            meta["current_stage"] = cur.get("stage")
            meta["current_prompts_done"] = cur.get("prompts_done", 0)
            meta["current_prompts_total"] = live.get("prompts_total", 0)
            # The "last started" shard's current_student doubles as the
            # legacy single-shard ``current`` for the dashboard's compat
            # mode (so even a frontend that doesn't know about shards
            # sees a sensible "scoring X" line).
            current_student = cur.get("student_name") or current_student
            current_stage = cur.get("stage") or current_stage
            current_prompts_done = max(
                current_prompts_done, int(cur.get("prompts_done") or 0),
            )
        if live:
            for c in live.get("completed", []) or []:
                completed.append(c)
            teacher_done = max(teacher_done, int(live.get("teacher_prompts_done") or 0))
            teacher_total = max(teacher_total, int(live.get("n_teacher_prompts_total") or 0))
            n_prompts_total = max(n_prompts_total, int(live.get("prompts_total") or 0))
            run_started_at = run_started_at or live.get("run_started_at")
        shards.append(meta)
    # De-dup completed entries (king + each shard writes its own copy of
    # any students it scored; the orchestrator's view is the union).
    seen = set()
    completed_uniq: list[dict] = []
    for c in completed:
        name = c.get("student_name")
        if not name or name in seen:
            continue
        seen.add(name)
        completed_uniq.append(c)
    # ``students_done`` drives the dashboard tile + the "0/N students"
    # bug the user just saw. With parallel runs we count completed
    # students across all shards.
    students_done = len(completed_uniq)
    phase = "scoring" if not finished else "composite"
    if students_done == 0 and teacher_total > 0 and teacher_done < teacher_total:
        phase = "teacher_generate"
    payload = {
        "active": not finished,
        "phase": phase,
        "current": {
            "student_name": current_student,
            "stage": current_stage,
            "prompts_done": current_prompts_done,
        } if current_student else None,
        "completed": completed_uniq,
        "students_done": students_done,
        "students_total": n_students_target,
        "n_students": n_students_target,
        "prompts_total": n_prompts_total,
        "teacher_prompts_done": teacher_done,
        "n_teacher_prompts_total": teacher_total,
        "run_started_at": run_started_at,
        # Parallel-mode extension. The dashboard renders this as a
        # multi-row "N students scoring simultaneously" tile; older
        # frontends ignore the extra key.
        "shards": shards,
        "n_gpus": len(shards),
        "orchestrator": "parallel",
    }
    tmp = unified_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, unified_path)
    except Exception as exc:
        print(f"[orch] WARN failed to write unified progress: {exc}", flush=True)


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


def merge_eval_data(workdir: str, out_path: str) -> None:
    """Merge per-shard ``eval_data.json`` files into ``<workdir>/eval_data.json``.

    pod_session.py downloads ``<run_dir>/eval_data.json`` after the
    round to drive the public prompt-level dataset and the private
    audit copy. With parallel fan-out each shard writes its own
    ``<run_dir>/gpu{k}/eval_data.json`` (same prompts, different
    students). The download will 404 if the top-level file isn't
    there, so we concatenate the unique rows from each shard.
    """
    out = {"prompts": None, "students": {}, "data": []}
    seen_prompts_hash = None
    # Look for shard dirs that contain eval_data.json.
    for entry in sorted(os.listdir(workdir)):
        sub = os.path.join(workdir, entry)
        if not os.path.isdir(sub):
            continue
        p = os.path.join(sub, "eval_data.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as exc:
            print(f"[orch] WARN cannot parse eval_data {p}: {exc}", flush=True)
            continue
        # The eval_data schema isn't perfectly stable across pod_eval
        # versions (sometimes a list-of-rows, sometimes a dict with
        # "data" key holding the list). Handle both.
        rows = d if isinstance(d, list) else (d.get("data") if isinstance(d, dict) else None)
        if isinstance(rows, list):
            # First shard's full eval_data is canonical (same prompts
            # across shards). Subsequent shards may add per-student
            # rows we haven't seen yet.
            if not out["data"]:
                out["data"] = rows
                if isinstance(d, dict):
                    for k in ("prompts", "effective_prompts_hash"):
                        if k in d:
                            out[k] = d[k]
                            if k == "effective_prompts_hash":
                                seen_prompts_hash = d[k]
            else:
                # Same prompts assumed — append any new per-student data
                # if it diverges. For now we just keep the first.
                pass
    final_path = out_path
    try:
        with open(final_path, "w") as f:
            json.dump(out, f)
        print(f"[orch] wrote merged eval_data ({len(out.get('data') or [])} rows) "
              f"to {final_path}", flush=True)
    except Exception as exc:
        print(f"[orch] WARN failed to write merged eval_data: {exc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/home/dev_eval")
    ap.add_argument("--prompts", default="/home/dev_eval/prompts.json")
    ap.add_argument("--teacher-cache", default="/home/dev_eval/teacher_cache.pt")
    ap.add_argument("--queue", default=None,
                    help="Path to JSON queue file with king + challengers. "
                         "Default: pull from https://api.arbos.life/api/queue.")
    ap.add_argument("--king-model", default=None,
                    help="Inline king model (alternative to --queue). "
                         "Use with --king-revision + --students/--revisions.")
    ap.add_argument("--king-revision", default="main")
    ap.add_argument("--students", default=None,
                    help="Comma-separated challenger models (inline mode).")
    ap.add_argument("--revisions", default=None,
                    help="Comma-separated challenger revisions matching --students.")
    ap.add_argument("--gpus", type=int, default=8,
                    help="Total GPUs to use (1 for king, N-1 for challengers).")
    ap.add_argument("--out", default="/home/dev_eval/full_round_results.json")
    ap.add_argument("--unified-progress", default=None,
                    help="Path to write aggregated multi-shard "
                         "eval_progress.json (so the validator's poll loop "
                         "sees parallel runs the same way as single-shard).")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Assume teacher_cache.pt + king result already exist; "
                         "go straight to challenger fan-out.")
    args = ap.parse_args()
    if args.unified_progress is None:
        args.unified_progress = os.path.join(args.workdir, "eval_progress.json")

    workdir = args.workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # ── Resolve king + challengers ──────────────────────────────────────
    # Three modes (in order of preference):
    #   1. ``--king-model`` + ``--students`` (CLI inline) — used by the
    #      validator's pod_session.py which already has the canonical
    #      models_to_eval dict and doesn't want a network round-trip.
    #   2. ``--queue <file>`` (JSON file with ``slots``) — used by
    #      operators running the orchestrator manually with a snapshot.
    #   3. Fetch from https://api.arbos.life/api/queue — default, but
    #      racy because the queue can change between the validator's
    #      decision and the orchestrator's fetch.
    if args.king_model:
        s_models = [s for s in (args.students or "").split(",") if s.strip()]
        s_revs_raw = [s for s in (args.revisions or "").split(",")]
        if s_revs_raw and len(s_revs_raw) < len(s_models):
            s_revs_raw = s_revs_raw + ["main"] * (len(s_models) - len(s_revs_raw))
        s_revs = s_revs_raw or ["main"] * len(s_models)
        king = {"role": "king", "model": args.king_model,
                "revision": args.king_revision or "main"}
        challengers = [
            {"role": "challenger", "model": m, "revision": (r or "main").strip()}
            for m, r in zip(s_models, s_revs)
        ]
    else:
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
    # Phase 1/king writes to its own gpu0 subdir so that pod_eval.py's
    # ``Path(args.output).with_name("eval_done.marker")`` lands in
    # ``<workdir>/gpu0/eval_done.marker`` instead of the top-level
    # ``<workdir>/eval_done.marker`` that the validator polls. Before this
    # fix, Phase 1 finishing tripped DISTIL_STATUS:done while the
    # challenger shards (and the merge step) were still running — the
    # validator then tried to download eval_results.json (not written
    # yet), failed, and cleared the round. Only THIS orchestrator should
    # write the top-level marker, and only after merge completes.
    gpu0_dir = os.path.join(workdir, "gpu0")
    os.makedirs(gpu0_dir, exist_ok=True)
    king_path = os.path.join(gpu0_dir, "king_result.json")
    # Defensive: clear any stale top-level marker. The validator polls this
    # via DISTIL_STATUS:done; any premature marker (e.g. a previous failed
    # run) would race us into the validator's results-download path.
    try:
        top_marker = os.path.join(workdir, "eval_done.marker")
        if os.path.exists(top_marker):
            os.unlink(top_marker)
    except Exception:
        pass

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
    # We watch each worker's log file: if it stops growing for
    # WATCHDOG_STALE_S seconds, or grows ONLY by repeating the same line
    # (the voidai001/v4 infinite-encode pattern), we SIGKILL the process
    # *and its EngineCore children* and let the merge proceed with
    # whatever the worker had already written.
    WATCHDOG_STALE_S = int(os.environ.get("DISTIL_ORCH_WATCHDOG_S", "360"))
    # Number of consecutive identical tail lines that means "infinite
    # loop, not real progress". 100 is conservative — at ~10ms/line that's
    # 1s of work; combined with the 60s poll interval, we trip after
    # ~6000 repeats which is unambiguously a wedged tokenizer/encode
    # loop and not a slow legit phase.
    WATCHDOG_REPEAT_N = int(os.environ.get("DISTIL_ORCH_WATCHDOG_REPEAT_N", "100"))
    print(f"[orch] {len(workers)} workers spawned; monitoring "
          f"(watchdog stale={WATCHDOG_STALE_S}s repeat_n={WATCHDOG_REPEAT_N}, "
          f"Ctrl-C to abort)…",
          flush=True)

    def _kill_worker(gpu_idx: int, proc, reason: str, out_path: str):
        sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(
            f"[orch] WATCHDOG: gpu{gpu_idx} pid={proc.pid} {reason} — "
            f"SIGKILL + reaping EngineCore children. "
            f"shard_result.json on disk = {sz}B.",
            flush=True,
        )
        try:
            proc.kill()
        except Exception:
            pass
        try:
            subprocess.run(
                ["pkill", "-9", "-P", str(proc.pid)],
                check=False, capture_output=True, timeout=10,
            )
        except Exception:
            pass
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

    def _tail_summary(log_path: str) -> tuple[str, int, str]:
        """Return (latest stage line, repeated-tail count, recent log slice)."""
        if not os.path.exists(log_path):
            return "", 0, ""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                # Read last 64 KB — enough to see >1000 repeated lines
                # of the infinite-encode pattern AND any [eval]/[bench]
                # progress line that landed shortly before.
                f.seek(max(0, end - 65536))
                chunk = f.read().decode("utf-8", errors="ignore")
        except Exception:
            return "", 0, ""
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return "", 0, ""
        last = lines[-1]
        repeat = 0
        for ln in reversed(lines):
            if ln == last:
                repeat += 1
            else:
                break
        stage = next(
            (ln for ln in reversed(lines)
             if "[bench" in ln or "[eval]" in ln
             or "Student" in ln or "stage" in ln),
            last,
        )[:140]
        return stage, repeat, chunk

    pending = list(workers)
    last_progress: dict[int, float] = {}
    last_log_size: dict[int, int] = {}
    now0 = time.time()
    for gpu_idx, _proc, _out, log_path in pending:
        last_progress[gpu_idx] = now0
        last_log_size[gpu_idx] = (
            os.path.getsize(log_path) if os.path.exists(log_path) else 0
        )
    # Track per-shard progress snapshots for the aggregated progress file.
    shard_progress: dict[int, dict] = {}
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
                tail, repeat, _chunk = _tail_summary(log_path)
                # Hard stall: log frozen for too long.
                if stale_for > WATCHDOG_STALE_S:
                    _kill_worker(gpu_idx, proc,
                                 f"silent for {int(stale_for)}s "
                                 f"(limit={WATCHDOG_STALE_S}s)",
                                 out_path)
                    continue
                # Soft stall: log growing only via a single repeating
                # line (e.g. the voidai001/v4 infinite "super().encode"
                # loop). This pattern doesn't trip the size watchdog.
                if repeat >= WATCHDOG_REPEAT_N:
                    _kill_worker(
                        gpu_idx, proc,
                        f"log stuck on a single repeating line "
                        f"({repeat}× same tail; likely tokenizer/encode "
                        f"infinite loop): {tail[:80]!r}",
                        out_path,
                    )
                    continue
                still.append((gpu_idx, proc, out_path, log_path))
                print(
                    f"[orch] gpu{gpu_idx} pid={proc.pid} alive  out={sz}B  "
                    f"stale={int(stale_for)}s  repeat={repeat}  {tail}",
                    flush=True,
                )
                shard_progress[gpu_idx] = {
                    "gpu": gpu_idx, "pid": proc.pid, "alive": True,
                    "stage_line": tail, "stale_s": int(stale_for),
                    "repeat_tail": repeat, "shard_result_bytes": sz,
                    "shard_out": out_path,
                }
            else:
                print(f"[orch] gpu{gpu_idx} pid={proc.pid} EXITED rc={ret}  "
                      f"out_exists={os.path.exists(out_path)}",
                      flush=True)
                shard_progress[gpu_idx] = {
                    "gpu": gpu_idx, "pid": proc.pid, "alive": False,
                    "exit_code": ret, "shard_out": out_path,
                    "shard_result_bytes": (
                        os.path.getsize(out_path) if os.path.exists(out_path) else 0
                    ),
                }
        pending = still
        _write_unified_progress(
            workdir, args.unified_progress, shard_progress,
            n_students_target=len([king_n] + chals_n),
        )

    # ── Merge ──────────────────────────────────────────────────────────
    print("[orch] all workers finished; merging…", flush=True)
    merged = merge_results(king_path, shard_paths, args.out)
    n_students = len(merged.get("students", {}))
    print(f"[orch] merged {n_students} students into {args.out}", flush=True)
    # Merge per-shard eval_data.json into the top-level one the
    # validator downloads via ``eval_data_remote``.
    try:
        merge_eval_data(workdir, os.path.join(workdir, "eval_data.json"))
    except Exception as exc:
        print(f"[orch] WARN eval_data merge failed: {exc}", flush=True)
    # Final progress write: mark the round done so the validator's poll
    # loop sees ``active=false`` and proceeds to download results.
    _write_unified_progress(
        workdir, args.unified_progress, shard_progress,
        n_students_target=len([king_n] + chals_n),
        finished=True,
    )
    # 2026-05-15: write the eval_done.marker the validator's
    # status_cmd polls for ("DISTIL_STATUS:done"). Without this the
    # validator's status loop never sees the run finish and the
    # outer ``run_eval_on_pod`` keeps polling until the per-round
    # 8h timeout, at which point it kills our pod_eval children
    # AND throws away the merged results. The marker name matches
    # what pod_eval.py writes (``Path(args.output).with_name("eval_done.marker")``).
    try:
        marker_path = os.path.join(os.path.dirname(args.out), "eval_done.marker")
        with open(marker_path, "w") as f:
            f.write(f"orchestrator merged {n_students} students at {time.time():.0f}\n")
        print(f"[orch] wrote done marker {marker_path}", flush=True)
    except Exception as exc:
        print(f"[orch] WARN failed to write done marker: {exc}", flush=True)
    return


def _safe_main():
    """Wrap main() so any uncaught error still trips the validator's
    DISTIL_STATUS:done watcher (with a failure-flavored marker + an
    empty results file). Without this, an exception inside Phase 1
    or the merge step leaves the validator polling for ~8h before
    timing out — and discarding any partial shard results."""
    # Compute the marker / results paths from argv BEFORE main runs,
    # so we can still write them if main blows up early.
    fallback_marker = None
    fallback_results = None
    try:
        # cheap pre-parse so we can find --out without duplicating
        # argparse logic. We only care about --out and --workdir.
        out_path = "/home/dev_eval/full_round_results.json"
        wd_path = "/home/dev_eval"
        i = 0
        while i < len(sys.argv) - 1:
            if sys.argv[i] == "--out":
                out_path = sys.argv[i + 1]
            elif sys.argv[i] == "--workdir":
                wd_path = sys.argv[i + 1]
            i += 1
        fallback_marker = os.path.join(os.path.dirname(out_path), "eval_done.marker")
        fallback_results = out_path
        _ = wd_path
    except Exception:
        pass
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("[orch] interrupted; writing failure marker", flush=True)
        if fallback_marker:
            try:
                with open(fallback_marker, "w") as f:
                    f.write(f"aborted by interrupt at {time.time():.0f}\n")
            except Exception:
                pass
        sys.exit(130)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        print(f"[orch] FATAL: {type(exc).__name__}: {exc}", flush=True)
        # Make sure the validator's download step gets *something*.
        if fallback_results and not os.path.exists(fallback_results):
            try:
                with open(fallback_results, "w") as f:
                    json.dump({"students": {}, "error": str(exc)[:500]}, f)
            except Exception:
                pass
        if fallback_marker:
            try:
                with open(fallback_marker, "w") as f:
                    f.write(f"failed: {type(exc).__name__}: {str(exc)[:200]}\n")
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    _safe_main()
