"""Multi-GPU parallel student-eval orchestrator (pod-side).

The single-process ``python -m distil.pod`` runs every phase serially
on one GPU. On the 8xB200 pod we have enough memory and compute to run
the per-student phase in parallel — that's how prod gets ~47-min rounds.

Design (mirrors ``scripts/parallel_orchestrator.py`` but in 250 LoC
instead of 923):

  Phase 1 (one process, GPU 0):
    python -m distil.pod spec.json --phase teacher
       └─ generates teacher continuations + saves the teacher_cache.

  Phase 2 (N processes in parallel, one per GPU):
    GPU 0:  python -m distil.pod spec.json --phase students --shard 0/N
    GPU 1:  python -m distil.pod spec.json --phase students --shard 1/N
    ...
       └─ each worker reads the shared teacher_cache, runs its slice of
          students, and writes shard_<k>.json + shard_<k>.raw.json. The
          orchestrator tails every worker's log and arms a
          :class:`LineStallDetector` per shard so a hung student
          (diffuznik-style tokenizer loop) is killed in seconds rather
          than wedging the whole round.

  Phase 3 (one process, GPU 0):
    Merge shard_*.json → results.json, merge raw → raw_merged.json.
    python -m distil.pod spec.json --phase judge --raw-in raw_merged.json
       └─ teacher reloads exactly once, grades every student's collected
          judge / long-form / chat-turns responses, merges scores back.

While Phase 2 runs we publish a unified ``eval_progress.json`` with a
``shards[]`` array so the dashboard can render N concurrent students.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from distil.pod.watchdog import LineStallDetector

logger = logging.getLogger("distil.pod.orchestrator")

POLL_INTERVAL_S = 3.0
SHARD_STALL_AFTER_S = 240.0  # 4 minutes silence ⇒ kill the shard.
SHARD_REPEAT_WINDOW = 16


def _python() -> str:
    return os.environ.get("DISTIL_PYTHON") or sys.executable or "python3"


def _spawn_phase(
    *,
    spec_path: Path,
    workdir: Path,
    phase: str,
    out: Path | None = None,
    progress: Path | None = None,
    gpu: int | None = None,
    shard: str | None = None,
    raw_in: Path | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Subprocess inherits env but not the parent's CWD-derived sys.path.
    # The orchestrator is normally launched from `/home/` (where `distil/`
    # lives as a top-level package) but each phase subprocess gets
    # ``cwd=workdir`` (e.g. /home/distil_eval/round_X), which has no
    # `distil/` on its sys.path. Propagate the parent of the `distil`
    # package via PYTHONPATH so the children can ``-m distil.pod`` cleanly.
    import distil as _distil_pkg

    distil_parent = str(Path(_distil_pkg.__file__).resolve().parents[1])
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        distil_parent + (os.pathsep + existing_pp if existing_pp else "")
    )
    cmd = [_python(), "-u", "-m", "distil.pod", str(spec_path), "--phase", phase]
    if out is not None:
        cmd += ["--out", str(out)]
    if progress is not None:
        cmd += ["--progress", str(progress)]
    if shard is not None:
        cmd += ["--shard", shard]
    if raw_in is not None:
        cmd += ["--raw-in", str(raw_in)]
    log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
    logger.info(f"spawn {' '.join(cmd)} (gpu={gpu}, log={log_path})")
    return subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log_fh, stderr=subprocess.STDOUT)


def _tail_log_into(detector: LineStallDetector, log_path: Path, stop: threading.Event) -> None:
    """Background thread: append-tail ``log_path`` into ``detector``."""
    pos = 0
    while not stop.is_set():
        try:
            if log_path.exists():
                with log_path.open() as fh:
                    fh.seek(pos)
                    for line in fh:
                        detector.observe(line)
                    pos = fh.tell()
        except Exception:
            pass
        time.sleep(2.0)


def _read_shard_progress(progress_path: Path) -> dict[str, Any]:
    try:
        if progress_path.exists():
            return json.loads(progress_path.read_text()) or {}
    except Exception:
        return {}
    return {}


def _write_unified_progress(
    path: Path, shard_state: dict[int, dict], n_students_total: int, finished: bool
) -> None:
    shards: list[dict[str, Any]] = []
    students_done = 0
    for gpu_idx in sorted(shard_state):
        s = shard_state[gpu_idx]
        live = _read_shard_progress(s["progress"]) or {}
        shards.append(
            {
                "gpu": gpu_idx,
                "pid": s.get("pid"),
                "alive": s.get("alive", True),
                "current_student": live.get("current_student"),
                "current_stage": live.get("phase"),
                "current_prompts_done": int(live.get("prompts_done") or 0),
                "current_prompts_total": int(live.get("prompts_total") or 0),
                "stale_s": int(s.get("stale_s", 0)),
                "repeat_tail": int(s.get("repeat_tail", 0)),
                "exit_code": s.get("exit_code"),
                "shard_result_bytes": (
                    s["out"].stat().st_size if Path(s["out"]).exists() else 0
                ),
            }
        )
        students_done += int(live.get("students_done") or 0)
    payload = {
        "active": not finished,
        "phase": "scoring" if not finished else "finished",
        "students_done": students_done,
        "students_total": n_students_total,
        "shards": shards,
        "n_gpus": len(shards),
        "orchestrator": "distil.pod.orchestrator",
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _merge_shards(shard_outs: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in shard_outs:
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            logger.warning(f"failed to read {path}: {exc}")
            continue
        for k, v in doc.items():
            if k.startswith("__"):
                continue
            merged[k] = v
    return merged


def _merge_raw(raw_paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in raw_paths:
        if not path.exists():
            continue
        try:
            merged.update(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning(f"failed to read raw {path}: {exc}")
    return merged


def run(
    *, spec_path: Path, workdir: Path, out: Path, progress: Path, n_gpus: int = 8
) -> int:
    spec = json.loads(spec_path.read_text())
    n_students = len(spec.get("students", []))
    workdir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: teacher generation on GPU 0 ────────────────────────
    logger.info("phase 1: teacher continuations (GPU 0)")
    teacher_log = workdir / "phase1_teacher.log"
    proc = _spawn_phase(
        spec_path=spec_path,
        workdir=workdir,
        phase="teacher",
        progress=progress,
        gpu=0,
        log_path=teacher_log,
    )
    rc = proc.wait()
    if rc != 0:
        logger.error(f"phase 1 (teacher) failed rc={rc}; see {teacher_log}")
        return rc

    # ── Phase 2: fan out N shards in parallel ───────────────────────
    logger.info(f"phase 2: {n_gpus} shards in parallel")
    shard_state: dict[int, dict[str, Any]] = {}
    detectors: dict[int, LineStallDetector] = {}
    stops: list[threading.Event] = []
    threads: list[threading.Thread] = []
    for gpu_idx in range(n_gpus):
        shard_dir = workdir / f"gpu{gpu_idx}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_path = shard_dir / "shard.json"
        prog_path = shard_dir / "eval_progress.json"
        log_path = shard_dir / "shard.log"
        proc = _spawn_phase(
            spec_path=spec_path,
            workdir=workdir,
            phase="students",
            out=out_path,
            progress=prog_path,
            shard=f"{gpu_idx}/{n_gpus}",
            gpu=gpu_idx,
            log_path=log_path,
        )
        det = LineStallDetector(
            stale_after_s=SHARD_STALL_AFTER_S, repeat_window=SHARD_REPEAT_WINDOW
        )
        detectors[gpu_idx] = det
        stop = threading.Event()
        stops.append(stop)
        t = threading.Thread(
            target=_tail_log_into, args=(det, log_path, stop), daemon=True
        )
        t.start()
        threads.append(t)
        shard_state[gpu_idx] = {
            "proc": proc,
            "pid": proc.pid,
            "out": out_path,
            "raw": out_path.with_suffix(".raw.json"),
            "progress": prog_path,
            "log": log_path,
            "alive": True,
            "stale_s": 0,
            "repeat_tail": 0,
            "exit_code": None,
        }

    # Poll until all done, kill any stalled shard.
    while True:
        alive_any = False
        for gpu_idx, s in shard_state.items():
            proc: subprocess.Popen = s["proc"]
            if s["alive"]:
                if proc.poll() is None:
                    alive_any = True
                    verdict = detectors[gpu_idx].check()
                    if verdict is not None:
                        logger.warning(
                            f"shard gpu{gpu_idx} STALLED: {verdict.describe()} — killing"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        s["alive"] = False
                        s["exit_code"] = -1
                    s["stale_s"] = int(detectors[gpu_idx]._last_change_ts and (time.time() - detectors[gpu_idx]._last_change_ts))
                    s["repeat_tail"] = detectors[gpu_idx].repeat_tail()
                else:
                    s["alive"] = False
                    s["exit_code"] = proc.returncode
                    logger.info(f"shard gpu{gpu_idx} exited rc={proc.returncode}")
        _write_unified_progress(progress, shard_state, n_students, finished=False)
        if not alive_any:
            break
        time.sleep(POLL_INTERVAL_S)

    for stop in stops:
        stop.set()

    # ── Merge shards ────────────────────────────────────────────────
    logger.info("phase 2 done; merging shards")
    merged = _merge_shards([s["out"] for s in shard_state.values()])
    raw_merged = _merge_raw([s["raw"] for s in shard_state.values()])
    out.write_text(json.dumps(merged, indent=2))
    raw_merged_path = out.with_suffix(".raw.json")
    raw_merged_path.write_text(json.dumps(raw_merged, indent=2))

    # ── Phase 3: judge grading on GPU 0 ─────────────────────────────
    logger.info("phase 3: judge grading (GPU 0)")
    judge_log = workdir / "phase3_judge.log"
    proc = _spawn_phase(
        spec_path=spec_path,
        workdir=workdir,
        phase="judge",
        out=out,
        progress=progress,
        gpu=0,
        raw_in=raw_merged_path,
        log_path=judge_log,
    )
    rc = proc.wait()
    if rc != 0:
        logger.error(f"phase 3 (judge) failed rc={rc}; see {judge_log}")
        return rc

    _write_unified_progress(progress, shard_state, n_students, finished=True)
    logger.info("orchestrator finished")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("spec_path")
    p.add_argument("--workdir", default=".")
    p.add_argument("--out", default="results.json")
    p.add_argument("--progress", default="eval_progress.json")
    p.add_argument("--n-gpus", type=int, default=int(os.environ.get("DISTIL_N_GPUS", "8")))
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(
        spec_path=Path(args.spec_path),
        workdir=Path(args.workdir),
        out=Path(args.out),
        progress=Path(args.progress),
        n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    sys.exit(main())
