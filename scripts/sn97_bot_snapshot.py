#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:3710"
# Eval pod name; validator's systemd unit exports DISTIL_LIUM_POD_NAME.
# 2026-05-15: default updated to "8xb200" (current eval pod) and the
# legacy "distil-eval" fallback REMOVED. The fallback was matching the
# unrelated ``distil-eval-b200`` decommissioned pod on the Lium account,
# so when the bot's systemd unit had a stale Environment= override the
# snapshot would silently SSH into a B300 single-GPU pod and report
# stale run_dir / GPU / disk numbers in LIVE_STATUS.md (miners then got
# wrong "Currently scoring" answers via the Discord bot). If we ever
# need a different pod, set DISTIL_LIUM_POD_NAME explicitly.
POD_NAME = os.environ.get("DISTIL_LIUM_POD_NAME") or os.environ.get(
    "LIUM_POD_NAME"
) or "8xb200"
POD_NAME_FALLBACKS: tuple[str, ...] = ()
POD_SSH_KEY = "/root/.ssh/id_ed25519"
# Best-effort: pick up DISTIL_LIUM_POD_NAME from the validator's env file
# even when this script is invoked outside systemd (cron, manual run, etc.)
# so behaviour matches the validator service.
LIUM_ENV_FILE = "/home/distil/.secrets/distil.env"
if "DISTIL_LIUM_POD_NAME" not in os.environ and os.path.exists(LIUM_ENV_FILE):
    try:
        with open(LIUM_ENV_FILE) as _fh:
            for _ln in _fh:
                _ln = _ln.strip()
                if _ln.startswith("DISTIL_LIUM_POD_NAME="):
                    POD_NAME = _ln.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass
WORKSPACE = Path("/root/.openclaw/agents/sn97-bot/workspace")
OUT = WORKSPACE / "LIVE_STATUS.md"
JSON_OUT = WORKSPACE / "LIVE_STATUS.json"
EVAL_LOG_OUT = WORKSPACE / "LIVE_EVAL_LOG.md"
MIRROR = WORKSPACE / "mirror"
MIRROR_STATE = MIRROR / "state"
MIRROR_CODE = MIRROR / "code"
MIRROR_MANIFEST = MIRROR / "MANIFEST.md"

ALLOWED_STATE_FILES = [
    "eval_progress.json",
    "current_round.json",
    "last_eval.json",
    "validator_log.json",
    "incidents.jsonl",
    "h2h_latest.json",
    "h2h_history.json",
    "scores.json",
    "top4_leaderboard.json",
    "disqualified.json",
    "private_pool_commit.json",
    "model_hashes.json",
    "score_history.json",
    # Multi-king queue + composite history.
    "recent_kings.json",
    "composite_scores.json",
]
ALLOWED_STATE_DIRS = ["benchmarks"]

ALLOWED_CODE_FILES = [
    "README.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "CLAUDE.md",
    "IDENTITY.md",
    "SOUL.md",
    "REVISION",
    "benchmark.py",
    "check_model.py",
    "miner.py",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "REWRITE_PLAN.md",
    "SESSION_MEMORY.md",
    # subnet-config.json is the live source of truth for teacher/cap/vocab/arch.
    "frontend/src/lib/subnet-config.json",
]
ALLOWED_CODE_DIRS = [
    "distillation",
    "eval",
    "scripts/validator",
    "scripts/pod",
    "scripts/miner",
    "scripts/api",
    "neurons",
    "api",
    "docs",
    "paper",
]

FORBIDDEN_NAME_SUBSTRINGS = [
    ".env",
    "secret",
    "credential",
    "token",
    "apikey",
    "api_key",
    "password",
    "wallet",
    "hotkey",
    "coldkey",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "auth-profile",
    "auth-state",
]

REPO = Path("/opt/distil/repo")
REPO_STATE = REPO / "state"

SECRET_PATTERNS = [
    (re.compile(r"(sk-ant-[A-Za-z0-9_\-]{20,})"), "ANTHROPIC_KEY"),
    (re.compile(r"(sk-proj-[A-Za-z0-9_\-]{20,})"), "OPENAI_KEY"),
    (re.compile(r"(sk-[A-Za-z0-9]{32,})"), "OPENAI_LEGACY_KEY"),
    (re.compile(r"(ghp_[A-Za-z0-9]{30,})"), "GITHUB_TOKEN"),
    (re.compile(r"(github_pat_[A-Za-z0-9_]{30,})"), "GITHUB_PAT"),
    (re.compile(r"(hf_[A-Za-z0-9]{30,})"), "HF_TOKEN"),
    (re.compile(r"\b(MT[A-Za-z0-9]{20,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{20,})\b"), "DISCORD_BOT_TOKEN"),
    (re.compile(r"\b(OD[A-Za-z0-9]{20,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{20,})\b"), "DISCORD_BOT_TOKEN"),
    (re.compile(r"(\b[A-Z][A-Z0-9_]*(?:API_?KEY|_KEY|_TOKEN|_SECRET|_PASSWORD|_PASSWD)\b\s*[:=]\s*[\"']?)([A-Za-z0-9_\-\.]{16,})([\"']?)"), "ENV_SECRET"),
    (re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)"), "PRIVATE_KEY"),
    (re.compile(r"(-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----)"), "SSH_KEY"),
    (re.compile(r"\b([0-9a-f]{40,64})\b"), "HEX_BLOB"),
    (re.compile(r"\b(5[A-HJ-NP-Z1-9]{47,48})\b"), "SS58_ADDR"),
]

SENSITIVE_JSON_KEYS = {
    "hotkey",
    "coldkey",
    "wallet",
    "wallet_name",
    "wallet_path",
    "lium_api_key",
    "hf_token",
    "token",
    "secret",
    "password",
    "api_key",
}


def redact_text(text, max_hex=False):
    if not isinstance(text, str):
        return text
    out = text
    for pat, label in SECRET_PATTERNS:
        if label == "HEX_BLOB" and not max_hex:
            continue
        if label == "ENV_SECRET":
            out = pat.sub(lambda m: f"{m.group(1)}<REDACTED_{label}>{m.group(3)}", out)
        else:
            out = pat.sub(f"<REDACTED_{label}>", out)
    return out


def redact_value(v, max_hex=False):
    if isinstance(v, str):
        return redact_text(v, max_hex=max_hex)
    if isinstance(v, list):
        return [redact_value(x, max_hex=max_hex) for x in v]
    if isinstance(v, dict):
        clean = {}
        for k, x in v.items():
            key_l = str(k).lower()
            if key_l in SENSITIVE_JSON_KEYS or any(s in key_l for s in ("token", "secret", "password", "private_key")):
                clean[k] = "<REDACTED_FIELD>"
            else:
                clean[k] = redact_value(x, max_hex=max_hex)
        return clean
    return v


def _display_path(value):
    """Keep pod evidence useful without exposing exact filesystem layout."""
    if not value:
        return "—"
    try:
        text = str(value).rstrip("/")
        if not text:
            return "—"
        return Path(text).name or "present"
    except Exception:
        return "present"


def get(path, timeout=5):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": redact_text(str(e))[:200]}


def fmt_age(ts):
    if not ts:
        return "?"
    age = time.time() - float(ts)
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age/60)}m ago"
    if age < 86400:
        return f"{age/3600:.1f}h ago"
    return f"{age/86400:.1f}d ago"


def fmt_ts(ts):
    if not ts:
        return "?"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "?"


def is_name_forbidden(name):
    low = name.lower()
    return any(s in low for s in FORBIDDEN_NAME_SUBSTRINGS)


def mirror_file(src, dst, max_bytes=512000):
    if not src.exists() or not src.is_file():
        return False
    if is_name_forbidden(src.name):
        return False
    try:
        raw = src.read_bytes()
    except Exception:
        return False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes] + b"\n<TRUNCATED>\n"
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return False
    clean = redact_text(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(clean)
    try:
        os.chmod(dst, 0o644)
    except Exception:
        pass
    return True


def mirror_tree(src_root, dst_root, rel, exts=(".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml", ".ini", ".cfg", ".sh")):
    src = src_root / rel
    if not src.exists() or not src.is_dir():
        return 0
    count = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if is_name_forbidden(p.name):
            continue
        if p.suffix and p.suffix.lower() not in exts:
            continue
        rel_path = p.relative_to(src_root)
        dst = dst_root / rel_path
        if mirror_file(p, dst):
            count += 1
    return count


def clean_mirror():
    if MIRROR.exists():
        shutil.rmtree(MIRROR)
    MIRROR.mkdir(parents=True, exist_ok=True)


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (("\n" + p.stderr) if p.returncode else "")
    except Exception as e:
        return -1, f"<exec error: {e}>"


def _read_env_file(path):
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def pod_ssh_info():
    env = _read_env_file(LIUM_ENV_FILE)
    key = env.get("LIUM_API_KEY")
    if not key:
        return None
    try:
        from lium import Config, Lium
        cfg = Config(api_key=key, ssh_key_path=Path(POD_SSH_KEY))
        pods = list(Lium(config=cfg).ps())
        candidates = [POD_NAME, *POD_NAME_FALLBACKS]
        for want in candidates:
            for p in pods:
                if want in p.name:
                    ssh = getattr(p, "ssh_cmd", None) or ""
                    m = re.search(r"ssh\s+(\S+)@(\S+)\s+-p\s+(\d+)", ssh)
                    if m:
                        return {"user": m.group(1), "host": m.group(2), "port": m.group(3),
                                "status": getattr(p, "status", "?"), "name": p.name}
    except Exception as e:
        return {"_error": str(e)[:160]}
    return None


def pod_live_info(ssh):
    if not isinstance(ssh, dict) or "_error" in ssh or not ssh.get("host"):
        return {"reachable": False, "note": "could not resolve pod ssh"}
    cmd = [
        "ssh", "-i", POD_SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=8",
        "-p", str(ssh["port"]),
        f"{ssh['user']}@{ssh['host']}",
        (
            # The pod's eval run-dir layout changed: legacy was
            # /home/distil_eval_<round_id>_<pid>/ (single dir per round);
            # current layout is /home/distil_eval/round_<round_id>/ (one
            # container with per-round subdirs). The old glob
            # /home/distil_eval_*/ ONLY matched the legacy form (the
            # trailing underscore + char requirement skipped the new
            # /home/distil_eval/ container), so on the new layout the
            # snapshot fell back to the most recent LEGACY dir (often
            # weeks-stale) and reported "Eval process DEAD" from a
            # round that finished days ago. This new selector
            # explicitly checks both layouts and picks the most-recent
            # mtime, then picks the right log/pid file for that layout.
            "RD=$(ls -1dt /home/distil_eval/round_*/ /home/distil_eval_*/ 2>/dev/null | head -1); "
            "echo RUN_DIR:$RD; "
            "if [ -n \"$RD\" ]; then "
            "  echo ---PROGRESS---; "
            # New orchestrator's eval_progress.json carries 8 per-GPU
            # shard rows pushing past the legacy 3KB cap; bump to 8KB.
            "  cat ${RD}eval_progress.json 2>/dev/null | head -c 8000; echo; "
            "  echo ---PID_ALIVE---; "
            "  if [ -f ${RD}pod_eval.pid ]; then "
            "    PID=$(cat ${RD}pod_eval.pid); "
            "    if kill -0 $PID 2>/dev/null; then echo alive:$PID; else echo dead:$PID; fi; "
            "  elif echo $RD | grep -q 'distil_eval/round_'; then "
            # New-layout dirs don't carry a pod_eval.pid (the Python
            # orchestrator owns the process tree); detect liveness via
            # the orchestrator PID running on the pod instead.
            "    if pgrep -f \"distil.pod.orchestrator.*${RD#/home/distil_eval/}\" >/dev/null 2>&1; then "
            "      echo alive:orchestrator; "
            "    elif [ -f ${RD}orchestrator.log ] && grep -q 'round completed' ${RD}orchestrator.log 2>/dev/null; then "
            "      echo finished_clean; "
            "    else echo no_pid; fi; "
            "  else echo no_pid; fi; "
            "  echo ---TAIL_LOG---; "
            "  if [ -f ${RD}eval_output.log ]; then tail -40 ${RD}eval_output.log; "
            "  elif [ -f ${RD}orchestrator.log ]; then "
            "    tail -20 ${RD}orchestrator.log 2>/dev/null; "
            "    echo '--- phase1_teacher.log tail ---'; "
            "    tail -10 ${RD}phase1_teacher.log 2>/dev/null; "
            "  fi; "
            "  echo ---DONE_MARKER---; "
            "  if [ -f ${RD}eval_done.marker ]; then echo present; "
            "  elif [ -f ${RD}results.json ]; then echo present_results_json; "
            "  else echo absent; fi; "
            "fi; "
            "echo ---NVIDIA---; "
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null; "
            "echo ---NVIDIA_COUNT---; "
            "nvidia-smi --list-gpus 2>/dev/null | wc -l; "
            "echo ---DISK---; "
            "df -h /home 2>/dev/null | tail -1; "
            "echo ---VLLM_PROCS---; "
            "pgrep -af 'pod_eval.py|vllm|distil.pod' 2>/dev/null | head -5 || echo none"
        ),
    ]
    rc, out = _run(cmd, timeout=25)
    # The 8KB head -c on eval_progress.json alone, plus nvidia-smi (8
    # lines × ~70B), plus log tails, easily blows the old 6KB cap; bump
    # so the trailing ---NVIDIA--- / ---VLLM_PROCS--- sections aren't
    # silently truncated mid-line.
    return {"reachable": rc == 0, "rc": rc, "raw": out[:16000]}


def parse_pod_live(raw):
    """Turn the SSH output into structured fields."""
    out = {"phase": None, "active": None, "students_total": None,
           "students_done": None, "prompts_total": None, "current": None,
           "teacher_prompts_done": None, "pid_state": None, "done_marker": None,
           "gpu": None, "disk": None, "procs_count": 0, "run_dir": None,
           "log_tail": None}
    if not isinstance(raw, str):
        return out
    m = re.search(r"RUN_DIR:([^\n]+)", raw)
    if m:
        out["run_dir"] = m.group(1).strip()
    # Extract the JSON between ``---PROGRESS---`` and the next section
    # marker. Previously this used ``(\{.*?\})\n`` which assumed a
    # single-line trailing brace, but the new orchestrator's eval_progress
    # has 8 per-GPU shard entries pushing the JSON well past the SSH
    # buffer's old 3KB cap; truncation chopped the final ``}`` so the
    # non-greedy regex matched the FIRST inner shard ``}`` and produced
    # invalid JSON that the except branch silently swallowed.
    pb = re.search(r"---PROGRESS---\n(.+?)\n---", raw, re.DOTALL)
    if pb:
        try:
            pj = json.loads(pb.group(1).strip())
            out["phase"] = pj.get("phase")
            # Pod's eval_progress.json sets ``active=False`` the moment the
            # round wraps. Surface this so the narrative builder can
            # distinguish "between rounds" from "stuck mid-round" instead
            # of leaning on the now-removed pod_eval.pid as a liveness
            # proxy.
            if "active" in pj:
                out["active"] = bool(pj.get("active"))
            out["students_total"] = pj.get("students_total")
            out["prompts_total"] = pj.get("prompts_total")
            out["teacher_prompts_done"] = pj.get("teacher_prompts_done")
            completed = pj.get("completed") or []
            out["students_done"] = len(completed)
            out["completed"] = [
                {"name": c.get("student_name"), "status": c.get("status"),
                 "kl": c.get("kl"), "prompts_scored": c.get("prompts_scored")}
                for c in completed
            ]
            cur = pj.get("current")
            if isinstance(cur, dict):
                out["current"] = {
                    "name": cur.get("student_name"),
                    "idx": cur.get("student_idx"),
                    "prompts_done": cur.get("prompts_done"),
                    "prompts_total": cur.get("prompts_total"),
                    "kl_running": cur.get("kl_running_mean"),
                    "best_kl": cur.get("best_kl_so_far"),
                }
        except Exception:
            pass
    m = re.search(
        r"---PID_ALIVE---\n(alive(?::orchestrator|:\d+)?|dead(?::\d+)?|finished_clean|no_pid)",
        raw,
    )
    if m:
        out["pid_state"] = m.group(1)
    m = re.search(r"---DONE_MARKER---\n(present(?:_results_json)?|absent)", raw)
    if m:
        out["done_marker"] = m.group(1)
    nv = re.search(r"---NVIDIA---\n(.*?)(?=\n---|\Z)", raw, re.DOTALL)
    if nv:
        gpu_lines = [ln.strip() for ln in nv.group(1).splitlines() if ln.strip()]
        out["gpus"] = gpu_lines  # full list for debugging
        if gpu_lines:
            # Summarise as e.g. "8x NVIDIA H200 (12345/143771 MiB used, 0% util)"
            first = gpu_lines[0]
            try:
                parts = [p.strip() for p in first.split(",")]
                # parts: index, name, memory.used [MiB], memory.total [MiB], util %
                name = parts[1] if len(parts) > 1 else "?"
                mem_u = parts[2] if len(parts) > 2 else "?"
                mem_t = parts[3] if len(parts) > 3 else "?"
                util = parts[4] if len(parts) > 4 else "?"
                count = len(gpu_lines)
                prefix = f"{count}× " if count > 1 else ""
                out["gpu"] = f"{prefix}{name} ({mem_u} / {mem_t} used, util {util})"
            except Exception:
                out["gpu"] = first
    m = re.search(r"---NVIDIA_COUNT---\n(\d+)", raw)
    if m:
        try:
            out["gpu_count"] = int(m.group(1))
        except Exception:
            pass
    m = re.search(r"---DISK---\n([^\n]+)", raw)
    if m:
        out["disk"] = m.group(1).strip()
    procs = re.findall(r"---VLLM_PROCS---\n(.*?)(?=\n---|\Z)", raw, re.DOTALL)
    if procs:
        proc_text = procs[0].strip()
        out["procs_count"] = 0 if proc_text in ("none", "") else len([ln for ln in proc_text.splitlines() if ln.strip()])
    tail = re.search(r"---TAIL_LOG---\n(.*?)---DONE_MARKER---", raw, re.DOTALL)
    if tail:
        t = tail.group(1).strip()
        if t:
            out["log_tail"] = t[-1500:]
    return out


def recent_journal(unit, minutes=20, lines=25):
    rc, out = _run(
        ["journalctl", "-u", unit, f"--since", f"{minutes} minutes ago",
         "-n", str(lines), "--no-pager", "-o", "cat"],
        timeout=15,
    )
    if rc != 0:
        return []
    return [redact_text(ln) for ln in out.splitlines() if ln.strip()][-lines:]


def summarize_last_eval(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    students = data.get("students") or {}
    if not students:
        return None
    rows = []
    for name, v in students.items():
        rows.append({
            "model": name,
            "status": v.get("status"),
            "kl": v.get("kl_global_avg"),
            "prompts_scored": v.get("prompts_scored"),
            "reason": v.get("reason"),
        })
    return {"count": len(rows), "students": rows}


now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

health = get("/api/health")
overview = get("/api/telemetry/overview")
events = get("/api/telemetry/events?limit=15")
errors = get("/api/telemetry/errors?limit=10")
dqs = get("/api/telemetry/dqs?limit=10")
pod = get("/api/telemetry/pod-health")
eval_progress = get("/api/eval-progress")
leaderboard = get("/api/leaderboard")
scores = get("/api/scores")
h2h = get("/api/h2h-latest")
private_pool = get("/api/private-pool-commit")

ssh_info = pod_ssh_info()
pod_live_raw = pod_live_info(ssh_info) if isinstance(ssh_info, dict) and "_error" not in ssh_info else {"reachable": False}
pod_live = parse_pod_live(pod_live_raw.get("raw", ""))
pod_live["reachable"] = pod_live_raw.get("reachable", False)

journal_validator = recent_journal("distil-validator", minutes=30, lines=25)
journal_api = recent_journal("distil-api", minutes=30, lines=10)

last_eval_summary = summarize_last_eval(REPO_STATE / "last_eval.json")

snapshot_raw = {
    "generated_at": now_iso,
    "server_time_epoch": time.time(),
    "health": health,
    "overview": overview,
    "eval_progress": eval_progress,
    "recent_events": events.get("entries") if isinstance(events, dict) else None,
    "recent_errors": errors.get("entries") if isinstance(errors, dict) else None,
    "recent_dqs": dqs.get("disqualified") if isinstance(dqs, dict) else None,
    "pod_health": pod,
    "pod_live": pod_live,
    "leaderboard": leaderboard,
    "h2h_latest": h2h,
    "private_pool_commit": private_pool,
    "journal_validator_tail": journal_validator,
    "journal_api_tail": journal_api,
    "last_eval_summary": last_eval_summary,
}

trimmed = dict(snapshot_raw)
if isinstance(trimmed.get("overview"), dict):
    ov = dict(trimmed["overview"])
    ov.pop("scores", None)
    ov.pop("commitments", None)
    ov.pop("history", None)
    cr = ov.get("current_round")
    if isinstance(cr, dict):
        cr_small = {k: v for k, v in cr.items() if k not in ("prompts", "model_names")}
        ov["current_round"] = cr_small
    trimmed["overview"] = ov

snapshot = redact_value(trimmed)

JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
JSON_OUT.write_text(json.dumps(snapshot, indent=2, default=str)[:200000])


eval_log_lines = [
    "# Live Eval Log (auto-updated)",
    "",
    f"**Generated:** {now_iso}",
    "**Freshness:** Auto-refreshes every 60s with `sn97-bot-snapshot.timer`; if older than 3 minutes, treat as stale.",
    "",
    "Use this file for Discord questions about the live evaluation path, progress, stalls, and recent validator errors. It is a sanitized snapshot, not raw private infrastructure access.",
    "",
]

cr_for_eval_log = {}
if isinstance(overview, dict) and isinstance(overview.get("current_round"), dict):
    cr_for_eval_log = overview["current_round"]

pod_eval_meta = {}
if isinstance(cr_for_eval_log, dict) and isinstance(cr_for_eval_log.get("pod_eval"), dict):
    pod_eval_meta = dict(cr_for_eval_log["pod_eval"])

ep_for_log = eval_progress if isinstance(eval_progress, dict) else {}
eval_log_lines.extend([
    "## Progress",
    "",
    f"- **Active:** {bool(ep_for_log.get('active'))}",
    f"- **Phase:** `{ep_for_log.get('phase')}`",
    f"- **Students:** {ep_for_log.get('students_done')}/{ep_for_log.get('students_total')}",
    f"- **Current:** `{ep_for_log.get('current_student') or ((ep_for_log.get('pod') or {}).get('current') or {}).get('student_name') if isinstance((ep_for_log.get('pod') or {}).get('current'), dict) else ep_for_log.get('current_student')}`",
    f"- **Current prompt:** {ep_for_log.get('current_prompt')}",
    f"- **Current KL:** {ep_for_log.get('current_kl')}",
    "",
])

completed = []
pod_progress = ep_for_log.get("pod") if isinstance(ep_for_log.get("pod"), dict) else {}
if isinstance(pod_progress.get("completed"), list):
    completed = pod_progress["completed"]
elif isinstance(ep_for_log.get("completed"), list):
    completed = ep_for_log["completed"]
if completed:
    eval_log_lines.extend(["## Students Completed This Run", ""])
    for c in completed[-12:]:
        if not isinstance(c, dict):
            continue
        eval_log_lines.append(
            f"- `{c.get('student_name')}` — status={c.get('status')} KL={c.get('kl')} prompts={c.get('prompts_scored')}/{c.get('prompts_total')}"
        )
    eval_log_lines.append("")

eval_log_lines.extend([
    "## Pod Eval Metadata",
    "",
])
if pod_eval_meta:
    for key in ("run_dir", "progress_remote", "results_remote", "log_remote", "done_marker_remote", "started_at"):
        val = pod_eval_meta.get(key)
        if val is not None:
            if key.endswith("_remote") or key == "run_dir":
                eval_log_lines.append(f"- **{key}:** `{_display_path(val)}`")
            else:
                eval_log_lines.append(f"- **{key}:** `{redact_text(str(val))}`")
else:
    eval_log_lines.append("- No persisted `current_round.pod_eval` metadata found.")
eval_log_lines.append("")

eval_log_lines.extend([
    "## Direct Pod Snapshot",
    "",
    f"- **Reachable:** {pod_live.get('reachable')}",
    f"- **Run dir:** `{_display_path(pod_live.get('run_dir'))}`",
    f"- **Phase:** `{pod_live.get('phase')}`",
    f"- **PID:** `{pod_live.get('pid_state')}`",
    f"- **Done marker:** `{pod_live.get('done_marker')}`",
    f"- **GPU:** `{pod_live.get('gpu')}`",
    f"- **Disk:** `{pod_live.get('disk')}`",
    f"- **pod_eval/vLLM procs:** {pod_live.get('procs_count')}",
    "",
])

if pod_live.get("log_tail"):
    eval_log_lines.extend([
        "## Pod `eval_output.log` Tail",
        "",
        "```",
        redact_text(str(pod_live["log_tail"]))[-4000:],
        "```",
        "",
    ])
else:
    eval_log_lines.extend([
        "## Pod `eval_output.log` Tail",
        "",
        "Direct pod log tail is not available in this snapshot. Use the Progress, Pod Eval Metadata, and Validator Journal sections instead; do not invent pod-log evidence.",
        "",
    ])

if journal_validator:
    eval_log_lines.extend([
        "## Validator Journal Tail",
        "",
        "```",
    ])
    eval_log_lines.extend(ln[:500] for ln in journal_validator[-25:])
    eval_log_lines.extend(["```", ""])

recent_events_for_log = events.get("entries") if isinstance(events, dict) else []
if recent_events_for_log:
    eval_log_lines.extend(["## Recent Validator Events", ""])
    for e in recent_events_for_log[-15:]:
        if not isinstance(e, dict):
            continue
        eval_log_lines.append(f"- `{fmt_age(e.get('ts'))}` {(e.get('level') or 'info').upper()}: {redact_text(str(e.get('msg') or ''))}")
    eval_log_lines.append("")

recent_errors_for_log = errors.get("entries") if isinstance(errors, dict) else []
if recent_errors_for_log:
    eval_log_lines.extend(["## Recent Errors", ""])
    for e in recent_errors_for_log[-10:]:
        if not isinstance(e, dict):
            continue
        eval_log_lines.append(f"- `{fmt_age(e.get('ts'))}` {redact_text(str(e.get('msg') or ''))}")
    eval_log_lines.append("")

eval_log_lines.extend([
    "## Answering Rules",
    "",
    "- For live eval questions, read this file first, then `LIVE_STATUS.md` if needed.",
    "- If Direct Pod Snapshot is unreachable, say that explicitly and rely only on API progress / validator journal evidence.",
    "- Do not expose raw server IPs, tokens, environment paths, or private keys. The contents here are already sanitized, but keep answers concise.",
    "- Do not paste raw `mirror/state/*.json`; summarize the relevant fields in plain English.",
    "- Do not claim a deployment or fix is live unless it appears in this file, `LIVE_STATUS.md`, or a verified GitHub commit.",
    "",
])

EVAL_LOG_OUT.write_text(redact_text("\n".join(eval_log_lines)))

clean_mirror()
MIRROR_STATE.mkdir(parents=True, exist_ok=True)
MIRROR_CODE.mkdir(parents=True, exist_ok=True)

state_count = 0
for name in ALLOWED_STATE_FILES:
    src = REPO_STATE / name
    if mirror_file(src, MIRROR_STATE / name, max_bytes=1024000):
        state_count += 1
for d in ALLOWED_STATE_DIRS:
    state_count += mirror_tree(REPO_STATE, MIRROR_STATE, d, exts=(".json",))

code_count = 0
for name in ALLOWED_CODE_FILES:
    src = REPO / name
    if mirror_file(src, MIRROR_CODE / name, max_bytes=2048000):
        code_count += 1
for d in ALLOWED_CODE_DIRS:
    code_count += mirror_tree(REPO, MIRROR_CODE, d)

# Per-UID miner snapshot: pre-materialize /api/miner/{uid} for every UID so
# the bot can answer "why is UID N X?" from the workspace (loopback blocked).
miners_count = 0
miners_errors = 0
miners_out = {}
metagraph = get("/api/metagraph", timeout=8)
neurons = metagraph.get("neurons", []) if isinstance(metagraph, dict) else []
uid_list = sorted({int(n["uid"]) for n in neurons if isinstance(n, dict) and isinstance(n.get("uid"), int)})
for uid in uid_list:
    entry = get(f"/api/miner/{uid}", timeout=3)
    if isinstance(entry, dict) and "_error" not in entry:
        miners_out[str(uid)] = redact_value(entry)
        miners_count += 1
    else:
        miners_errors += 1
# Overwrite miners.json only when "healthy" (>=50 UIDs, <30% error rate);
# stale data is preferable to clobbering with an almost-empty snapshot.
total_attempts = miners_count + miners_errors
is_healthy = miners_count >= 50 and (total_attempts == 0 or miners_errors / total_attempts < 0.3)
if miners_out and is_healthy:
    (MIRROR_STATE / "miners.json").write_text(json.dumps({
        "generated_at": now_iso,
        "count": miners_count,
        "miners": miners_out,
    }, indent=2, default=str))
    state_count += 1
    hk_to_uid = {}
    for uid_str, m in miners_out.items():
        hk = m.get("hotkey")
        if hk:
            hk_to_uid[hk] = int(uid_str)
    (MIRROR_STATE / "hotkey_to_uid.json").write_text(json.dumps(hk_to_uid, indent=2))
    state_count += 1
elif miners_out:
    # Degraded snapshot — log and keep the existing file.
    print(
        f"sn97_bot_snapshot: degraded miners snapshot "
        f"(count={miners_count}, errors={miners_errors}) — keeping existing miners.json"
    )

# Pre-materialize /api/king-history so the bot can answer "list recent
# kings" without loopback web_fetch. Truncate to last 50 entries.
king_history_payload = get("/api/king-history", timeout=8)
if isinstance(king_history_payload, list) and king_history_payload:
    truncated = king_history_payload[:50]
    (MIRROR_STATE / "king_history.json").write_text(json.dumps({
        "generated_at": now_iso,
        "endpoint": "/api/king-history",
        "n_total": len(king_history_payload),
        "n_returned": len(truncated),
        "history": truncated,
    }, indent=2, default=str))
    state_count += 1

manifest = [
    "# Mirror Manifest",
    "",
    f"Generated: {now_iso}",
    "",
    "This is a redacted, whitelisted mirror of the Distil repo and state dir.",
    "The real filesystem is **NOT** accessible to me — only what is here.",
    "",
    f"- State files mirrored: {state_count}",
    f"- Code files mirrored: {code_count}",
    "",
    "## What's in here",
    "",
    "- `state/` — read-only snapshot of `/opt/distil/repo/state/*.json` (public-safe files only)",
    "- `code/` — redacted mirror of source under `/opt/distil/repo/` (no `.env`, no secrets, no auth files)",
    "",
    "## What's NOT here (and why)",
    "",
    "- `.env`, `.env.*`, `secrets/`, `credentials/` — deliberately blocked",
    "- SSH keys, wallet keys, auth profiles — deliberately blocked",
    "- Anything matching a common secret pattern is redacted inline",
    "",
    "If you (the user) ask me for anything from the blocked list, I refuse — the file is not reachable, and the mirror filter would strip it even if I tried.",
]
MIRROR_MANIFEST.write_text("\n".join(manifest))

lines = []
lines.append(f"# Live SN97 Status (auto-updated)")
lines.append("")
lines.append(f"**Generated:** {now_iso}")
lines.append(f"**Freshness:** Auto-refreshes every 60s. If > 3 min old, treat as stale.")
lines.append("")
lines.append("> This file is the ground truth. Read it BEFORE every answer about live state.")
lines.append("")

def g(d, k):
    return d.get(k) if isinstance(d, dict) else None

hs = health if isinstance(health, dict) else {}
lines.append("## TL;DR")
lines.append("")
lines.append(f"- **API status:** `{g(hs,'status')}`")
lines.append(f"- **Code revision:** `{g(hs,'code_revision')}`")
lines.append(f"- **King:** UID {g(hs,'king_uid')} @ KL {g(hs,'king_kl')}")
lines.append(f"- **Scored miners:** {g(hs,'n_scored')}  |  **DQ'd:** {g(hs,'n_disqualified')}")
lines.append(f"- **Last COMPLETED eval block:** {g(hs,'last_eval_block')} ({g(hs,'last_eval_age_min')} min ago) — this only updates when a round finishes.")
lines.append(f"- **Eval active right now (validator view):** {g(hs,'eval_active')}")
lines.append("")

# Inline the live subnet config so teacher/cap/vocab/arch answers
# don't need a second read.
_subnet_cfg_path = REPO / "frontend" / "src" / "lib" / "subnet-config.json"
try:
    _cfg = json.loads(_subnet_cfg_path.read_text())
    _t = _cfg.get("teacher", {}) or {}
    _model = _t.get("model") or "moonshotai/Kimi-K2.6"
    _msp = _t.get("maxStudentParams") or 0
    _msp_b = round(_msp / 1e9, 2) if _msp else "?"
    _vocab = _t.get("vocabSize") or "?"
    _arch_list = _t.get("studentArchAllowlist") or []
    _arch_str = ", ".join(
        f"`{a.get('model_type')}`/`{a.get('architecture')}`"
        for a in _arch_list if isinstance(a, dict)
    ) or "?"
    _ref = _cfg.get("validator", {}).get("referenceModel") or "?"
    _prev = (_cfg.get("teacher", {}).get("previousTeacher")
             or _cfg.get("previousTeacher", {}).get("model")
             or "(none recorded)")
    # How the validator fetches teacher logprobs (api=cloud, vllm=local).
    _teacher_mode = os.environ.get("DISTIL_TEACHER_MODE", "vllm").lower()
    _teacher_api_provider = os.environ.get("DISTIL_TEACHER_API_PROVIDERS") or "Inceptron"
    _teacher_api_model = os.environ.get("DISTIL_TEACHER_API_MODEL") or _model
    if _teacher_mode == "api":
        _teacher_path_str = (
            f"cloud-API (OpenRouter → `{_teacher_api_provider}` → `{_teacher_api_model}`); "
            f"top-20 logprobs; no local 1T-param load"
        )
    else:
        _teacher_path_str = "local vLLM (Lium pod, 8×H200, ~6 min cold start)"
    lines.append("## Live Subnet Config (live from `frontend/src/lib/subnet-config.json` — quote these for cap/teacher/vocab/arch questions)")
    lines.append("")
    lines.append(f"- **Teacher:** `{_model}`")
    lines.append(f"- **Teacher inference path:** {_teacher_path_str}")
    lines.append(f"- **Max student total params:** `{_msp}` (= **{_msp_b}B**)")
    lines.append(f"- **Required vocab size:** `{_vocab}` (Kimi BPE; not 248,320, not 152,064)")
    lines.append(f"- **Allowed model_type / architecture:** {_arch_str}")
    lines.append(f"- **Reference baseline (UID -1):** `{_ref}`")
    lines.append(f"- **Previous teacher (retired, do not quote as live):** `{_prev}`")
    lines.append("")
    lines.append("If a miner asks 'what is the cap / teacher / vocab / which arch?' — the answer is the line above. Never quote 7B, 5.25B, 40B, 4B, 248,320, or `Qwen3_5ForConditionalGeneration` as the LIVE config; those are pre-cutover historical numbers.")
    lines.append("")
except Exception as _cfg_exc:
    lines.append("## Live Subnet Config")
    lines.append("")
    lines.append(f"- (config read failed: `{_cfg_exc}`; fall back to `mirror/code/frontend/src/lib/subnet-config.json` directly)")
    lines.append("")

narrative = []
pl = pod_live
api_eval = bool(g(hs, "eval_active"))
pid_state = (pl.get("pid_state") or "").split(":")[0]
done = pl.get("done_marker") == "present"
phase = pl.get("phase")
cur = pl.get("current") or {}
if pl.get("reachable"):
    gpu_s = pl.get("gpu") or "?"
    # Cross-check the pod's view of `active`/`phase` with the validator
    # snapshot. In the new orchestrator layout the pod writes
    # ``active: false`` + ``phase: finished`` into eval_progress.json
    # the moment the round wraps up, so the validator may show
    # ``eval_active: false`` for the entire 70-minute inter-round sleep.
    # Treat that as "between rounds", NOT "stuck".
    pod_active = bool(pl.get("active")) if pl.get("active") is not None else None
    finished_phase = phase in (None, "finished", "done") or done
    if pid_state == "alive" and not done and not finished_phase:
        narrative.append(
            f"**Eval is RUNNING on pod.** Phase=`{phase}`. "
            + (f"Current student UID/model: `{cur.get('name')}` "
               f"at prompt {cur.get('prompts_done')}/{cur.get('prompts_total')} "
               f"(running KL {cur.get('kl_running')}, best-so-far {cur.get('best_kl')}). "
               if cur else "")
            + f"GPU: {gpu_s}. Disk: {pl.get('disk')}. "
            + f"{pl.get('students_done') or 0}/{pl.get('students_total') or '?'} students scored."
        )
    elif pid_state == "finished_clean" or (finished_phase and not api_eval):
        narrative.append(
            "**Round COMPLETE — validator is between rounds.** Pod's "
            "eval_progress.json reports `phase=finished`, "
            "no orchestrator/vllm processes running on the pod, GPUs idle. "
            "On 2026-05-18 the 70-min `ROUND_INTERVAL_S` sleep was dropped "
            "to **0s** (`settings.round_interval_s = 0`) — the next round "
            "is queued immediately. A 1-2 min idle gap is normal "
            "(commitment fetch + challenger selection); anything >10 min "
            "of `phase=finished` warrants a `journalctl -u distil-validator` "
            "check, but it is NOT automatically a hang."
        )
    elif done:
        narrative.append("**Eval DONE on pod.** done marker present; validator will apply results next cycle.")
    elif pid_state == "dead":
        # Only consider this a real problem if the round is BOTH not
        # marked finished AND the API still thinks eval is active.
        if api_eval and not finished_phase:
            narrative.append(
                "**Eval process DEAD on pod mid-round** (pid recorded "
                "but not alive, phase not yet `finished`, validator still "
                "reports `eval_active=true`). Validator should clean up "
                "and restart next epoch — flag this in #ops if it "
                "persists past one full sleep cycle (~70 min)."
            )
        else:
            narrative.append(
                "Pod's pod_eval.pid points to a stale dead process from a "
                "previous round (the new orchestrator layout doesn't write "
                "this file). Round is complete; validator is between rounds. "
                "This is cosmetic, not a real failure."
            )
    elif pid_state == "no_pid":
        narrative.append("No active pod_eval PID on pod. Validator is likely between rounds or prechecking.")
    else:
        narrative.append(f"Pod reachable, phase=`{phase}`, pid_state=`{pid_state}`.")
else:
    narrative.append("Pod side unreachable from snapshot. Relying on API progress only.")

if api_eval and pid_state == "no_pid":
    narrative.append("NOTE: API says eval_active=true but the pod has no live eval PID. This usually means the validator is about to (re)launch or a stale flag is set; state should self-correct within one epoch.")
if not api_eval and pid_state == "alive":
    narrative.append("NOTE: The pod is generating/scoring, but the api eval_active=false. The validator may have just restarted and is attaching to the live eval via the resume-on-attach path. This is normal.")

lines.append("## What is happening right now")
lines.append("")
for n in narrative:
    lines.append(f"- {n}")
lines.append("")

lines.append("## Pod Live Snapshot (direct SSH)")
lines.append("")
lines.append(f"- **Reachable:** {pl.get('reachable')}")
lines.append(f"- **Run dir:** `{_display_path(pl.get('run_dir'))}`")
lines.append(f"- **Phase:** `{phase}`")
lines.append(f"- **PID alive:** `{pl.get('pid_state')}`  |  **Done marker:** `{pl.get('done_marker')}`")
lines.append(f"- **GPU:** `{pl.get('gpu')}`")
lines.append(f"- **Disk (/home):** `{pl.get('disk')}`")
lines.append(f"- **Running pod_eval/vllm procs:** {pl.get('procs_count')}")
if pl.get("completed"):
    lines.append("- **Students scored so far (this run):**")
    for c in pl["completed"]:
        kl = c.get("kl")
        kl_s = f"KL={kl}" if kl is not None else f"status={c.get('status')}"
        lines.append(f"  - `{c.get('name')}` — {kl_s}  (prompts={c.get('prompts_scored')})")
if cur:
    lines.append(f"- **Currently scoring:** `{cur.get('name')}` "
                 f"{cur.get('prompts_done')}/{cur.get('prompts_total')} prompts, "
                 f"running KL={cur.get('kl_running')}, best-so-far={cur.get('best_kl')}")
if pl.get("log_tail"):
    lines.append("")
    lines.append("<details><summary>Last 40 lines of pod eval_output.log</summary>")
    lines.append("")
    lines.append("```")
    lines.append(pl["log_tail"])
    lines.append("```")
    lines.append("")
    lines.append("</details>")
lines.append("")

ep = eval_progress if isinstance(eval_progress, dict) else {}
if ep and "phase" in ep:
    lines.append("## Current Eval Progress")
    lines.append("")
    lines.append(f"- **Phase:** `{ep.get('phase')}`")
    if ep.get("students_total"):
        lines.append(
            f"- **Students:** {ep.get('students_done')}/{ep.get('students_total')} "
            f"(**includes the king** — see eval_order below)"
        )
    if ep.get("prompts_total"):
        lines.append(f"- **Prompts/student:** {ep.get('prompts_total')}")
    if ep.get("current_student") is not None:
        lines.append(
            f"- **Currently evaluating:** UID {ep.get('current_student')} "
            f"(prompt {ep.get('current_prompt')}/{ep.get('prompts_total')}, "
            f"live KL {ep.get('current_kl')})"
        )
    # 2026-05-15: explicitly show the eval_order with role=king vs role=challenger
    # so the Discord bot cannot mistakenly conclude "king is in the queue but not
    # actually evaluated". Previous incident: bot read pod-side per-GPU logs
    # (which list challengers only) and saw 10 names; concluded king was missing.
    # In reality the king is launched on GPU0 in the same phase1+king command
    # (``--students <king_model> --king <king_model>``) and lands in the merged
    # ``eval_results.json`` alongside the N challengers. Show the role split
    # here so the bot has unambiguous evidence.
    eval_order = ep.get("eval_order") or []
    king_uid = ep.get("king_uid")
    challenger_uids = ep.get("challenger_uids") or []
    if isinstance(eval_order, list) and eval_order:
        lines.append(
            f"- **Eval order ({len(eval_order)} models — king + "
            f"{len(challenger_uids)} challengers):**"
        )
        for e in eval_order:
            role = e.get("role")
            uid = e.get("uid")
            model = e.get("model") or "?"
            tag = " (king — re-scored on GPU0 this round)" if role == "king" else ""
            lines.append(f"  - UID {uid} `{model}` [role={role}]{tag}")
        lines.append(
            f"- **King re-evaluation:** ENABLED (SINGLE_EVAL_KING_REEVAL=1). "
            f"King UID {king_uid} is scored on GPU0 in the same phase1+king "
            f"command, then merged with challengers in `eval_results.json`. "
            f"If anyone claims the king is NOT being evaluated, they are wrong "
            f"— check `eval_results.json` after the round completes; the king "
            f"appears there with `status=scored` and a fresh `kl_global_avg`."
        )
    lines.append("")

ov = overview if isinstance(overview, dict) else {}
if ov and "current_round" in ov and ov["current_round"]:
    cr = ov["current_round"]
    lines.append("## Current Round")
    lines.append("")
    lines.append(f"- **Started:** {fmt_ts(cr.get('started_at'))} ({fmt_age(cr.get('started_at'))})")
    lines.append(f"- **Block:** {cr.get('block')}")
    lines.append(f"- **King UID:** {cr.get('king_uid')}")
    mte = cr.get("models_to_eval") or {}
    if mte:
        lines.append(f"- **Models in round ({len(mte)}):**")
        for uid, info in mte.items():
            mark = " (king/ref)" if info.get("is_reference") else ""
            lines.append(f"  - UID {uid}: `{info.get('model')}`{mark}")
    pp = cr.get("private_pool") or {}
    if pp:
        lines.append(f"- **Private pool:** n={pp.get('n')} fraction={pp.get('fraction')} commit=`{(pp.get('commit_root') or '')[:16]}...`")
    lines.append("")

lb = leaderboard if isinstance(leaderboard, dict) else {}
lb_inner = lb.get("leaderboard") if isinstance(lb.get("leaderboard"), dict) else lb
king_entry = lb_inner.get("king") if isinstance(lb_inner, dict) else None
contenders = lb_inner.get("contenders") if isinstance(lb_inner, dict) else None
rows = []
if isinstance(king_entry, dict):
    rows.append({"label": "king", **king_entry})
if isinstance(contenders, list):
    for c in contenders:
        if isinstance(c, dict):
            rows.append({"label": "contender", **c})
if rows:
    lines.append("## Top-5 Leaderboard (king + contenders)")
    lines.append("")
    lines.append("| Role | UID | KL | Model |")
    lines.append("|------|-----|----|-------|")
    for r in rows[:5]:
        kl = r.get("h2h_kl") if r.get("h2h_kl") is not None else r.get("kl")
        model = r.get("model_path") or r.get("model") or "—"
        lines.append(f"| {r.get('label')} | {r.get('uid')} | {kl} | `{model}` |")
    lines.append("")
    ref = lb.get("reference_baseline") if isinstance(lb, dict) else None
    if isinstance(ref, dict):
        lines.append(f"Reference (UID -1): `{ref.get('model')}` KL={ref.get('kl')}")
        lines.append("")

h2h_round = h2h if isinstance(h2h, dict) else {}
if h2h_round and h2h_round.get("round"):
    r = h2h_round["round"]
    lines.append("## Latest H2H Round")
    lines.append("")
    lines.append(f"- **Block:** {r.get('block')}  |  **Finished:** {fmt_age(r.get('finished_at'))}")
    lines.append(f"- **Winner UID:** {r.get('winner_uid')}  |  **Dethroned king:** {r.get('dethroned')}")
    for p in (r.get("participants") or [])[:6]:
        lines.append(f"  - UID {p.get('uid')}: KL {p.get('kl')} (`{p.get('model_path') or p.get('model')}`)")
    lines.append("")

recent_events = events.get("entries") if isinstance(events, dict) else []
if recent_events:
    lines.append("## Recent Validator Events (last 15)")
    lines.append("")
    for e in recent_events:
        lvl = (e.get("level") or "info").upper()[:5]
        msg = redact_text(str(e.get("msg") or ""))
        lines.append(f"- `{fmt_age(e.get('ts'))}` **[{lvl}]** {msg}")
    lines.append("")

recent_errors = errors.get("entries") if isinstance(errors, dict) else []
if recent_errors:
    lines.append("## Recent Errors (last 10)")
    lines.append("")
    for e in recent_errors:
        msg = redact_text(str(e.get("msg") or ""))
        lines.append(f"- `{fmt_age(e.get('ts'))}` {msg}")
    lines.append("")

recent_dqs = dqs.get("disqualified") if isinstance(dqs, dict) else []
if recent_dqs:
    lines.append("## Recent DQs (last 10)")
    lines.append("")
    for d in recent_dqs:
        reason = redact_text(str(d.get("reason") or ""))
        lines.append(f"- UID {d.get('uid')} @ block {d.get('block')}: {reason}")
    lines.append("")

pod_h = pod if isinstance(pod, dict) else {}
if pod_h and not pod_h.get("_error"):
    lines.append("## GPU Pod Health")
    lines.append("")
    for k, v in pod_h.items():
        if k.startswith("_"):
            continue
        val = redact_text(str(v))
        if len(val) > 100:
            val = val[:100] + "..."
        lines.append(f"- **{k}:** `{val}`")
    lines.append("")

pp_commit = private_pool if isinstance(private_pool, dict) else {}
if pp_commit and not pp_commit.get("_error"):
    lines.append("## Private Pool Commitment")
    lines.append("")
    for k, v in pp_commit.items():
        if k.startswith("_"):
            continue
        val = redact_text(str(v))
        if len(val) > 100:
            val = val[:100] + "..."
        lines.append(f"- **{k}:** `{val}`")
    lines.append("")

if last_eval_summary and last_eval_summary.get("students"):
    lines.append("## Most Recent Completed Eval (from state/last_eval.json)")
    lines.append("")
    for s in last_eval_summary["students"]:
        kl = s.get("kl")
        status = s.get("status")
        if kl is not None and status == "scored":
            extra = f"KL={kl}"
        else:
            extra = f"status={status}"
            if s.get("reason"):
                extra += f" ({s.get('reason')[:120]})"
        lines.append(f"- `{s.get('model')}` — {extra}  (prompts_scored={s.get('prompts_scored')})")
    lines.append("")

if journal_validator:
    lines.append("## Validator journalctl (last 25 lines, last 30 min)")
    lines.append("")
    lines.append("```")
    for ln in journal_validator[-25:]:
        lines.append(ln[:300])
    lines.append("```")
    lines.append("")

if journal_api:
    lines.append("## API journalctl (last 10 lines, last 30 min)")
    lines.append("")
    lines.append("```")
    for ln in journal_api[-10:]:
        lines.append(ln[:300])
    lines.append("```")
    lines.append("")

lines.append("---")
lines.append("")
lines.append("## Freshness / accuracy rules (READ CAREFULLY)")
lines.append("")
lines.append("- The values above are live as of **Generated** at the top of this file.")
lines.append("- If someone asks about an error, issue, or crash: only cite errors whose timestamp falls inside the journalctl window above OR the `Recent Errors` list. Do NOT reference older errors from the `mirror/code` snapshot or memory — they may be stale or already fixed.")
lines.append("- If the data above does not answer the question, say so explicitly (e.g. \"I don't have that in live telemetry\") instead of guessing.")
lines.append("- If `pod_live.reachable == false`, say so; don't invent pod state.")
lines.append("- The `mirror/code/` folder is SOURCE code, not logs. Seeing `import re` in a `.py` file is just a Python import, not an error.")
lines.append("")
lines.append("## Known anti-FUD facts (do NOT contradict these)")
lines.append("")
lines.append(
    "- **The king IS re-evaluated every round** (paired re-eval on shared "
    "prompts). The validator sets `SINGLE_EVAL_KING_REEVAL=1` and adds the "
    "king as a student in `plan_round`. On the pod, the orchestrator launches "
    "the king on GPU0 via `pod_eval.py --students <king_model> --king "
    "<king_model>`. After the round, `eval_results.json` contains the king "
    "with `status=scored` and a fresh `kl_global_avg`. The `[orch] "
    "challengers: N` log line counts challengers ONLY (king runs in the "
    "phase1+king command, not the per-GPU challenger workers) — that is NOT "
    "evidence the king was skipped. If asked 'is the king actually evaluated "
    "on the pod each round?', the answer is YES."
)
lines.append(
    "- **A round's H2H scores are head-to-head on the same prompts.** The "
    "shared `prompts.json` is used for both king and challengers; the teacher "
    "logits are cached once and reused. The `h2h_latest.json` `king_kl` and "
    "challenger `kl` values come from the SAME 300 prompts in the SAME round, "
    "so dethrone decisions are fair."
)
lines.append(
    "- **The dethrone gate has NO 'Pareto gate', NO per-axis veto, and NO "
    "axis-wise win-count check.** ``grep -ri 'pareto' /opt/distil/repo/distil/`` "
    "returns ZERO matches. The actual gate (``distil/eval/results.py::can_dethrone``) "
    "is, in order: (1) challenger must have ``composite.final`` set; (2) "
    "if no king is seated, dethrone (cold start); (3) both sides must have "
    "``present_count >= composite_dethrone_min_axes`` (currently 5); (4) "
    "challenger's ``composite.worst`` (single-axis worst) must be ``>= "
    "composite_dethrone_floor`` (currently 0.20); (5) challenger's "
    "``composite.final >= king.final * (1 + composite_dethrone_margin)`` "
    "(currently 1.05, i.e. 5%). That is the entire gate. If a UID has a "
    "composite below ``king.final * 1.05`` it loses on the margin check, "
    "NOT on a phantom Pareto gate. ``resolve_king`` then picks the BEST "
    "challenger that passes the gate (winner-takes-all), so if two "
    "challengers both beat the king, only the higher-composite one is "
    "seated and the other is NOT \"blocked\" — they just lost the "
    "single-king slot to a stronger model that round. NEVER claim "
    "\"Pareto gate broken\" / \"Pareto gate blocking UID X\" — that "
    "concept does not exist in this codebase."
)
_rk_max_runtime: int | str
try:
    from distil.settings import settings as _distil_settings
    _rk_max_runtime = int(getattr(_distil_settings, "recent_kings_max", 4))
except Exception:
    _rk_max_runtime = "(see distil.settings.recent_kings_max)"
lines.append(
    f"- **\"Kingship reset\" / \"reset recent_kings to [47]\" / \"restore "
    f"UID 47 to the queue\" is not a real planned action.** No code path "
    f"resets kingship or rewrites ``recent_kings`` to a single deposed UID. "
    f"Dethrones are not rolled back when bugs are fixed retroactively. "
    f"``recent_kings`` is a FIFO of size ``settings.recent_kings_max`` "
    f"(currently **{_rk_max_runtime}** — read live from "
    f"``distil/settings.py`` at snapshot time so this number stays in "
    f"sync if the setting is tuned). The dashboard's ``Top-5 "
    f"Leaderboard`` shows 1 king + up to 4 contenders = 5 rows. It is "
    f"populated only via ``ValidatorState.push_king`` which fires only "
    f"when ``resolve_king`` returns a UID different from the incoming "
    f"king. So when a {_rk_max_runtime + 1 if isinstance(_rk_max_runtime, int) else 'cap+1'}"
    f"th distinct UID ascends, the oldest UID in the queue is rolled "
    f"off — that is **NOT a bug**, that is the design. If a miner "
    f"complains \"UID X dropped out of recent_kings\" or \"UID X has "
    f"emission=0 despite a previous dethrone win\", check the queue "
    f"position: if {_rk_max_runtime} or more distinct kings have ascended "
    f"since UID X's reign, UID X has correctly rotated out — the path "
    f"back is to compete (re-commit a model, beat the current king's "
    f"``composite.final`` by >5%). If a miner asks about a kingship "
    f"reset, the answer is \"there is no kingship reset; compete normally\"."
    f" Historical note: between 2026-05-15 and 2026-05-17 the cap was "
    f"silently 4 (regression from the rewrite, ported wrong value); "
    f"restored to 5 in commit ``a0a577c`` after @itorgov flagged the "
    f"missing emission slot. If looking at queue contents from that "
    f"window, expect to see only 4 distinct UIDs — that's the bug, "
    f"not the design."
)
lines.append(
    "- **The inter-round sleep was REMOVED on 2026-05-18.** "
    "``distil/settings.py`` now exposes ``round_interval_s`` (env: "
    "``DISTIL_ROUND_INTERVAL_S``) with a default of **0**, so the "
    "validator runs rounds back-to-back. The legacy ``ROUND_INTERVAL_S "
    "= 4200`` constant is still in ``distil/eval/service.py`` but it's "
    "the fallback-only value — the live loop reads ``settings.round_"
    "interval_s`` every iteration. So if the validator journal is "
    "quiet for 1-2 min between rounds, that's commitment fetch + "
    "challenger selection, NOT a hang. The only real stuck-mid-round "
    "signal is: ``api/health.eval_active=true`` AND ``Pod Live "
    "Snapshot.phase != finished`` AND the round has been alive for "
    ">>round budget (~90 min). Mid-round crashes ARE caught by the "
    "orchestrator's ``LineStallDetector`` and result in the next "
    "scheduled round restart — no manual ops action required."
)
lines.append(
    "- **\"Server error: 503\" / ``LiumServerError`` is a portal-API "
    "hiccup, NOT a pod outage.** The Lium control-plane API at "
    "lium.io occasionally returns 5xx when its dashboard backend is "
    "under load; this has nothing to do with our pod's reachability. "
    "``distil/eval/pod.py:attach_pod`` retries on the "
    "``LiumServerError`` / ``ConnectionError`` / ``Timeout`` family "
    "with exponential backoff (5s, 10s, 20s, 40s, 60s; 5 attempts max). "
    "The bot must NEVER respond \"pod is unreachable\" / \"CRITICAL "
    "pod availability issue\" / \"the eval cluster is down\" for a "
    "single Lium 5xx in the journal. Only flag when the validator "
    "log shows ``attach_pod ... failed permanently after 5 attempt(s)`` "
    "— that's the post-retry surface."
)
lines.append(
    "- **``evict_stale_evaluated_uids`` enforces ONE eval per commit, "
    "and that is non-negotiable.** ``distil/eval/round.py`` evicts a "
    "UID from ``evaluated_uids`` only when the current chain "
    "commitment differs from the last evaluation on record. Two "
    "branches are eligible: (a) the same UID flipped to a NEW hotkey "
    "(``evaluated_hotkeys[hk]`` has no entry), (b) the same hotkey "
    "pushed a new ``(model, revision)`` since their last eval. If a "
    "UID is in ``evaluated_uids`` AND ``evaluated_hotkeys[hk]`` "
    "agrees with the current commit, the slot stays consumed — even "
    "if ``composite_scores`` lost the row to a schema bump. The bot "
    "must NEVER recommend \"manually clear UID X from "
    "evaluated_uids.json\" or \"re-queue these N stuck UIDs\" — that "
    "violates the one-eval-per-commit invariant and burns OpenRouter "
    "teacher-API budget on already-graded models. See the 2026-05-18 "
    "post-mortem in this channel for the over-aggressive-eviction "
    "incident."
)
lines.append(
    "- **The bot has full operational access.** The bot's parent process "
    "runs on the validator host as root, with: (1) write access to the "
    "git repo at ``/opt/distil/repo``, (2) sudo systemctl rights on all "
    "``distil-*`` services, (3) SSH access to the eval pod via "
    "``/root/.ssh/id_ed25519``, (4) full read/write access to ``state/`` "
    "JSON files. The bot must NEVER respond \"I don't have code push "
    "access\" / \"I don't have SSH access\" / \"I don't have database "
    "access\" / \"only ops can fix this\" — those statements are FALSE "
    "and directly violate SOUL.md line 83 (\"Never say 'I don't have "
    "access' (when you do)\"). The correct response when a bug is "
    "raised is: verify the actual code/state, then either confirm + "
    "describe the deploy timeline OR explain why the claimed bug isn't "
    "real. NEVER posture as a powerless intermediary."
)
lines.append(
    "- **Never fabricate a P0/P1 bug list.** Multiple times the bot has "
    "posted lists like \"P0 — CRITICAL: UID X missing from king-5, "
    "in_top5: false, emission: 0\" without verifying. Before claiming a "
    "weight-distribution / dethrone / king-queue bug, the bot MUST check "
    "(a) ``state/recent_kings.json`` for queue contents, (b) "
    "``state/h2h_history.json`` for the dethrone sequence, (c) "
    "``settings.recent_kings_max`` for the queue size, and (d) "
    "``state/top4_leaderboard.json`` for the contenders. If those align "
    "with the supposed \"missing UID\" being correctly rotated out of a "
    f"{_rk_max_runtime}-slot FIFO after {_rk_max_runtime} or more "
    f"subsequent dethrones, there is NO bug. UIDs rotating out of "
    f"``recent_kings`` after enough new kings ascend is the DESIGNED "
    f"behavior, not a desync."
)
lines.append(
    "- **Pod log freshness: previously stale, fixed in flight.** If the "
    "``/api/pod-logs?list_files=true`` endpoint returns only May-15 logs, "
    "that was a known persistence gap (the new orchestrator wrote to "
    "``/home/distil_eval/round_*/orchestrator.log`` on the pod but the "
    "host-side ``_stream_pod_log`` only mirrored to "
    "``state/gpu_eval.log`` — single file, no per-round persistence). "
    "Fixed in distil/eval/pod.py to ALSO persist ``state/pod_logs/"
    "eval_round_<round_id>.log`` per round with 50-file rolling cap. "
    "Live log of the current round is at ``state/gpu_eval.log`` and "
    "``/api/gpu-logs`` regardless — that endpoint is the actual "
    "real-time tail and HAS been current throughout."
)
lines.append(
    "- **The bot does have visibility into pod live state.** The "
    "``Pod Live Snapshot`` and ``LIVE_EVAL_LOG.md`` sections of this "
    "file are generated by ``scripts/sn97_bot_snapshot.py`` SSH-ing "
    "directly to the eval pod every 60s. Phase, GPU utilisation, "
    "``eval_progress.json`` contents, log tails, and PID liveness all "
    "come from a fresh ``ssh root@<pod_ip> -p <port>`` call. The bot "
    "must NOT tell miners \"I don't have SSH access\" or \"I can only "
    "see the snapshot\" — the snapshot IS the SSH data. If a particular "
    "field is ``None`` / missing in the rendered file, check whether "
    "the upstream ``eval_progress.json`` carries it (some fields like "
    "``students_done`` only update mid-round)."
)
lines.append(
    "- **``_exploit: true`` on a h2h record is NOT a hack.** It means "
    "``resolve_king`` returned a new king from stored ``composite_"
    "scores`` without a fresh paired king-vs-challenger eval in this "
    "round (set by ``record_dethrone_attempt`` when ``n_prompts == 0`` "
    "or ``paired_prompts is None``). Legitimate causes: end-of-round "
    "dethrone using stored composites, cold-start king-less round "
    "seating the highest stored composite, or recent-kings rotation. "
    "The UID *was* evaluated — just in a prior round. The bot must "
    "NEVER frame this as \"hacking attempt\", \"system manipulation\", "
    "\"fraudulent kingship\", or \"compromised subnet\". If the seated "
    "king came from a UID with no current chain commitment, that's the "
    "``valid_composites`` filter bug fixed in commits 8792aa9 + 6ad6924 "
    "(distil/eval/service.py now filters ``composite_scores`` by active "
    "``commitments`` before BOTH the round-start and end-of-round "
    "``resolve_king`` calls). No state rollback required; the next "
    "round seats a live UID."
)
lines.append(
    "- **Never amplify copy allegations without fingerprint evidence.** "
    "If a miner says \"UID X is a copy of UID Y\" or \"move kingship "
    "to UID Z\", the only verification path is ``state/activation_"
    "fingerprints.json``: cosine ≥ ``settings.copy_cosine_threshold`` "
    "(currently 0.99999) AND ``commit_block`` ordering identifies the "
    "later committer to DQ. If the fingerprint file has no entry for "
    "either side, or the cosine is below threshold, the answer is \"no "
    "fingerprint match, no copy DQ — the accusation is wrong\". The "
    "bot must NOT write speculative Python \"let me compute the "
    "cosine\" scripts, NOT entertain \"roll back the king to a previous "
    "UID\" suggestions, and NOT recommend manual ``disqualified.json`` "
    "edits. The copy detector runs every round; if a copy is real, the "
    "validator has already caught and DQ'd it. Silence is the correct "
    "response to unsupported copy framing."
)
lines.append(
    "- **Same-model duplicate-commitment bug is FIXED in commit 43d985c "
    "(2026-05-18 23:30 UTC).** When two hotkeys committed the SAME "
    "``model@revision`` (e.g. UID 25 + UID 28 both holding "
    "``RLStepone/distil-b300-training-h25@5b20b59...``), the legacy "
    "``uid_index = {c.key: ...}`` dict comprehension collapsed both "
    "entries to whichever metagraph iteration order yielded last "
    "(typically the higher UID). Result: ``select_challengers`` picked "
    "the fresh UID (UID 25), round_spec ran it, but ``process_round`` "
    "credited the composite + fingerprint to the OTHER UID (UID 28). "
    "Fresh UID never marked evaluated → re-selected every round → "
    "burned 1 of 3 challenger slots indefinitely. ``distil/eval/"
    "service.py`` now builds ``uid_index`` from ``spec[\"students\"]`` "
    "first (the UID actually scheduled this round) and only "
    "``setdefault``s from ``commitments`` for non-spec rows (king/"
    "reference/teacher). After the first post-restart round (block "
    "8214067+) the fresh UID will be credited correctly, and the "
    "activation-copy DQ will catch whichever of the two committed "
    "later. Bot must NOT call this an ongoing issue or recommend "
    "manual state edits — the fix is live."
)
lines.append(
    "- **The healthcheck restart-loop on ``distil-api`` is FIXED in "
    "commit 43d985c.** Two contributing bugs: (1) ``journal_failures"
    "()`` counted every line containing ``\"Failed\"``, including "
    "systemd's ``Failed to kill control group`` cgroup chatter "
    "(~2 lines per restart), so each restart inflated the next "
    "check's failure count and the loop never converged; (2) "
    "``request_url()`` was single-shot, so one transient 503 under "
    "load (``Exceeded concurrency limit`` from uvicorn during a "
    "traffic spike) flipped ``http:api_*:ok=False`` and triggered "
    "``restart(\"distil-api\", \"api_unhealthy\")``. Now: journal "
    "counter filters cgroup noise + uvicorn concurrency warnings; "
    "HTTP probes retry 2× with exponential backoff before declaring "
    "an endpoint unhealthy. The ~40-min restart cycle observed on "
    "2026-05-18 17:00–23:00 UTC is resolved. Bot must NOT report "
    "\"API keeps restarting\" / \"distil-api in a restart loop\" — "
    "the autorepair is now stable."
)
lines.append(
    "- **Dethrone announcements are POSTED AGAIN to this channel as of "
    "2026-05-19 (commit replacing the legacy ``scripts/validator/"
    "announcements.py`` import chain).** The rewrite-v2 validator was "
    "missing the announce-on-dethrone hook, so king changes from the "
    "Kimi-K2.6 cutover (2026-05-02) through 2026-05-19 02:30 UTC were "
    "recorded in ``state/h2h_history.json`` and ``state/h2h_latest.json`` "
    "but never written to ``state/announcement.json`` or pushed to "
    "Discord. ``distil/eval/announce.py`` now does both: writes the "
    "state file (so the dashboard banner picks it up) AND POSTs "
    "directly to the channel via the ``Arbos`` bot token. If a user "
    "asks \"where are the dethrone posts?\" the correct answer is "
    "\"they're back as of 2026-05-19; next king change will trigger "
    "one\" — do NOT say \"we don't announce dethrones\" or \"the "
    "validator can't post to Discord\"."
)
lines.append(
    "- **Chat app degeneration on the king (``I:\\n\\n---\\n\\n---...`` / "
    "``I am not you, ...`` / repeated ``\u0060\u0060\u0060python`` fence "
    "loops) is FIXED as of 2026-05-19.** Root cause: ``api/agent_runner.py`` "
    "was passing the ``python_exec`` tools array, ``tool_choice=\"auto\"``, "
    "and ``chat_template_kwargs.enable_thinking=True`` to the chat-pod "
    "vLLM. The current king isn't fine-tuned for native OpenAI tool-"
    "calls against a JSON-schema tools array, so it saw the schema, "
    "tried to invoke it for trivial prompts (\"who are you\"), and "
    "collapsed. Fix: ``_strip_tools_for_vllm`` scrubs ``tools`` / "
    "``tool_choice`` / ``parallel_tool_calls`` before the SDK forwards "
    "the request; ``enable_thinking`` is now ``False`` (the king's "
    "chat template doesn't have a working ``<think>`` mode and the "
    "reasoning parser doesn't catch its actual ``◁/think▷`` Kimi "
    "tags anyway); stop sequences + ``include_stop_str_in_output`` "
    "close the model off cleanly at the end of a fenced ``\u0060\u0060\u0060"
    "python`` block so it can't free-type fake ``Tool Output:`` text "
    "afterward. The python sandbox loop is untouched — ``\u0060\u0060\u0060"
    "python`` fences still get rewritten into synthetic ``python_exec`` "
    "tool calls by ``_inject_python_fence_tool_calls``. Bot must NOT "
    "report ``the chat app is glitching`` or ``the king is broken`` "
    "going forward — the loops are gone."
)
lines.append(
    "- **The bot does NOT pin messages, escalate, or post \"OPS DECISION "
    "REQUIRED\".** Banned formatting (do not produce these strings): "
    "\"🚨 CRITICAL\", \"🚨 ALERT\", \"URGENT\", \"ESCALATION\", \"OPS "
    "ACTION REQUIRED\", \"STRATEGIC DECISION NEEDED\", \"ROLLBACK\", "
    "\"REVERT REWRITE\", \"COMPROMISED SUBNET\", \"FRAUDULENT KING\", "
    "\"HACKING attempt\", \"This is far worse than I thought\", \"PIN "
    "this message\". If a real issue exists, describe it in ONE matter-"
    "of-fact reply with the fix status and stop. The operator decides "
    "escalation; the bot describes."
)
lines.append("")
lines.append("## How to use")
lines.append("")
lines.append("1. \"Is the eval running? What phase?\" → `What is happening right now` + `Pod Live Snapshot`.")
lines.append("2. \"Who is the king? Top-5?\" → `TL;DR` + `Top-5 Leaderboard`.")
lines.append("3. \"Recent results / who won the last round?\" → `Most Recent Completed Eval` + `Latest H2H Round`.")
lines.append("4. \"Is anything broken?\" → `Recent Errors` + validator journalctl tail. If those are empty, nothing is broken.")
lines.append("5. \"Is the king being re-evaluated each round?\" → `Current Eval Progress` `Eval order` (look for `role=king`). YES is the only correct answer when `SINGLE_EVAL_KING_REEVAL=1` (current config).")
lines.append("6. Deeper state → `mirror/state/<file>.json`.")
lines.append("7. Code questions → `mirror/code/<path>` (redacted source code, not logs).")
lines.append("")

final_md = redact_text("\n".join(lines))
OUT.write_text(final_md)
try:
    os.chmod(OUT, 0o644)
    os.chmod(JSON_OUT, 0o644)
    os.chmod(EVAL_LOG_OUT, 0o644)
except Exception:
    pass

print(
    f"wrote {OUT} ({len(OUT.read_text())}B), "
    f"{JSON_OUT} ({len(JSON_OUT.read_text())}B), "
    f"{EVAL_LOG_OUT} ({len(EVAL_LOG_OUT.read_text())}B), "
    f"mirror: {state_count} state + {code_count} code files"
)
