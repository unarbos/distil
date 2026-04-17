#!/usr/bin/env python3
"""Deterministic healthcheck and safe repair helper for Distil ops."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(os.environ.get("DISTIL_REPO_ROOT", "/opt/distil/repo"))
STATE_DIR = Path(os.environ.get("DISTIL_STATE_DIR", str(REPO_ROOT / "state")))
OPEN_WEBUI_CONTAINER = os.environ.get("DISTIL_CHAT_CONTAINER", "open-webui")
MAX_EVAL_STALENESS_SEC = int(os.environ.get("DISTIL_MAX_EVAL_STALENESS_SEC", str(3 * 60 * 60)))
GPU_LOG_STALE_SEC = int(os.environ.get("DISTIL_GPU_LOG_STALE_SEC", str(15 * 60)))
DISK_WARN_PCT = int(os.environ.get("DISTIL_DISK_WARN_PCT", "80"))
DISK_FAIL_PCT = int(os.environ.get("DISTIL_DISK_FAIL_PCT", "90"))
INCIDENT_LOG = Path(os.environ.get("DISTIL_INCIDENT_LOG", str(STATE_DIR / "incidents.jsonl")))
RESTART_BUDGET_PATH = STATE_DIR / "restart_budget.json"
RESTART_BUDGET_MAX = int(os.environ.get("DISTIL_RESTART_BUDGET_MAX", "2"))
RESTART_BUDGET_WINDOW_SEC = int(os.environ.get("DISTIL_RESTART_BUDGET_WINDOW_SEC", "3600"))
JOURNAL_FAIL_THRESHOLD = int(os.environ.get("DISTIL_JOURNAL_FAIL_THRESHOLD", "5"))
CHAT_POD_HOST = os.environ.get("DISTIL_CHAT_POD_HOST", "")
CHAT_POD_SSH_PORT = os.environ.get("DISTIL_CHAT_POD_SSH_PORT", "22")
CHAT_POD_APP_PORT = os.environ.get("DISTIL_CHAT_POD_APP_PORT", "8000")

SERVICE_UNITS = {
    "validator": "distil-validator",
    "api": "distil-api",
    "dashboard": "distil-dashboard",
    "benchmark_timer": "distil-benchmark-sync.timer",
    "chat_tunnel": "chat-tunnel",
    "caddy": "caddy",
}

LOCAL_ENDPOINTS = {
    "api_local": "http://127.0.0.1:3710/api/health",
    "dashboard_local": "http://127.0.0.1:3720/",
}

PUBLIC_ENDPOINTS = {
    "api_public": "https://api.arbos.life/api/health",
    "dashboard_public": "https://distil.arbos.life/",
    "chat_public": "https://chat.arbos.life/",
}


def run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def request_url(url: str, timeout: int = 10) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "sn97-healthcheck/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "body_excerpt": body[:400],
                "body_bytes": len(body.encode("utf-8", errors="ignore")),
                "json": maybe_json(body),
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "body_excerpt": body[:400],
            "body_bytes": len(body.encode("utf-8", errors="ignore")),
            "json": maybe_json(body),
        }
    except URLError as exc:
        text = str(exc)
        return {
            "ok": False,
            "status": None,
            "body_excerpt": text[:400],
            "body_bytes": len(text.encode("utf-8", errors="ignore")),
            "json": None,
        }


def systemd_state(unit: str) -> dict[str, Any]:
    active = run(["systemctl", "is-active", unit], timeout=10)
    enabled = run(["systemctl", "is-enabled", unit], timeout=10)
    return {
        "unit": unit,
        "active": active.stdout.strip(),
        "enabled": enabled.stdout.strip() if enabled.returncode == 0 else enabled.stderr.strip() or enabled.stdout.strip(),
        "ok": active.stdout.strip() == "active",
    }


def git_revision(repo_root: Path) -> dict[str, Any]:
    git = run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], timeout=10)
    revision_file = repo_root / "REVISION"
    revision_text = revision_file.read_text().strip() if revision_file.exists() else None
    return {
        "git": git.stdout.strip() if git.returncode == 0 else None,
        "file": revision_text or None,
        "ok": git.returncode == 0 or bool(revision_text),
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def inspect_open_webui() -> dict[str, Any]:
    result = run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            OPEN_WEBUI_CONTAINER,
        ],
        timeout=15,
    )
    if result.returncode != 0:
        return {
            "container": OPEN_WEBUI_CONTAINER,
            "exists": False,
            "running": False,
            "health": "missing",
            "ok": False,
            "stderr": result.stderr.strip(),
        }
    parts = result.stdout.strip().split()
    running = parts[0].lower() == "true" if parts else False
    health = parts[1] if len(parts) > 1 else "none"
    return {
        "container": OPEN_WEBUI_CONTAINER,
        "exists": True,
        "running": running,
        "health": health,
        "ok": running and health in {"healthy", "none"},
    }


def disk_usage(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        try:
            st = shutil.disk_usage(p)
            pct = int(round((st.used / st.total) * 100)) if st.total else 0
            out[str(p)] = {"used_pct": pct, "total_gb": round(st.total / 1e9, 1), "free_gb": round(st.free / 1e9, 1)}
        except Exception as exc:
            out[str(p)] = {"error": str(exc)}
    return out


def inode_usage(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        result = run(["df", "-i", str(p)], timeout=10)
        if result.returncode != 0:
            out[str(p)] = {"error": result.stderr.strip() or result.stdout.strip()}
            continue
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if len(lines) < 2:
            out[str(p)] = {"error": "no_data"}
            continue
        parts = lines[-1].split()
        pct = 0
        for tok in parts:
            if tok.endswith("%"):
                try:
                    pct = int(tok.rstrip("%"))
                except ValueError:
                    pass
        out[str(p)] = {"used_pct": pct}
    return out


def gpu_log_probe() -> dict[str, Any]:
    log = STATE_DIR / "gpu_eval.log"
    if not log.exists():
        return {"ok": True, "reason": "missing"}
    try:
        mtime = int(log.stat().st_mtime)
        tail = subprocess.run(["tail", "-n", "200", str(log)], capture_output=True, text=True, timeout=5).stdout
    except Exception as exc:
        return {"ok": False, "reason": f"read_error:{exc}"}
    hits = [m.group(0) for m in re.finditer(r"(?i)\b(OOM|Died|CUDA error|segmentation fault|out of memory)\b", tail)]
    return {
        "ok": not hits,
        "mtime": mtime,
        "age_sec": int(time.time()) - mtime,
        "hits": hits[:5],
    }


def chat_pod_probe(expected_model: str | None) -> dict[str, Any]:
    if not CHAT_POD_HOST:
        return {"ok": True, "skipped": True, "reason": "no_host"}
    ssh = [
        "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes", "-p", CHAT_POD_SSH_PORT, f"root@{CHAT_POD_HOST}",
    ]
    nvsmi = run(ssh + ["nvidia-smi", "--query-gpu=name,utilization.gpu", "--format=csv,noheader"], timeout=15)
    models = run(ssh + [f"curl -fsS http://localhost:{CHAT_POD_APP_PORT}/v1/models"], timeout=15)
    served = None
    if models.returncode == 0:
        data = maybe_json(models.stdout) or {}
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list) and items:
            served = items[0].get("id") if isinstance(items[0], dict) else None
    model_ok = expected_model is None or (served is not None and served == expected_model)
    return {
        "ok": nvsmi.returncode == 0 and models.returncode == 0 and model_ok,
        "nvidia_smi": nvsmi.stdout.strip() if nvsmi.returncode == 0 else None,
        "served_model": served,
        "expected_model": expected_model,
        "model_match": model_ok,
    }


def chain_weight_sanity(expected_king_uid: int | None) -> dict[str, Any]:
    if expected_king_uid is None:
        return {"ok": True, "skipped": True, "reason": "no_king"}
    if not os.environ.get("VALIDATOR_UID"):
        return {"ok": True, "skipped": True, "reason": "no_validator_uid"}
    probe_script = (
        "import os,sys,json;"
        "sys.path.insert(0, '/opt/distil/repo');"
        "import bittensor as bt;"
        "from eval.chain import get_validator_weight_target;"
        "net=os.environ.get('BT_NETWORK','finney');"
        "netuid=int(os.environ.get('BT_NETUID','97'));"
        "vuid=int(os.environ.get('VALIDATOR_UID','-1'));"
        "st=bt.subtensor(network=net);"
        "t=get_validator_weight_target(st, netuid, vuid) if vuid>=0 else None;"
        "print(json.dumps({'target': t}))"
    )
    venv = "/opt/distil/venv/bin/python"
    py = venv if Path(venv).exists() else "python3"
    result = run([py, "-c", probe_script], timeout=40)
    if result.returncode != 0:
        return {"ok": False, "reason": (result.stderr.strip() or result.stdout.strip())[:400]}
    data = maybe_json(result.stdout.strip().splitlines()[-1] if result.stdout else "")
    target = (data or {}).get("target")
    return {
        "ok": target == expected_king_uid,
        "chain_target_uid": target,
        "expected_king_uid": expected_king_uid,
    }


def journal_failures(units: list[str], since: str = "1 hour ago") -> dict[str, int]:
    out: dict[str, int] = {}
    for unit in units:
        result = run(
            ["journalctl", "-u", unit, "--since", since, "--output=short", "--no-pager"],
            timeout=15,
        )
        if result.returncode != 0:
            out[unit] = -1
            continue
        out[unit] = sum(1 for line in result.stdout.splitlines() if "Failed" in line or "failed with result" in line)
    return out


def _load_json_safe(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_json_safe(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception:
        pass


def restart_budget_check(unit: str) -> tuple[bool, int]:
    now = int(time.time())
    budget = _load_json_safe(RESTART_BUDGET_PATH, {}) or {}
    hist = [t for t in budget.get(unit, []) if now - int(t) <= RESTART_BUDGET_WINDOW_SEC]
    return (len(hist) < RESTART_BUDGET_MAX, len(hist))


def restart_budget_record(unit: str) -> None:
    now = int(time.time())
    budget = _load_json_safe(RESTART_BUDGET_PATH, {}) or {}
    hist = [t for t in budget.get(unit, []) if now - int(t) <= RESTART_BUDGET_WINDOW_SEC]
    hist.append(now)
    budget[unit] = hist
    _save_json_safe(RESTART_BUDGET_PATH, budget)


def append_incidents(events: list[dict[str, Any]]) -> None:
    if not events:
        return
    try:
        INCIDENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INCIDENT_LOG.open("a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def collect() -> dict[str, Any]:
    now = int(time.time())
    service_states = {name: systemd_state(unit) for name, unit in SERVICE_UNITS.items()}
    local_http = {name: request_url(url) for name, url in LOCAL_ENDPOINTS.items()}
    public_http = {name: request_url(url) for name, url in PUBLIC_ENDPOINTS.items()}
    revision = git_revision(REPO_ROOT)
    open_webui = inspect_open_webui()

    health_json = local_http["api_local"].get("json") or {}
    h2h_latest = read_json(STATE_DIR / "h2h_latest.json") or {}
    eval_progress = read_json(STATE_DIR / "eval_progress.json") or {}
    validator_log = read_json(STATE_DIR / "validator_log.json") or []

    eval_summary = {}
    if isinstance(eval_progress, dict):
        eval_summary = {
            "active": bool(eval_progress.get("active")),
            "phase": eval_progress.get("phase"),
            "students_total": eval_progress.get("students_total"),
            "students_done": len(eval_progress.get("completed", [])) if isinstance(eval_progress.get("completed"), list) else None,
            "prompts_total": eval_progress.get("prompts_total"),
            "teacher_prompts_done": eval_progress.get("teacher_prompts_done"),
            "current_student": (eval_progress.get("current") or {}).get("student_name") if isinstance(eval_progress.get("current"), dict) else None,
            "current_prompt": (eval_progress.get("current") or {}).get("prompts_done") if isinstance(eval_progress.get("current"), dict) else None,
        }

    h2h_summary = {}
    if isinstance(h2h_latest, dict):
        h2h_summary = {
            "type": h2h_latest.get("type"),
            "block": h2h_latest.get("block"),
            "timestamp": h2h_latest.get("timestamp"),
            "king_uid": h2h_latest.get("king_uid"),
            "king_model": h2h_latest.get("king_model"),
            "king_changed": h2h_latest.get("king_changed"),
            "new_king_uid": h2h_latest.get("new_king_uid"),
            "p_value": h2h_latest.get("p_value"),
            "n_prompts": h2h_latest.get("n_prompts"),
        }

    validator_log_mtime = None
    validator_log_path = STATE_DIR / "validator_log.json"
    if validator_log_path.exists():
        validator_log_mtime = int(validator_log_path.stat().st_mtime)

    issues: list[str] = []
    critical_files = {
        "scores.json": (STATE_DIR / "scores.json").exists(),
        "disqualified.json": (STATE_DIR / "disqualified.json").exists(),
        "h2h_latest.json": (STATE_DIR / "h2h_latest.json").exists(),
        "announcement.json": (STATE_DIR / "announcement.json").exists(),
    }

    for name, state in service_states.items():
        if not state["ok"]:
            issues.append(f"service:{name}:{state['active']}")

    for name, response in local_http.items():
        if not response["ok"]:
            issues.append(f"http:{name}:{response['status']}")

    for name, response in public_http.items():
        if not response["ok"]:
            issues.append(f"http:{name}:{response['status']}")

    if not open_webui["ok"]:
        issues.append(f"container:{OPEN_WEBUI_CONTAINER}:{open_webui['health']}")

    for filename, exists in critical_files.items():
        if not exists:
            issues.append(f"state:{filename}:missing")

    if eval_progress.get("active") and validator_log_mtime:
        age_sec = now - validator_log_mtime
        if age_sec > MAX_EVAL_STALENESS_SEC:
            issues.append(f"validator:stale_eval_progress:{age_sec}")

    disk = disk_usage([REPO_ROOT, Path("/")])
    for mount, info in disk.items():
        pct = info.get("used_pct") if isinstance(info, dict) else None
        if pct is None:
            continue
        if pct >= DISK_FAIL_PCT:
            issues.append(f"disk:{mount}:{pct}")
        elif pct >= DISK_WARN_PCT:
            issues.append(f"disk_warn:{mount}:{pct}")
    inodes = inode_usage([REPO_ROOT])
    for mount, info in inodes.items():
        pct = info.get("used_pct") if isinstance(info, dict) else None
        if pct is not None and pct >= DISK_FAIL_PCT:
            issues.append(f"inode:{mount}:{pct}")

    gpu_probe = gpu_log_probe()
    if not gpu_probe.get("ok") and gpu_probe.get("hits"):
        issues.append(f"gpu:log_error:{len(gpu_probe['hits'])}")
    if eval_progress.get("active") and isinstance(gpu_probe.get("age_sec"), int) and gpu_probe["age_sec"] > GPU_LOG_STALE_SEC:
        issues.append(f"gpu:log_stale:{gpu_probe['age_sec']}")

    king_model = (h2h_latest or {}).get("king_model") if isinstance(h2h_latest, dict) else None
    king_uid = (h2h_latest or {}).get("king_uid") if isinstance(h2h_latest, dict) else None
    chat_probe = chat_pod_probe(king_model)
    if not chat_probe.get("ok") and not chat_probe.get("skipped"):
        if chat_probe.get("model_match") is False:
            issues.append(f"chat:model_mismatch:{chat_probe.get('served_model')}")
        else:
            issues.append("chat:pod_probe_failed")

    weight_check = chain_weight_sanity(king_uid)
    if not weight_check.get("ok") and not weight_check.get("skipped"):
        issues.append(f"chain:weight_mismatch:{weight_check.get('chain_target_uid')}")

    failures = journal_failures([u for u in SERVICE_UNITS.values() if not u.endswith(".timer")])
    for unit, count in failures.items():
        if count >= JOURNAL_FAIL_THRESHOLD:
            issues.append(f"journal:{unit}:failures:{count}")

    payload = {
        "timestamp": now,
        "hostname": socket.gethostname(),
        "repo_root": str(REPO_ROOT),
        "state_dir": str(STATE_DIR),
        "revision": revision,
        "services": service_states,
        "http": {**local_http, **public_http},
        "open_webui": open_webui,
        "validator": {
            "health": health_json,
            "eval_progress": eval_summary,
            "h2h_latest": h2h_summary,
            "validator_log_entries": len(validator_log) if isinstance(validator_log, list) else None,
            "validator_log_mtime": validator_log_mtime,
        },
        "critical_files": critical_files,
        "disk": disk,
        "inodes": inodes,
        "gpu_log": gpu_probe,
        "chat_pod": chat_probe,
        "chain_weight": weight_check,
        "journal_failures": failures,
        "issues": issues,
    }
    payload["healthy"] = not issues
    return payload


def repair(report: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    services = report["services"]
    http_status = report["http"]
    restarted: set[str] = set()

    def restart(unit: str, reason: str) -> None:
        if unit in restarted:
            return
        allowed, used = restart_budget_check(unit)
        if not allowed:
            actions.append(f"budget_exceeded:{unit}:{reason}:{used}/{RESTART_BUDGET_MAX}")
            return
        result = run(["systemctl", "restart", unit], timeout=30)
        if result.returncode == 0:
            restart_budget_record(unit)
            restarted.add(unit)
            actions.append(f"restarted:{unit}:{reason}")
        else:
            actions.append(f"failed_restart:{unit}:{reason}:{result.stderr.strip() or result.stdout.strip()}")

    for name in ("validator", "api", "dashboard", "chat_tunnel", "caddy"):
        if not services[name]["ok"]:
            restart(services[name]["unit"], "inactive")

    if not services["benchmark_timer"]["ok"]:
        result = run(["systemctl", "enable", "--now", services["benchmark_timer"]["unit"]], timeout=30)
        if result.returncode == 0:
            actions.append("enabled:distil-benchmark-sync.timer")
        else:
            actions.append(f"failed_enable:distil-benchmark-sync.timer:{result.stderr.strip() or result.stdout.strip()}")

    if not http_status["api_local"]["ok"] or not http_status["api_public"]["ok"]:
        restart("distil-api", "api_unhealthy")

    if not http_status["dashboard_local"]["ok"] or not http_status["dashboard_public"]["ok"]:
        restart("distil-dashboard", "dashboard_unhealthy")

    if not http_status["chat_public"]["ok"] or not report["open_webui"]["ok"]:
        result = run(["docker", "restart", OPEN_WEBUI_CONTAINER], timeout=45)
        if result.returncode == 0:
            actions.append(f"restarted:{OPEN_WEBUI_CONTAINER}:chat_unhealthy")
        else:
            actions.append(f"failed_restart:{OPEN_WEBUI_CONTAINER}:{result.stderr.strip() or result.stdout.strip()}")
        restart("chat-tunnel", "chat_unhealthy")
        restart("caddy", "chat_unhealthy")

    if any(issue.startswith("validator:stale_eval_progress:") for issue in report["issues"]):
        restart("distil-validator", "stale_eval_progress")

    if any(issue.startswith("gpu:log_stale:") for issue in report["issues"]):
        restart("distil-validator", "gpu_log_stale")

    if any(issue.startswith("chat:model_mismatch:") for issue in report["issues"]):
        actions.append("notify:chat:model_mismatch")

    if any(issue.startswith("chain:weight_mismatch:") for issue in report["issues"]):
        actions.append("notify:chain:weight_mismatch")

    if any(issue.startswith("disk:") for issue in report["issues"]):
        actions.append("notify:disk:full")

    for issue in report["issues"]:
        if issue.startswith("journal:"):
            actions.append(f"notify:{issue}")

    return actions


def render_markdown(report: dict[str, Any]) -> str:
    health = report["validator"]["health"] or {}
    checks = [
        ("Code revision", report["revision"].get("git") or report["revision"].get("file") or "unknown"),
        ("Validator", report["services"]["validator"]["active"]),
        ("API", f"{report['services']['api']['active']} / {report['http']['api_local']['status']}"),
        ("Dashboard", f"{report['services']['dashboard']['active']} / {report['http']['dashboard_local']['status']}"),
        ("Chat", f"{report['services']['chat_tunnel']['active']} / {report['http']['chat_public']['status']}"),
        ("King", str(health.get("king_uid")) if health else "unknown"),
        ("Eval active", str(bool(health.get("eval_active"))) if health else "unknown"),
    ]
    lines = [
        f"# SN97 Healthcheck — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(report['timestamp']))}",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    for key, value in checks:
        lines.append(f"| {key} | {value} |")
    lines.append("")
    if report.get("actions"):
        lines.append("## Actions")
        for action in report["actions"]:
            lines.append(f"- {action}")
        lines.append("")
    if report["issues"]:
        lines.append("## Remaining Issues")
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("All checks green.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Distil ops healthcheck and safe repair helper.")
    parser.add_argument("--repair", action="store_true", help="Attempt safe service-level repairs before reporting.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    before = collect()
    actions: list[str] = []
    if args.repair and before["issues"]:
        actions = repair(before)
    after = collect()
    after["actions"] = actions
    after["issues_before_repair"] = before["issues"]
    after["healthy_after_repair"] = after["healthy"]

    now = int(time.time())
    events: list[dict[str, Any]] = []
    for issue in before["issues"]:
        resolved = issue not in after["issues"]
        events.append({"ts": now, "type": "issue", "issue": issue, "resolved": resolved})
    for action in actions:
        events.append({"ts": now, "type": "action", "action": action})
    append_incidents(events)

    if args.format == "markdown":
        print(render_markdown(after))
    else:
        print(json.dumps(after, indent=2, sort_keys=True))
    return 0 if after["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
