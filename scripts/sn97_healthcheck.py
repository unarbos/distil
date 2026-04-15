#!/usr/bin/env python3
"""Deterministic healthcheck and safe repair helper for Distil ops."""

from __future__ import annotations

import argparse
import json
import os
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
        "issues": issues,
    }
    payload["healthy"] = not issues
    return payload


def repair(report: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    services = report["services"]
    http_status = report["http"]

    def restart(unit: str, reason: str) -> None:
        result = run(["systemctl", "restart", unit], timeout=30)
        if result.returncode == 0:
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

    if args.format == "markdown":
        print(render_markdown(after))
    else:
        print(json.dumps(after, indent=2, sort_keys=True))
    return 0 if after["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
