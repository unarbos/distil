#!/usr/bin/env python3
"""Idempotent configurator that wires the chat.arbos.life Open-WebUI deployment
for end-to-end tool calling against the SN97 king model.

What this does (each step is safe to re-run):

  1. Generates OpenAI-format tool specs for ``openwebui_tools/sn97_status.py`` by
     invoking Open-WebUI's own ``get_tool_specs`` helper inside the running
     ``open-webui`` Docker container. This guarantees the schema matches what
     the live runtime will accept.
  2. Inserts (or updates) a row in the ``tool`` table for the SN97 status
     toolkit, owned by the local admin user. Old rows are replaced — the
     module is the single source of truth.
  3. Updates the ``sn97-king`` model row:
        - sets ``params.function_calling = "native"`` so Open-WebUI passes
          ``tools[]`` to vLLM (which uses the ``qwen3_xml`` parser to extract
          tool calls back into OpenAI ``tool_calls`` format),
        - prepends a tools-aware system prompt so the 4B king actually picks
          up new affordances, while preserving the existing math-formatting
          rules,
        - attaches ``meta.toolIds`` so the SN97 status toolkit is on by
          default for every chat,
        - keeps ``meta.builtinTools`` aligned with the global config (only
          flips on the things whose backends are actually wired up).
  4. Patches the global Open-WebUI ``config`` JSON to enable the built-in
     tools we want available without per-user setup:
        - ``rag.web.search`` with the DuckDuckGo backend (no API key needed),
        - code interpreter (Pyodide, browser-side, no API key needed).
     Image generation is intentionally left off — there is no provider
     configured and an OpenAI key would be required for DALL-E.

Flags:
  --dry-run    Print what would change but don't write.

Layout assumes the host paths used on this validator:
  - DB:       /opt/distil/chat/webui-data/webui.db
  - Tool src: scripts/chat_pod/openwebui_tools/*.py
  - Container: ``open-webui`` (host network mode, port 3730)

Run:
  sudo /opt/distil/venv/bin/python scripts/chat_pod/configure_webui_tools.py

After running, restart the container so the global config + tool registry are
re-loaded:
  docker restart open-webui
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "scripts" / "chat_pod" / "openwebui_tools"
DB_PATH = Path(os.environ.get("WEBUI_DB", "/opt/distil/chat/webui-data/webui.db"))
CONTAINER = os.environ.get("WEBUI_CONTAINER", "open-webui")
MODEL_ID = "sn97-king"
TOOL_ID = "sn97_status"

ADMIN_EMAIL = "admin@localhost"
NOW = int(time.time())
# config.updated_at / created_at are SQLAlchemy DATETIME columns that read
# back through ``str_to_datetime`` — they MUST be ISO strings, not integers.
# tool/model use plain INTEGER timestamps, so they get NOW directly.
NOW_ISO = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# 2026-05-02: system prompt. The marker lets us update it idempotently —
# re-running this script will replace the prompt, never stack copies on top
# of each other.
#
# v7 changes (over v6):
#  - One line. Empirically the distilled-student kings (e.g.
#    ``bodenmaurice/distil-new-v16``) regurgitate or pattern-match
#    against ANY system prompt longer than that:
#      * v5 (~80 lines) → 122k-char loops repeating prompt phrases
#      * v6 (~12 lines) → "Hi" returned 2.4k chars of pseudo-Pyodide
#        instructions (chat 297a48c3 on 2026-05-05 18:14)
#      * v7 (1 line)    → minimal regurgitation surface
#  - The Pyodide / "Persistent File System" / "Do not install packages"
#    text the model was emitting wasn't even from OUR prompt — Open-WebUI
#    auto-injects ``CODE_INTERPRETER_PYODIDE_PROMPT`` (defined in
#    open_webui/config.py) as a user message before every turn whenever
#    code_interpreter is enabled on the model row, even with native
#    function calling. The companion fix in this commit disables
#    ``meta.builtinTools.code_interpreter`` on sn97-king so OWUI stops
#    that injection. The current king is too weak to use the code
#    interpreter productively anyway — re-enable per-king when we get
#    a stronger king.
SYSTEM_PROMPT_MARKER = "<!--sn97-system-v7-minimal-->"
SYSTEM_PROMPT = (
    f"{SYSTEM_PROMPT_MARKER}\n"
    "You are the SN97 (Distil) chat assistant. Be concise, accurate, "
    "and helpful. Use tools when available."
)


def fail(msg: str) -> None:
    print(f"[configure_webui_tools] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[configure_webui_tools] {msg}")


# ---------------------------------------------------------------------------
# Step 1: generate specs by exec'ing Open-WebUI's helper inside the container
# ---------------------------------------------------------------------------
def _generate_specs(tool_path: Path) -> list[dict]:
    """Exec inside the open-webui container so we use the exact spec
    generator the live runtime uses (langchain pydantic conversion + tool
    schema cleanup). The spec list is returned as JSON over stdout."""
    if not tool_path.is_file():
        fail(f"tool source not found: {tool_path}")

    src = tool_path.read_text(encoding="utf-8")

    helper = textwrap.dedent(
        """
        import json, sys, types
        # Read tool source from stdin so we don't have to copy files into the
        # container. Construct a fresh module and let exec() populate it; this
        # mirrors what Open-WebUI's plugin loader (load_tool_module_by_id)
        # does. After exec(), the loader returns ``module.Tools()`` — an
        # instance — and that is what get_tool_specs expects (it iterates the
        # instance's methods so module-level imports like ``Optional`` don't
        # leak into the function set).
        src = sys.stdin.read()
        module = types.ModuleType("sn97_status")
        exec(compile(src, "sn97_status.py", "exec"), module.__dict__)
        if not hasattr(module, "Tools"):
            print("ERROR: no Tools class in module", file=sys.stderr)
            sys.exit(2)
        instance = module.Tools()

        from open_webui.utils.tools import get_tool_specs
        specs = get_tool_specs(instance)
        json.dump(specs, sys.stdout)
        """
    ).strip()

    cmd = ["docker", "exec", "-i", CONTAINER, "python", "-c", helper]
    proc = subprocess.run(cmd, input=src, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        fail(
            "spec generation failed inside container:\n"
            f"  cmd: {shlex.join(cmd)}\n"
            f"  stdout: {proc.stdout[:1000]}\n"
            f"  stderr: {proc.stderr[:1000]}"
        )
    try:
        specs = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        fail(f"could not parse spec output: {exc}\n  raw: {proc.stdout[:500]}")
    if not isinstance(specs, list) or not specs:
        fail(f"spec generator returned empty list (raw: {proc.stdout[:500]})")
    return specs


# ---------------------------------------------------------------------------
# Step 2: upsert the tool row
# ---------------------------------------------------------------------------
def _upsert_tool(conn: sqlite3.Connection, *, user_id: str, content: str, specs: list[dict], dry: bool) -> None:
    meta = {
        "description": "Live SN97 subnet data: king, leaderboard, miner info, eval round status.",
        "manifest": {
            "title": "SN97 Subnet Status",
            "author": "distil",
            "author_url": "https://arbos.life",
            "version": "1.0.0",
        },
    }
    existing = conn.execute("SELECT id FROM tool WHERE id=?", (TOOL_ID,)).fetchone()
    if existing:
        info(f"updating existing tool '{TOOL_ID}' (specs: {len(specs)} fns)")
        if dry:
            return
        conn.execute(
            "UPDATE tool SET name=?, content=?, specs=?, meta=?, updated_at=? WHERE id=?",
            ("SN97 Subnet Status", content, json.dumps(specs), json.dumps(meta), NOW, TOOL_ID),
        )
    else:
        info(f"inserting new tool '{TOOL_ID}' (specs: {len(specs)} fns)")
        if dry:
            return
        conn.execute(
            "INSERT INTO tool(id, user_id, name, content, specs, meta, created_at, updated_at, valves) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                TOOL_ID,
                user_id,
                "SN97 Subnet Status",
                content,
                json.dumps(specs),
                json.dumps(meta),
                NOW,
                NOW,
                None,
            ),
        )


# ---------------------------------------------------------------------------
# Step 3: update the sn97-king model row for native FC + tool attach
# ---------------------------------------------------------------------------
def _update_model(conn: sqlite3.Connection, *, dry: bool) -> None:
    row = conn.execute("SELECT meta, params FROM model WHERE id=?", (MODEL_ID,)).fetchone()
    if not row:
        fail(f"sn97-king row missing in model table — was the chat configured before this script?")
    meta = json.loads(row[0]) if row[0] else {}
    params = json.loads(row[1]) if row[1] else {}

    # 3a. Native function calling. Without this Open-WebUI uses prompt-injection
    # mode, which is unreliable on small models.
    params["function_calling"] = "native"

    # 3b. Tool-use system prompt (idempotent via marker). Always overwrite
    # so prompt edits in this file ship; the marker is just an audit hint.
    existing_sys = params.get("system") or ""
    if SYSTEM_PROMPT_MARKER not in existing_sys:
        info(f"installing system prompt {SYSTEM_PROMPT_MARKER} on sn97-king")
    else:
        info(f"refreshing system prompt {SYSTEM_PROMPT_MARKER} on sn97-king")
    params["system"] = SYSTEM_PROMPT

    # 3c. Anti-devolution sampling defaults.
    #
    # Pre-2026-05-05 the model row carried ``max_tokens=24576`` and no
    # repetition penalty. Combined with weak distilled students, this
    # was a recipe for 122k-character loop-the-prompt responses (chat
    # logs e14c2b84 "who are you?" devolved into 124k chars repeating
    # the system prompt; 90ef690a "jelly beans" looped a fake tool-call
    # pattern 50+ times). Direct vLLM probe at the time confirmed the
    # model loops even at max_tokens=80 with default sampling.
    #
    # New defaults (always overwrite so a manual ``max_tokens=…``
    # override on the row gets reset back to safe — pre-fix the row had
    # 24576 from the previous configurator):
    #   * ``max_tokens = 2048``    — covers a multi-paragraph answer
    #     with a tool call or two; far below the previous 24K cap so
    #     the worst-case loop bottoms out fast. Most well-formed
    #     answers fit in <800 tokens; the extra 1248 is for tool
    #     conversations that spill over multiple turns.
    #   * ``temperature = 0.6``    — slightly less wide than 0.7 to
    #     reduce drift on weak students.
    #   * ``top_p = 0.9``          — unchanged, well-tested floor.
    #   * ``frequency_penalty = 0.6`` — pushes back hard on token-level
    #     loops (jelly-bean pattern). 1.0 broke coherence in live
    #     vLLM probes (responses devolved to nonsense word-salad);
    #     0.6 is the sweet spot.
    #   * ``presence_penalty = 0.4``  — discourages re-introducing the
    #     same concept after it's been mentioned (the longer
    #     "system-prompt regurgitation" loops).
    #   * ``repetition_penalty = 1.07`` — vLLM-specific multiplicative
    #     knob, slightly above the conservative 1.05 floor based on
    #     live probes showing the model still loops at 1.05.
    #   * ``stop = ["\\n\\n\\n", ...]`` — emergency brake for the
    #     multi-newline list-junk pattern. Natural prose never has 3+
    #     consecutive blank lines, so stopping there is safe even for
    #     legitimate long answers. Also adds ``"</s>"`` and similar
    #     end-of-text stragglers that some kings emit literally.
    #
    # If a future king is genuinely strong and these defaults are
    # cramping it, raise them per-king in a future config patch — the
    # right place is here, not in users' chat-by-chat overrides.
    params["max_tokens"] = 2048
    params["temperature"] = 0.6
    params["top_p"] = 0.9
    params["frequency_penalty"] = 0.6
    params["presence_penalty"] = 0.4
    params["repetition_penalty"] = 1.07
    params["stop"] = ["\n\n\n\n", "</s>", "<|im_end|>"]

    # 3c.1 Open-WebUI passes ``params.chat_template_kwargs`` through to
    # vLLM — we use it to opt the request into the Kimi K2.6 template's
    # "thinking" branch (template auto-prepends ``<think>`` so the
    # model is structurally inside thinking). The custom
    # ``distil_kimi`` reasoning parser splits on ``</think>`` if the
    # model emits it (Thinking pane shows) and falls back to all-content
    # if it doesn't (answer is at least visible).
    params["chat_template_kwargs"] = {"thinking": True}

    # 3d. Attach the SN97 status toolkit (always-on) + keep any existing.
    tool_ids = list(meta.get("toolIds") or [])
    if TOOL_ID not in tool_ids:
        tool_ids.insert(0, TOOL_ID)
    meta["toolIds"] = tool_ids

    # 3e. Capabilities + builtinTools.
    #
    # 2026-05-05: ``code_interpreter`` was flipped off across the board
    # because Open-WebUI's middleware unconditionally appends the
    # ``CODE_INTERPRETER_PYODIDE_PROMPT`` from
    # ``open_webui/config.py`` (≈10 lines of "Pyodide Environment / Do
    # not install packages / /mnt/uploads/...") to the user message
    # right before every turn whenever ``builtinTools.code_interpreter
    # = True`` AND ``params.function_calling = "native"``. The current
    # weak king (``bodenmaurice/distil-new-v16``) regurgitates that
    # injected text verbatim instead of answering the question (chat
    # 297a48c3 returned 2.4k chars of Pyodide instructions to "Hi").
    # Disabling it on the model row stops the injection. The
    # ``execute_code`` builtin tool is also gone, but the king couldn't
    # use it productively anyway — every observed "code execution"
    # turn was a fake ``print(...)`` blob in plain text content.
    # Re-enable per-king when a tool-fluent king lands.
    caps = meta.get("capabilities") or {}
    caps.update(
        {
            "function_calling": True,
            "file_context": True,
            "vision": False,
            "file_upload": True,
            "web_search": True,
            "image_generation": False,
            "code_interpreter": False,
            "citations": True,
            "status_updates": True,
            "builtin_tools": True,
        }
    )
    meta["capabilities"] = caps

    builtin = meta.get("builtinTools") or {}
    builtin.update(
        {
            "time": True,
            "memory": False,
            "chats": False,
            "notes": False,
            "knowledge": False,
            "channels": False,
            "web_search": True,
            "image_generation": False,
            "code_interpreter": False,
        }
    )
    meta["builtinTools"] = builtin

    info(
        f"updating sn97-king: function_calling=native, toolIds={tool_ids}, "
        f"max_tokens={params['max_tokens']}, web_search=on, code_interpreter=off"
    )
    if dry:
        return
    conn.execute(
        "UPDATE model SET meta=?, params=?, updated_at=? WHERE id=?",
        (json.dumps(meta), json.dumps(params), NOW, MODEL_ID),
    )


# ---------------------------------------------------------------------------
# Step 4: patch the global Open-WebUI config blob
# ---------------------------------------------------------------------------
def _patch_global_config(conn: sqlite3.Connection, *, dry: bool) -> None:
    row = conn.execute("SELECT id, data FROM config LIMIT 1").fetchone()
    if not row:
        fail("config row missing — Open-WebUI hasn't been initialised")
    cfg = json.loads(row[1]) if row[1] else {}

    # 4a. Web search via DuckDuckGo (no API key needed).
    rag = cfg.setdefault("rag", {})
    web = rag.setdefault("web", {})
    search = web.setdefault("search", {})
    search.update(
        {
            "enable": True,
            "engine": "duckduckgo",
            "result_count": 3,
            "concurrent_requests": 5,
            # Open-WebUI defaults to scraping with playwright if installed;
            # not available in our minimal image, so fall back to plain HTTP.
            "duckduckgo_backend": "html",
            "trust_env": True,
            "loader_engine": "safe_web",
        }
    )
    web["loader"] = web.get("loader") or {"engine": "safe_web"}

    # 4b. Code interpreter — globally disabled (see step 3e for why; the
    # OWUI middleware keeps injecting Pyodide context even when the
    # per-model row turns it off, because some code paths only check the
    # global flag). Setting ``enable=False`` here too prevents the
    # injection regardless of the per-model setting.
    code = cfg.setdefault("code", {})
    interpreter = code.setdefault("interpreter", {})
    interpreter.update({"enable": False, "engine": "pyodide"})
    execution = code.setdefault("execution", {})
    execution.update({"enable": False, "engine": "pyodide"})

    # 4c. Feature gates. Code interpreter is OFF (see 4b); web search
    # stays ON (works via the search_web/fetch_url function tools without
    # injecting any prompt text).
    features = cfg.setdefault("features", {})
    features["enable_code_interpreter"] = False
    features["enable_web_search"] = True

    # 4d. Kill the "Arena Model" experiment. Open-WebUI's PersistentConfig
    # routes ENABLE_EVALUATION_ARENA_MODELS to ``evaluation.arena.enable``
    # (NOT ``evaluation.enable_arena_models`` — that key is silently ignored)
    # and EVALUATION_ARENA_MODELS to ``evaluation.arena.models``. We were
    # writing to the wrong keys before, which is why the picker still showed
    # an "Arena Model" entry.
    evaluation = cfg.setdefault("evaluation", {})
    arena = evaluation.setdefault("arena", {})
    arena["enable"] = False
    arena["models"] = []

    # 4e. ui.default_models is the comma-joined fallback for new chats. Force
    # it to just sn97-king so users land on the right model.
    ui = cfg.setdefault("ui", {})
    existing_default = ui.get("default_models") or ""
    if "arena" in existing_default.lower() or existing_default.strip() != MODEL_ID:
        ui["default_models"] = MODEL_ID

    # 4f. Code interpreter persistent path: PersistentConfig stores under
    # ``code_interpreter.enable`` / ``code_interpreter.engine`` (singular,
    # no second segment) — not the ``code.interpreter.*`` path. Set both
    # so the new path wins on rehydration; the old path is harmless but
    # vestigial.
    ci = cfg.setdefault("code_interpreter", {})
    ci["enable"] = False
    ci["engine"] = "pyodide"

    info(
        "global config: web_search=duckduckgo (html), code_interpreter=OFF, "
        f"arena.enable=False, default_models={ui['default_models']}, "
        "image_generation left disabled"
    )
    if dry:
        return
    # NOTE: config.updated_at is a SQLAlchemy DATETIME — store as ISO string.
    conn.execute(
        "UPDATE config SET data=?, updated_at=? WHERE id=?",
        (json.dumps(cfg), NOW_ISO, row[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not DB_PATH.is_file():
        fail(f"webui DB not found at {DB_PATH}")
    if not TOOLS_DIR.is_dir():
        fail(f"tools dir not found at {TOOLS_DIR}")

    tool_src_path = TOOLS_DIR / "sn97_status.py"
    info(f"reading tool source from {tool_src_path}")
    tool_content = tool_src_path.read_text(encoding="utf-8")

    info("generating OpenAI specs via Open-WebUI helper inside the container")
    specs = _generate_specs(tool_src_path)
    info(f"got {len(specs)} function specs: " + ", ".join(s["name"] for s in specs))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        admin = conn.execute("SELECT id FROM user WHERE email=? AND role='admin' LIMIT 1", (ADMIN_EMAIL,)).fetchone()
        if not admin:
            fail(f"no admin user with email '{ADMIN_EMAIL}' — initialise Open-WebUI first")
        admin_id = admin["id"]
        info(f"using admin user id {admin_id}")

        _upsert_tool(conn, user_id=admin_id, content=tool_content, specs=specs, dry=args.dry_run)
        _update_model(conn, dry=args.dry_run)
        _patch_global_config(conn, dry=args.dry_run)

        if args.dry_run:
            info("DRY RUN — no DB changes committed")
            conn.rollback()
        else:
            conn.commit()
            info("DB changes committed")
    finally:
        conn.close()

    info(
        "done. Restart the container to pick up the new global config:\n"
        "  docker restart open-webui"
    )


if __name__ == "__main__":
    main()
