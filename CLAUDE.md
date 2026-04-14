# CLAUDE.md

This workspace backs the Discord channel `Distil Channel`.

## Runtime
- Current runtime is Claude Code only.
- Do not describe OpenClaw or Codex as the active runtime for this workspace.
- Historical transcripts may mention OpenClaw or Codex. Treat those as legacy references only.

## Continuity Rules
- Do not claim this workspace is a cold start if `SESSION_MEMORY.md` or any identity/user/memory files exist.
- Before asking the user to restate context, use `SESSION_MEMORY.md` plus the local workspace files.
- If the user says `continue`, `resume`, or asks what is in progress, continue from the imported context in `SESSION_MEMORY.md`.
- If imported context conflicts with current code, files, or live data, trust the current code/files/data and say the imported notes may be stale.

## Files To Treat As Available Context
- `SESSION_MEMORY.md`
- `AGENTS.md`
- `SOUL.md`
- `TOOLS.md`
- `IDENTITY.md`
- `USER.md`
- `HEARTBEAT.md`

## Imported Context Summary
- scripts/validator/eval_orchestrator.py:550:        f"cd /home && python3 -u pod_eval.py " | scripts/validator/eval_orchestrator.py:639:                    pod.exec("pkill -9 -f pod_eval.py; echo killed", timeout=30) | eval/pod.py:63:        pm.upload("scripts/pod_eval_vllm.py", "/home/pod_eval.py") | eval/pod.py:64:        result = pm.exec("python3 /home/pod_eval.py ...") | scripts/pod_eval_vllm.py:27:    python3 pod_eval_vllm.py \\ | scripts/validator/eval_orchestrator.py:550:        f"cd /home && python3 -u pod_eval.py " | scripts/validator/eval_orchestrator.py:639:                    pod.exec("pkill -9 -f pod_eval.py; echo killed", timeout=30) | scripts/remote_validator.py:116:    eval_script = "scripts/pod_eval_vllm.py" | scripts/remote_validator.py:117:    eval_script_remote = "/home/pod_eval.py" | scripts/verify_round.py:363:    info(f"  2. Run eval: python scripts/pod_eval_vllm...
- frontend/src/lib/api.ts:47:  commitments: Record< | frontend/src/lib/api.ts:49:    { block: number; model?: string; revision?: string; raw?: string } | frontend/src/lib/api.ts:90:  revision: string; | frontend/src/lib/api.ts:95:  commitBlock: number; | frontend/src/lib/api.ts:161:  return safeFetch(`${API_BASE}/api/commitments`); | frontend/src/lib/api.ts:274:  commitments: CommitmentsResponse | null, | frontend/src/lib/api.ts:278:  if (!metagraph || !commitments) return []; | frontend/src/lib/api.ts:280:  // Map hotkey → commitment | frontend/src/lib/api.ts:281:  const hotkeyCom = commitments.commitments; | frontend/src/lib/api.ts:320:    // Check DQ by hotkey:block (per-commit), hotkey (legacy), or UID (legacy) | frontend/src/lib/api.ts:321:    const commitBlock = com.block; | frontend/src/lib/api.ts:322:    const dqReason = (commitBlock != null ? scores?.disqualified?.[`${neuron.ho...
- """Health check and root redirect endpoints.""" | import os | import time as _time | from fastapi import APIRouter | from fastapi.responses import RedirectResponse | from config import NETUID, STATE_DIR | from helpers.sanitize import _safe_json_load | router = APIRouter() | @router.get("/", include_in_schema=False) | def root(): | """Redirect to interactive API docs.""" | return RedirectResponse(url="/docs") | @router.get("/api/health", tags=["Overview"], summary="Service health and quick status", | description="""One-stop health check that returns the current state of the validator and subnet. | Response includes: | - `status`: `ok` if the API is running | - `king_uid` / `king_kl`: Current king and their KL score (lower = better) | - `n_scored` / `n_disqualified`: Number of active vs disqualified miners | - `last_eval_block` / `last_eval_age_min`: When the last eval happened | - `eva...
- """Evaluation endpoints: H2H, leaderboard, eval progress, history, benchmarks, announcements.""" | import json | import os | import time | from fastapi import APIRouter | from fastapi.responses import JSONResponse | from config import STATE_DIR | from helpers.cache import _get_stale | from helpers.sanitize import _sanitize_floats, _safe_json_load | router = APIRouter() | @router.get("/api/leaderboard", tags=["Evaluation"], summary="Top-4 leaderboard", | description="Returns the top-4 leaderboard - current king and contenders. Dethronement uses paired t-test (p < 0.05).") | def get_leaderboard(): | top4 = _safe_json_load(os.path.join(STATE_DIR, "top4_leaderboard.json"), {}) or {} | scores = _safe_json_load(os.path.join(STATE_DIR, "scores.json"), {}) | h2h_latest = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {}) | uid_map = _safe_json_load(os.path.join(STATE_DIR, "uid_ho...
- # Reverse so newest first, then paginate | data_rev = list(reversed(data)) | start = (page - 1) * limit | end = start + limit | page_data = data_rev[start:end] | return JSONResponse( | content=_sanitize_floats({"rounds": page_data, "total": total, "page": page, "limit": limit, "has_more": end < total}), | headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"}, | ) | except Exception: | pass | return JSONResponse( | content={"rounds": [], "total": 0, "page": 1, "limit": limit, "has_more": False}, | headers={"Cache-Control": "public, max-age=10"}, | ) | @router.get("/api/king-history", tags=["Evaluation"], summary="King dethronement history", | description="Returns the chain of king changes (dethronements). Each entry shows the block, new king, and the dethroned UID with margin of victory.") | def get_king_history(): | """Extract all king changes from h2h_history.jso...
