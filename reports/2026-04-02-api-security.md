# API Security Hardening Report

**Date:** 2026-04-02  
**Branch:** `improvements/validator-fixes-v2`  
**File:** `api/server.py`

## Changes Made

### 1. Command Injection Fix (CRITICAL)
**Before:** `/api/chat` built shell commands by embedding JSON payloads directly into f-strings:
```python
payload_str = json.dumps(payload).replace("'", "'\\''")
cmd = f"curl -s -X POST ... -d '{payload_str}'"
```
This was vulnerable to shell injection via crafted message content.

**After:** Payload is written to a temp file on the pod via heredoc, then curl reads from `@/tmp/_chat_payload.json`. Applied to both `_sync_chat` and `_stream_chat`.

### 2. Rate Limiting
Added `RateLimiter` class (in-memory, sliding window):
- **General endpoints:** 60 req/min per IP (via `RateLimitMiddleware`)
- **`/api/chat`:** 10 req/min per IP (checked in handler)
- Docs endpoints (`/docs`, `/redoc`, `/openapi.json`) are exempt
- Returns HTTP 429 `{"error": "rate limit exceeded"}`

### 3. CORS Tightening
Changed `allow_origins=["*"]` to:
```python
allow_origins=["https://distil.arbos.life", "http://localhost:3000", "http://localhost:5173"]
```

### 4. Chat Input Validation
- `max_tokens` clamped to 1–4096 (default 2048)
- `messages` array max length: 50
- Per-message content max: 10,000 chars
- `temperature` validated 0–2 (default 0.7)
- `top_p` validated 0–1 (default 0.9)
- Invalid values silently replaced with defaults

### 5. Log Sanitization Enhancement
Extended `_SECRET_RE` to also match:
- SSH public keys (`ssh-rsa`, `ssh-ed25519`, etc.)
- Base64 credential blobs (`AAAA...`)

Extended `_SENSITIVE_KW` to include:
- `"PRIVATE KEY"`, `"ssh-rsa"`, `"ssh-ed25519"`, `"credentials"`

### 6. `/api/king-history` Endpoint
New `GET /api/king-history` — reads `state/h2h_history.json`, filters for `king_changed=true` entries, returns:
```json
[{"block": 123, "timestamp": "...", "king_uid": 172, "king_model": "...", "dethroned_uid": 34, "margin": 0.012}]
```
Margin is calculated as relative KL improvement: `(prev_kl - king_kl) / prev_kl`.

### 7. Import Cleanup
Consolidated scattered imports (`Request`, `StreamingResponse`, `RedirectResponse`, `BaseHTTPMiddleware`) to file top. Removed unused `tempfile` import.

## Files Changed
- `api/server.py` — +120 lines, -14 lines

## Verification
- File compiles clean (`py_compile`)
- All curl commands now use `-d @file` pattern (no inline payload)
- Rate limiter instances are module-level singletons
