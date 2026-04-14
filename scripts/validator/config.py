"""
Constants and configuration for the remote validator.
"""
import os

# ── Model / Network ───────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
NETUID = 97
MAX_KL_THRESHOLD = 2.0
MAX_NEW_TOKENS = 8192
MAX_PROMPT_TOKENS = 1024

# ── Eval Parameters ───────────────────────────────────────────────────────
EVAL_PROMPTS_FULL = 60    # Full eval: many models, need speed
EVAL_PROMPTS_H2H = 300    # Head-to-head prompts per round
VLLM_CONCURRENCY = 8       # Parallel requests to vLLM for teacher generation
EPSILON = 0.01             # Legacy fallback if per-prompt data unavailable
PAIRED_TEST_ALPHA = 0.03   # Significance level for paired t-test dethronement
STALE_H2H_EPOCHS = 50      # Re-test if last H2H was >N epochs ago
TOP_N_ALWAYS_INCLUDE = 5   # king + top 4 contenders always in eval

# ── Reference Model ──────────────────────────────────────────────────────
REFERENCE_MODEL = "Qwen/Qwen3.5-4B"  # Undistilled 4B base — shows how much distillation helps
REFERENCE_UID = -1  # Synthetic UID, never written to scores/weights

# ── Copy Detection ───────────────────────────────────────────────────────
ACTIVATION_COPY_THRESHOLD = 0.9999  # Cosine similarity above this = functional copy

# ── Discord ──────────────────────────────────────────────────────────────
DISTIL_ROLE_ID = "1482026585358991571"

# ── Chat Pod ─────────────────────────────────────────────────────────────
CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST", "91.224.44.207")
CHAT_POD_SSH_PORT = os.environ.get("CHAT_POD_SSH_PORT", "40070")
CHAT_POD_APP_PORT = 8100
