import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBNET_CONFIG_PATH = REPO_ROOT / "frontend" / "src" / "lib" / "subnet-config.json"
ENV_PATH = REPO_ROOT / ".env"


def _load_env():
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _load_config():
    return json.loads(SUBNET_CONFIG_PATH.read_text())


_load_env()
CONFIG = _load_config()
TEACHER = CONFIG["teacher"]
VALIDATOR = CONFIG["validator"]
CHAT = CONFIG["chat"]
API = CONFIG["api"]

NETUID = CONFIG["netuid"]
TEACHER_MODEL = TEACHER["model"]
TEACHER_VOCAB_SIZE = TEACHER["vocabSize"]
TEACHER_CONFIG_VOCAB_SIZE = TEACHER["configVocabSize"]
TEACHER_TOTAL_PARAMS = TEACHER["totalParams"]
TEACHER_ACTIVE_PARAMS = TEACHER["activeParams"]
TEACHER_ARCHITECTURE = TEACHER["architecture"]
MAX_STUDENT_PARAMS = TEACHER["maxStudentParams"]

# Architecture allowlist for miner submissions. Consumers (eval/model_checker.py,
# scripts/validator/precheck.py, scripts/chat_pod/chat_server.py) should treat
# this as the single source of truth — never hardcode a model_type or
# architectures string in a second place.
# Format: list of {"model_type": str, "architecture": str} pairs. Matching is
# OR across entries (any pair with model_type AND that architecture in the
# config.json ``architectures`` array = allowed). A bare list of strings is
# accepted as shorthand for architecture-only matching (model_type ignored).
STUDENT_ARCH_ALLOWLIST: list[dict] = list(TEACHER.get("studentArchAllowlist") or [])
# Flattened helper: set of allowed ``architectures`` strings. Precheck's cheap
# path only inspects ``architectures``; the full checker also compares
# ``model_type``.
STUDENT_ARCH_NAMES: set[str] = {
    entry["architecture"] for entry in STUDENT_ARCH_ALLOWLIST if isinstance(entry, dict) and entry.get("architecture")
}
STUDENT_MODEL_TYPES: set[str] = {
    entry["model_type"] for entry in STUDENT_ARCH_ALLOWLIST if isinstance(entry, dict) and entry.get("model_type")
}

MAX_KL_THRESHOLD = VALIDATOR["maxKlThreshold"]
# 2026-04-25 17:00 UTC: env-override added so we can shorten teacher generation
# without redeploying. Subnet config baseline is 8192 but mean teacher length
# is ~700 tokens; the rare 8192-cap hits dominate per-prompt wall time. Reduce
# via TEACHER_MAX_NEW_TOKENS=2048 in distil.env to halve KL-pass wall.
_ENV_MAX_NEW_TOKENS = (
    os.environ.get("TEACHER_MAX_NEW_TOKENS")
    or os.environ.get("EVAL_MAX_NEW_TOKENS")
    or os.environ.get("MAX_NEW_TOKENS")
)
try:
    MAX_NEW_TOKENS = int(_ENV_MAX_NEW_TOKENS) if _ENV_MAX_NEW_TOKENS else int(VALIDATOR["maxNewTokens"])
except (TypeError, ValueError):
    MAX_NEW_TOKENS = int(VALIDATOR["maxNewTokens"])
MAX_PROMPT_TOKENS = VALIDATOR["maxPromptTokens"]
EVAL_PROMPTS_FULL = VALIDATOR["evalPromptsFull"]
EVAL_PROMPTS_H2H = VALIDATOR["evalPromptsH2h"]
_ENV_VLLM_CONCURRENCY = os.environ.get("DISTIL_VLLM_CONCURRENCY")
try:
    VLLM_CONCURRENCY = (
        int(_ENV_VLLM_CONCURRENCY)
        if _ENV_VLLM_CONCURRENCY else int(VALIDATOR["vllmConcurrency"])
    )
except (TypeError, ValueError):
    VLLM_CONCURRENCY = int(VALIDATOR["vllmConcurrency"])
EPSILON = VALIDATOR["epsilon"]
PAIRED_TEST_ALPHA = VALIDATOR["pairedTestAlpha"]
STALE_H2H_EPOCHS = VALIDATOR["staleH2hEpochs"]
TOP_N_ALWAYS_INCLUDE = VALIDATOR["topNAlwaysInclude"]
REFERENCE_MODEL = VALIDATOR["referenceModel"]
REFERENCE_UID = VALIDATOR["referenceUid"]
ACTIVATION_COPY_THRESHOLD = VALIDATOR["activationCopyThreshold"]
DISTIL_ROLE_ID = VALIDATOR["distilRoleId"]

CACHE_TTL = API["cacheTtl"]
TMC_BASE = API["tmcBase"]
PUBLIC_API_URL = API["publicUrl"]
DASHBOARD_URL = API["dashboardUrl"]
ALLOWED_ORIGINS = list(API["allowedOrigins"])

STATE_DIR = os.environ.get("DISTIL_STATE_DIR", str(REPO_ROOT / "state"))
DISK_CACHE_DIR = os.path.join(STATE_DIR, "api_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)


# ── Chat pod coordinates ─────────────────────────────────────────────────────
# Resolution order (first non-empty wins):
#   1. CHAT_POD_HOST / CHAT_POD_SSH_PORT / CHAT_POD_SSH_KEY env vars
#   2. DISTIL_CHAT_POD_* env vars (legacy alias)
#   3. ``state/chat_pod.json`` (single source of truth, updated via
#      ``python -m scripts.validator.chat_pod_admin set ...``)
#
# Why a state file: Lium pods get reprovisioned, the king's serving pod
# gets reaped, and miners ship new checkpoints continuously. Hardcoding the
# host in systemd / .env means every churn triggers a manual edit on the
# validator host. Reading from a JSON state file lets the validator (and
# the chat-tunnel.path systemd watcher) pick up new coordinates without
# reloading services. Pre-2026-04-26 every churn caused chat.arbos.life to
# 502 until ops noticed; the state-file + watcher loop closes that gap.
def _load_chat_pod_state() -> dict:
    state_path = os.path.join(STATE_DIR, "chat_pod.json")
    try:
        import json as _json
        with open(state_path) as f:
            data = _json.load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except (FileNotFoundError, ValueError, OSError):
        return {}


_chat_pod_state = _load_chat_pod_state()


def _chat_pod_value(env_keys: tuple[str, ...], state_key: str, default: str = "") -> str:
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val
    raw = _chat_pod_state.get(state_key)
    if raw is None:
        return default
    return str(raw)


CHAT_POD_HOST = _chat_pod_value(("CHAT_POD_HOST", "DISTIL_CHAT_POD_HOST"), "host", "")
CHAT_POD_SSH_PORT = int(
    _chat_pod_value(("CHAT_POD_SSH_PORT", "DISTIL_CHAT_POD_SSH_PORT"), "ssh_port", "22")
)
CHAT_POD_APP_PORT = int(
    _chat_pod_value(("CHAT_POD_APP_PORT", "DISTIL_CHAT_POD_APP_PORT"), "app_port", str(CHAT["appPort"]))
)
CHAT_POD_SSH_KEY = _chat_pod_value(
    ("CHAT_POD_SSH_KEY",), "ssh_key", os.path.expanduser("~/.ssh/id_ed25519"),
)
# Persisted ``model`` is informational — the validator's side_effects loop
# rewrites chat_pod.json with the actual king on every restart_chat_server,
# but at module-load time we don't know yet which UID is reigning.
CHAT_POD_MODEL = _chat_pod_state.get("model") or ""

TMC_KEY = os.environ.get("TMC_API_KEY", "")
TMC_HEADERS = {"Authorization": TMC_KEY} if TMC_KEY else {}
