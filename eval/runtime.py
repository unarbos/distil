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
VLLM_CONCURRENCY = VALIDATOR["vllmConcurrency"]
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

# Chat pod coordinates are intentionally env-only. Lium pods can be
# reprovisioned, and a stale hardcoded host makes healthcheck/API repair loops
# hammer the wrong machine. When unset, chat endpoints report unavailable
# instead of trying old infrastructure.
CHAT_POD_HOST = os.environ.get("CHAT_POD_HOST") or os.environ.get("DISTIL_CHAT_POD_HOST", "")
CHAT_POD_SSH_PORT = int(os.environ.get("CHAT_POD_SSH_PORT") or os.environ.get("DISTIL_CHAT_POD_SSH_PORT", "22"))
CHAT_POD_APP_PORT = CHAT["appPort"]
CHAT_POD_SSH_KEY = os.environ.get("CHAT_POD_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

STATE_DIR = os.environ.get("DISTIL_STATE_DIR", str(REPO_ROOT / "state"))
DISK_CACHE_DIR = os.path.join(STATE_DIR, "api_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

TMC_KEY = os.environ.get("TMC_API_KEY", "")
TMC_HEADERS = {"Authorization": TMC_KEY} if TMC_KEY else {}
