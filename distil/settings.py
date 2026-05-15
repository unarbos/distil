"""Typed runtime configuration.

Replaces the ~70 ``os.environ.get(...)`` calls scattered across the legacy
``scripts/validator/composite.py`` + ``scripts/eval_policy.py`` with one
Pydantic Settings class. Defaults match v31.2 / v32.6 production.

Environment variables override fields by name. A ``~/.secrets/distil.env``
file (the legacy convention) is loaded if present. All paths in
``Settings`` are absolute :class:`pathlib.Path` objects after validation.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_env_files() -> tuple[str, ...]:
    candidates = [
        Path.home() / ".secrets" / "distil.env",
        REPO_ROOT / ".env",
    ]
    return tuple(str(p) for p in candidates if p.is_file())


class Settings(BaseSettings):
    """All runtime knobs the validator + API + pod runner care about."""

    model_config = SettingsConfigDict(
        env_file=_default_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Paths ───────────────────────────────────────────────────────
    repo_root: Path = Field(default=REPO_ROOT)
    state_dir: Path = Field(default=REPO_ROOT / "state")
    cache_dir: Path = Field(default=REPO_ROOT / "state" / "_cache")
    log_dir: Path = Field(default=REPO_ROOT / "state" / "_logs")

    # ── Bittensor chain ─────────────────────────────────────────────
    netuid: int = 97
    network: str = "finney"
    wallet_name: str = "default"
    hotkey_name: str = "default"
    chain_endpoint: str = ""

    # ── Teacher / reference ─────────────────────────────────────────
    teacher_repo: str = "moonshotai/Kimi-K2.6"
    teacher_revision: str = ""
    reference_repo: str = "Qwen/Qwen3.5-4B"
    teacher_hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # ── vLLM (improvement #4: 32K + chunked prefill, hard-coded) ────
    vllm_max_model_len: int = 32768
    vllm_enable_chunked_prefill: bool = True
    vllm_gpu_memory_utilization: float = 0.85
    vllm_dtype: str = "bfloat16"
    vllm_max_logprobs: int = 128
    vllm_tensor_parallel_size: int = 1
    vllm_swap_space_gb: int = 0

    # ── Eval policy ─────────────────────────────────────────────────
    eval_n_prompts: int = 256
    eval_max_new_tokens: int = 512
    eval_per_axis_n: int = 16
    teacher_top_k: int = 20
    student_prompt_logprobs: int = 20
    eval_round_max_minutes: int = 60
    eval_per_student_max_minutes: int = 30
    teacher_phase_max_minutes: int = 25
    pod_disk_floor_gb: int = 25
    pod_dl_max_seconds: int = 1500

    # ── Composite aggregation ──────────────────────────────────────
    composite_final_bottom_weight: float = 0.75
    worst_3_mean_k: int = 3
    composite_dethrone_min_axes: int = 5
    composite_dethrone_margin: float = 0.05
    composite_schema_version: int = 32
    teacher_sanity_floor: float = 0.70

    # ── Axis weights (live in v31.2 / v32.6) ────────────────────────
    weight_on_policy_rkl: float = 0.39
    weight_top_k_overlap: float = 0.09
    weight_kl: float = 0.05
    weight_capability: float = 0.05
    weight_length: float = 0.05
    weight_degeneracy: float = 0.05
    weight_judge_probe: float = 0.20
    weight_long_form_judge: float = 0.20
    weight_long_gen_coherence: float = 0.25
    weight_chat_turns_probe: float = 0.14
    weight_reasoning_density: float = 0.05
    weight_calibration_bench: float = 0.06

    weight_v31_math_gsm_symbolic: float = 0.06
    weight_v31_math_competition: float = 0.05
    weight_v31_math_robustness: float = 0.03
    weight_v31_code_humaneval_plus: float = 0.08
    weight_v31_reasoning_logic_grid: float = 0.05
    weight_v31_reasoning_dyval_arith: float = 0.04
    weight_v31_long_context_ruler: float = 0.05
    weight_v31_knowledge_multi_hop_kg: float = 0.04
    weight_v31_ifeval_verifiable: float = 0.04
    weight_v31_truthfulness_calibration: float = 0.03
    weight_v31_consistency_paraphrase: float = 0.03

    # ── Baseline-relative penalty ──────────────────────────────────
    baseline_penalty_enabled: bool = True
    baseline_penalty_alpha: float = 1.5

    # ── Activation fingerprinting ──────────────────────────────────
    activation_fp_dim: int = 512
    activation_fp_threshold: float = 0.985

    # ── HF / dataset ────────────────────────────────────────────────
    eval_dataset: str = "karpathy/climbmix-400b-shuffle"
    eval_dataset_split: str = "train"
    hf_cache_dir: Path = Field(default=Path.home() / ".cache" / "huggingface")
    hf_dl_token: str = Field(default="", validation_alias="HF_TOKEN")

    # ── API ─────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 3710
    api_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "https://distil.arbos.life",
            "https://api.arbos.life",
            "https://api-v2.arbos.life",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    )
    api_rate_limit_per_minute: int = 120
    api_chat_rate_limit_per_minute: int = 30

    # ── Chat surface ────────────────────────────────────────────────
    chat_pod_url: str = "http://127.0.0.1:8100"
    chat_model_name: str = "sn97-king"
    chat_max_turns: int = 6
    chat_timeout_s: int = 90
    chat_audit_log: Path = Field(default=REPO_ROOT / "state" / "chat_audit.jsonl")

    # ── External providers ─────────────────────────────────────────
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    web_search_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")

    # ── Multi-king / weight setting ────────────────────────────────
    multi_king_payout_enabled: bool = True
    recent_kings_max: int = 4
    set_weights_on_dq: bool = True

    # ── Lium / pod ─────────────────────────────────────────────────
    lium_api_key: str = Field(default="", validation_alias="LIUM_API_KEY")
    lium_default_pod_size: str = "B200x1"
    lium_chat_pod_size: str = "B200x1"
    lium_pod_name: str = Field(
        default="",
        validation_alias="DISTIL_LIUM_POD_NAME",
        description="Persistent pod name. When set, `attach_pod()` reuses this "
        "pod every round instead of provisioning a fresh one (prod default).",
    )
    eval_n_gpus: int = Field(
        default=8,
        validation_alias="DISTIL_N_GPUS",
        description="GPUs to fan student shards across; 1 = serial.",
    )
    eval_persistent_pod: bool = Field(
        default=True,
        validation_alias="DISTIL_PERSISTENT_POD",
        description="If true, the validator reuses `lium_pod_name`; otherwise "
        "it provisions a fresh ephemeral pod per round.",
    )

    @field_validator(
        "repo_root",
        "state_dir",
        "cache_dir",
        "log_dir",
        "hf_cache_dir",
        "chat_audit_log",
        mode="before",
    )
    @classmethod
    def _expand(cls, v):
        if v is None:
            return v
        return Path(str(v)).expanduser().resolve()

    def axis_weights(self) -> dict[str, float]:
        """Return the v31.2 + v32.6 axis-name → weight map."""
        return {
            "on_policy_rkl": self.weight_on_policy_rkl,
            "top_k_overlap": self.weight_top_k_overlap,
            "kl": self.weight_kl,
            "capability": self.weight_capability,
            "length": self.weight_length,
            "degeneracy": self.weight_degeneracy,
            "judge_probe": self.weight_judge_probe,
            "long_form_judge": self.weight_long_form_judge,
            "long_gen_coherence": self.weight_long_gen_coherence,
            "chat_turns_probe": self.weight_chat_turns_probe,
            "reasoning_density": self.weight_reasoning_density,
            "calibration_bench": self.weight_calibration_bench,
            "v31_math_gsm_symbolic": self.weight_v31_math_gsm_symbolic,
            "v31_math_competition": self.weight_v31_math_competition,
            "v31_math_robustness": self.weight_v31_math_robustness,
            "v31_code_humaneval_plus": self.weight_v31_code_humaneval_plus,
            "v31_reasoning_logic_grid": self.weight_v31_reasoning_logic_grid,
            "v31_reasoning_dyval_arith": self.weight_v31_reasoning_dyval_arith,
            "v31_long_context_ruler": self.weight_v31_long_context_ruler,
            "v31_knowledge_multi_hop_kg": self.weight_v31_knowledge_multi_hop_kg,
            "v31_ifeval_verifiable": self.weight_v31_ifeval_verifiable,
            "v31_truthfulness_calibration": self.weight_v31_truthfulness_calibration,
            "v31_consistency_paraphrase": self.weight_v31_consistency_paraphrase,
        }

    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axis_weights().keys())


settings = Settings()


def reload_settings() -> Settings:
    """For tests / migration scripts."""
    global settings
    settings = Settings()
    return settings


def env_or(name: str, default: str = "") -> str:
    """Compatibility helper: read an env var bypassing the Settings cache."""
    return os.environ.get(name, default)
