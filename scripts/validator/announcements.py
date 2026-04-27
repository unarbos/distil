"""
Announcement queue and Discord posting for king changes.
"""
import json
import logging
import time

from eval.state import ValidatorState
from scripts.validator.config import DISTIL_ROLE_ID, PAIRED_TEST_ALPHA, EVAL_PROMPTS_H2H
from scripts.validator.composite import (
    ARENA_V3_AXIS_WEIGHTS,
    BENCH_AXIS_WEIGHTS,
    COMPOSITE_SHADOW_VERSION,
)

logger = logging.getLogger("distillation.remote_validator")


# Structural / relative axes are always part of the composite dethrone gate
# regardless of which bench axes are weighted in the current schema. Listing
# them keeps the public Discord text honest (we *do* score chat behaviour,
# KL, judge probes, etc) without implying they're optional.
_STRUCTURAL_AXES = (
    "on_policy_rkl",
    "kl",
    "capability",
    "judge_probe",
    "chat_turns_probe",
    "length",
    "degeneracy",
    "reasoning_density",
)


def _active_axis_summary() -> tuple[int, str]:
    """Return ``(count, comma-separated names)`` of axes with non-zero composite
    weight, computed live from ``BENCH_AXIS_WEIGHTS`` + ``ARENA_V3_AXIS_WEIGHTS``
    plus the always-on structural axes.

    Used by the Discord announcement so the axis list can never drift after a
    schema bump (v27 → v28 muted six bench axes and the literal text was left
    stale, which surfaced in Discord as inaccurate axis attribution).
    """
    active_bench = [k for k, w in BENCH_AXIS_WEIGHTS.items() if w > 0]
    active_arena = [k for k, w in ARENA_V3_AXIS_WEIGHTS.items() if w > 0]
    names = active_bench + active_arena + list(_STRUCTURAL_AXES)
    pretty = ", ".join(n.replace("_bench", "") for n in names)
    return len(names), pretty


def announce_new_king(new_uid, new_model, new_kl, old_uid, old_model, old_kl,
                      state: ValidatorState, paired_prompts=None, total_prompts=None,
                      p_value=None, *, new_composite_worst=None,
                      new_composite_weighted=None, new_limiting_axis=None,
                      old_composite_worst=None, old_composite_weighted=None):
    """Write a pending announcement to state for async Discord posting.

    Composite-worst is the production ranking key (since v27). When the
    caller passes ``new_composite_worst``, the Discord post leads with the
    composite scores and demotes KL to a parenthetical "axis component"
    line so the public framing matches the eval mechanism. KL-only
    framing was the headline through 2026-04-26 and contributed to the
    reasoning-spiral failure mode (see paper/off_policy_cot_collapse.md);
    this signature keeps backward compatibility for callers that don't
    know the composite yet (the headline falls back to KL in that case).
    """
    import os as _os
    single_eval_active = bool(int(_os.environ.get("SINGLE_EVAL_MODE", "0") or 0))

    earnings_line = ""
    try:
        import urllib.request
        resp = urllib.request.urlopen("https://api.arbos.life/api/price", timeout=10)
        price_data = json.loads(resp.read())
        tao_per_day = price_data.get("miners_tao_per_day", 0)
        tao_usd = price_data.get("tao_usd", 0)
        usd_per_day = tao_per_day * tao_usd
        if tao_per_day > 0 and tao_usd > 0:
            earnings_line = (
                f"\n💰 **King earns ~{tao_per_day:.1f} τ/day (${usd_per_day:,.0f}/day)** — "
                f"winner takes all!\n"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch price data for announcement: {e}")

    role_ping = f"<@&{DISTIL_ROLE_ID}>"
    prompt_count = paired_prompts or total_prompts or EVAL_PROMPTS_H2H

    # Wording branch: in single-eval mode the king is selected cross-round from
    # composite_scores (different prompts per challenger), so "paired prompts"
    # is misleading. Show the per-challenger prompt count instead and explain
    # the actual gate. In legacy paired-t-test mode keep the old wording.
    if single_eval_active:
        prompt_line = f"🧪 Scored on {prompt_count} block-seeded prompts"
        n_axes, axis_list = _active_axis_summary()
        gate_explainer = (
            f"Dethronement uses an absolute composite across {n_axes} weighted axes "
            f"({axis_list}). 3-stage gate: clear win on `worst` (>3% margin) → take "
            f"crown; clear regression on `worst` (<-3%) → reject; tied region → "
            f"fall back to `weighted` with the same 3% margin. KL shown above is "
            f"the global distillation distance, not the ranking key. One eval per "
            f"commitment; no re-evals. (composite schema v{COMPOSITE_SHADOW_VERSION})"
        )
    else:
        prompt_line = f"🧪 Compared on {prompt_count} paired prompts"
        if total_prompts and paired_prompts and total_prompts != paired_prompts:
            prompt_line = f"🧪 Compared on {paired_prompts}/{total_prompts} paired prompts"
        gate_explainer = (
            f"Dethronement uses one-sided paired t-test (p<{PAIRED_TEST_ALPHA})."
        )
    p_line = f" (p={p_value:.4f})" if isinstance(p_value, (int, float)) else ""

    # Build the headline. Composite-worst is the production ranking key
    # under v28 (see ``select_king_by_composite``); KL is one of 17 axes,
    # never the gate. Fall back to the KL-only line only when the caller
    # didn't pass composite payloads (legacy callers, dev mode).
    if new_composite_worst is not None:
        worst_line = f"📊 **Composite worst: {new_composite_worst:.3f}**"
        if new_limiting_axis:
            axis_pretty = new_limiting_axis.replace("_bench", "").replace("_", " ")
            worst_line += f" (limiting axis: {axis_pretty})"
        if old_composite_worst is not None:
            worst_line += f" — previous king: {old_composite_worst:.3f}"
        weighted_part = ""
        if new_composite_weighted is not None:
            weighted_part = f"\n📐 Weighted mean: {new_composite_weighted:.3f}"
            if old_composite_weighted is not None:
                weighted_part += f" (was {old_composite_weighted:.3f})"
        kl_part = f"\n└ KL component: {new_kl:.6f} (1 of 17 axes — not the ranking key)"
        headline = worst_line + weighted_part + kl_part
    else:
        # Backward-compat path. Lead with KL but explicitly note it's
        # one of N axes so the framing isn't misleading even here.
        headline = (
            f"📊 **KL: {new_kl:.6f}** (one of 17 axes; previous king's KL: {old_kl:.6f})"
        )

    announcement = {
        "type": "new_king",
        "timestamp": time.time(),
        "posted": False,
        "message": (
            f"{role_ping}\n"
            f"## 🏆 New King of Distil SN97!\n\n"
            f"**UID {new_uid}** has dethroned **UID {old_uid}**\n\n"
            f"{headline}\n"
            f"{prompt_line}{p_line}\n"
            f"🤗 Model: [{new_model}](<https://huggingface.co/{new_model}>)\n"
            f"👑 Previous king: [{old_model}](<https://huggingface.co/{old_model}>)\n"
            f"{earnings_line}\n"
            f"{gate_explainer} "
            f"Check the [mining guide](<https://github.com/unarbos/distil#mining-guide>) to get started.\n\n"
            f"📈 [Live Dashboard](<https://distil.arbos.life>)"
        ),
        "data": {
            "new_uid": new_uid, "new_model": new_model, "new_kl": new_kl,
            "old_uid": old_uid, "old_model": old_model, "old_kl": old_kl,
            "new_composite_worst": new_composite_worst,
            "new_composite_weighted": new_composite_weighted,
            "new_limiting_axis": new_limiting_axis,
            "old_composite_worst": old_composite_worst,
            "old_composite_weighted": old_composite_weighted,
            "single_eval_mode": single_eval_active,
        },
    }
    state.save_announcement(announcement)
    logger.info(f"Announcement written: UID {new_uid} dethroned UID {old_uid}")
