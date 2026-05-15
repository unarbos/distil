"""Validator service loop — fetch chain → eval → score → set_weights.

Each iteration:
  1. Fetch metagraph + commitments (with retry).
  2. Evict stale composite scores; select challengers (FIFO + king re-eval).
  3. Acquire a Lium pod, upload code, run the eval.
  4. Process results into composites + state shards.
  5. Resolve king + dethrone gate; build emission weights.
  6. ``set_weights`` (skip on ``--dry-run``).
  7. Sleep until the next round (one round per ~360 blocks ≈ 70 minutes).
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from distil.chain.commitments import fetch_revealed, parse_commitments
from distil.chain.metagraph import fetch_metagraph, get_subtensor
from distil.chain.weights import SetWeightsError, set_weights
from distil.eval.king import build_emission_weights, resolve_king
from distil.eval.pod import acquire_pod, install_runtime, run_eval_on_pod, upload_runtime
from distil.eval.results import process_round
from distil.eval.round import build_round_spec, evict_stale_composites, select_challengers
from distil.settings import settings
from distil.state.files import ValidatorState, log_event

logger = logging.getLogger("distil.eval.service")

ROUND_INTERVAL_S = 4200  # ~70 minutes / one Bittensor super-block

_STOP = False


def _install_signal_handlers() -> None:
    def _on_sig(signum, frame):
        global _STOP
        _STOP = True
        logger.info(f"signal {signum}; stopping after this round")

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)


def _wallet():
    import bittensor as bt

    return bt.wallet(name=settings.wallet_name, hotkey=settings.hotkey_name)


def _round(state: ValidatorState, *, dry_run: bool) -> None:
    sub = get_subtensor()
    mg, block, block_hash = fetch_metagraph(sub, settings.netuid)
    revealed = fetch_revealed(sub, settings.netuid)
    commitments, uid_to_hotkey, _ = parse_commitments(mg, revealed, n_uids=len(mg.hotkeys))

    state.uid_hotkey_map = {str(uid): hk for uid, hk in uid_to_hotkey.items()}
    n_evict = evict_stale_composites(state)
    if n_evict:
        logger.info(f"evicted {n_evict} stale composite rows (schema bump)")

    # Resolve seated king from current composite_scores.
    king_name, _king_reason = resolve_king(state.composite_scores, current_king_model=None)
    king_uid = next(
        (c.uid for c in commitments.values() if c.key == king_name),
        None,
    )
    king_commitment = commitments.get(king_uid) if king_uid is not None else None

    challengers = select_challengers(commitments, state, king_uid=king_uid)
    if not challengers and king_commitment is None:
        logger.info("no challengers and no king; skipping round")
        return

    spec = build_round_spec(
        block=block,
        block_hash=block_hash,
        teacher_repo=settings.teacher_repo,
        reference_repo=settings.reference_repo,
        king=king_commitment,
        challengers=challengers,
    )
    out_dir = Path(settings.state_dir) / "_rounds" / f"round_{spec['round_id']}"
    log_event(f"starting round block={block} king={king_name} challengers={len(challengers)}")

    with acquire_pod(label=f"r{spec['round_id']}") as pod:
        upload_runtime(pod)
        install_runtime(pod)
        results = run_eval_on_pod(
            pod=pod,
            round_spec=spec,
            out_dir=out_dir,
            timeout_s=settings.eval_round_max_minutes * 60,
        )

    record = process_round(
        state=state,
        pod_results=results,
        king_name=king_name,
        reference_name=spec["teacher_repo"],
        teacher_name=spec["reference_repo"],
        block=block,
        block_hash=block_hash,
        timings=results.get("__per_bench_timing__"),
    )
    new_king, why = resolve_king(state.composite_scores, current_king_model=king_name)
    record["king_after"] = new_king
    record["king_reason"] = why
    if new_king and new_king != king_name:
        new_king_uid = next((c.uid for c in commitments.values() if c.key == new_king), None)
        if new_king_uid is not None:
            state.push_king(new_king_uid)
            king_uid = new_king_uid
            king_name = new_king

    weights = build_emission_weights(
        n_uids=len(mg.hotkeys),
        king_uid=king_uid,
        recent_kings=state.recent_kings,
    )
    if dry_run:
        logger.info(f"[dry-run] weights would be: max={max(weights):.3f} king_uid={king_uid}")
        return
    if not weights or sum(weights) == 0:
        logger.info("no weights to set; skipping set_weights")
        return
    try:
        set_weights(
            sub,
            _wallet(),
            netuid=settings.netuid,
            n_uids=len(mg.hotkeys),
            weights=weights,
            label=f"king={king_uid}",
        )
        log_event(f"set_weights ok; king_uid={king_uid}", level="info")
    except SetWeightsError as exc:
        logger.error(f"set_weights failed: {exc}")
        log_event(f"set_weights failed: {exc}", level="error")


def run(
    *,
    wallet_name: str | None = None,
    hotkey_name: str | None = None,
    once: bool = False,
    dry_run: bool = False,
) -> int:
    if wallet_name:
        settings.wallet_name = wallet_name
    if hotkey_name:
        settings.hotkey_name = hotkey_name
    _install_signal_handlers()
    state = ValidatorState.load(settings.state_dir)
    while not _STOP:
        try:
            _round(state, dry_run=dry_run)
        except Exception as exc:
            logger.exception(f"round failed: {exc}")
            log_event(f"round crashed: {exc}", level="error")
            time.sleep(60)
            if once:
                return 1
        if once:
            return 0
        for _ in range(ROUND_INTERVAL_S):
            if _STOP:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(run(once="--once" in sys.argv, dry_run="--dry-run" in sys.argv))
