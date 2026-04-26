"""Focused unit tests for SINGLE_EVAL_MODE (one-eval-per-commitment policy).

The validator switched to "one registration → one commitment → one eval" on
2026-04-25. These tests pin the contracts that policy depends on:

* ``select_challengers`` only returns commitments not yet in
  ``state.composite_scores`` (or whose stored commit signature changed).
* ``add_top5_contenders``/``add_dormant_rotation``/``cap_challengers``/
  ``assert_top_contenders_present`` are no-ops with the flag on.
* ``select_king_by_composite`` reads cross-round and prefers higher worst.
* ``commitment_changed`` flags re-commits regardless of which field moved.
* ``evict_stale_evaluated_uids`` clears the gate when a re-commit lands.
* ``merge_composite_scores`` writes one record per non-DQ scored UID.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_torch_modules():
    """Provide a minimal torch stub so importing pod_eval_vllm-adjacent
    modules doesn't pull a real torch wheel into the test environment.
    Mirrors the stub used in test_procedural_bench_generation."""
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    fake.bfloat16 = object()
    fake.float32 = object()
    fake.long = object()
    nn_mod = types.ModuleType("torch.nn")
    nn_func = types.ModuleType("torch.nn.functional")
    nn_func.kl_div = lambda *a, **k: None
    nn_mod.functional = nn_func
    fake.nn = nn_mod
    fake.compile = lambda fn=None, **k: (fn or (lambda x: x))
    fake.cuda = types.ModuleType("torch.cuda")
    fake.cuda.is_available = lambda: False
    sys.modules["torch"] = fake
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_func
    sys.modules["torch.cuda"] = fake.cuda


_stub_torch_modules()


from scripts.validator import single_eval  # noqa: E402
from scripts.validator import challengers as ch_mod  # noqa: E402
from eval import state as state_mod  # noqa: E402


class _FakeState:
    """Minimal stand-in for ValidatorState (just the attributes the
    challenger planner and single-eval helpers read/write)."""

    def __init__(self):
        self.scores = {}
        self.evaluated_uids = set()
        self.composite_scores = {}
        self.permanently_bad_models = set()
        self.model_score_history = {}
        self.top4_leaderboard = {"king": None, "contenders": [], "phase": "maintenance"}
        self.h2h_latest = {}
        self.h2h_history = []
        self.h2h_tested_against_king = {}
        self.dq_reasons = {}

    def save_top4(self):
        pass


def _commit(uid, model, block, hotkey="", revision="main", is_reference=False):
    return {
        uid: {
            "model": model,
            "revision": revision,
            "commit_block": block,
            "hotkey": hotkey,
            "is_reference": is_reference,
        }
    }


class CommitmentChangedTests(unittest.TestCase):
    def test_no_record_means_changed(self):
        self.assertTrue(single_eval.commitment_changed(None, {"model": "m", "revision": "main", "commit_block": 1}))
        self.assertTrue(single_eval.commitment_changed({}, {"model": "m", "revision": "main", "commit_block": 1}))

    def test_same_signature_is_unchanged(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/x", "revision": "main", "commit_block": 100}
        self.assertFalse(single_eval.commitment_changed(rec, info))

    def test_block_change_is_detected(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/x", "revision": "main", "commit_block": 200}
        self.assertTrue(single_eval.commitment_changed(rec, info))

    def test_model_change_is_detected(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/y", "revision": "main", "commit_block": 100}
        self.assertTrue(single_eval.commitment_changed(rec, info))

    def test_bootstrap_record_ignores_block_when_model_matches(self):
        """Regression for round 8045570 (2026-04-25): bootstrap stores the H2H
        round block, not the UID's commit_block. Without this exception every
        bootstrapped UID gets evicted on first restart and re-evaluated, which
        violates the one-eval-per-commitment policy."""
        rec = {
            "model": "miner/x",
            "revision": "main",
            "block": 8042152,
            "_bootstrapped": True,
        }
        info = {"model": "miner/x", "revision": "main", "commit_block": 8044000}
        self.assertFalse(single_eval.commitment_changed(rec, info))

    def test_bootstrap_record_still_detects_model_change(self):
        rec = {
            "model": "miner/x",
            "revision": "main",
            "block": 8042152,
            "_bootstrapped": True,
        }
        info = {"model": "miner/y", "revision": "main", "commit_block": 8044000}
        self.assertTrue(single_eval.commitment_changed(rec, info))

    def test_bootstrap_record_ignores_revision_change(self):
        """h2h_latest frequently records ``revision=None`` even when the
        on-chain commit pinned a specific SHA; bootstrap then stores
        revision="main" (the default fallback), and on next restart the
        live commitment looks like a hash-pinned revision. Without
        ignoring revision for bootstrap records we'd evict every seeded
        UID — round 8045570 (2026-04-25) reproduced exactly this with all
        eight prior-king-era UIDs requeued."""
        rec = {
            "model": "miner/x",
            "revision": "main",
            "block": 8042152,
            "_bootstrapped": True,
        }
        info = {"model": "miner/x", "revision": "abc123", "commit_block": 8044000}
        self.assertFalse(single_eval.commitment_changed(rec, info))


class EvictStaleEvaluatedUidsTests(unittest.TestCase):
    def test_evicts_re_commits_only(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"model": "miner/a", "revision": "main", "block": 100, "worst": 0.5},
            "2": {"model": "miner/b", "revision": "main", "block": 100, "worst": 0.4},
        }
        state.evaluated_uids = {"1", "2"}
        state.scores = {"1": 0.1, "2": 0.2}
        valid_models = {}
        valid_models.update(_commit(1, "miner/a-v2", 200))  # re-commit
        valid_models.update(_commit(2, "miner/b", 100))  # unchanged
        evicted = single_eval.evict_stale_evaluated_uids(state, valid_models)
        self.assertEqual(evicted, ["1"])
        self.assertNotIn("1", state.evaluated_uids)
        self.assertNotIn("1", state.composite_scores)
        self.assertIn("2", state.evaluated_uids)
        self.assertIn("2", state.composite_scores)

    def test_evicts_stale_bootstrap_record_without_evaluated_flag(self):
        """Re-commits whose only stored row is a bootstrapped composite
        (i.e. not yet in evaluated_uids) must still be evicted.

        Regression for round 8046286 (2026-04-25): UID 12 swapped
        natrium43/p1 → natrium43/t2 but was filtered out of the queue
        because ``select_challengers`` short-circuited on the stale
        ``composite_scores`` row before the eviction pass could run.
        """
        state = _FakeState()
        state.composite_scores = {
            "12": {
                "model": "natrium43/p1",
                "revision": "main",
                "block": 5000,
                "worst": 0.4,
                "_bootstrapped": True,
            },
        }
        state.evaluated_uids = set()  # bootstrap path doesn't touch this
        valid_models = _commit(12, "natrium43/t2", 8044887)  # re-commit
        evicted = single_eval.evict_stale_evaluated_uids(state, valid_models)
        self.assertEqual(evicted, ["12"])
        self.assertNotIn("12", state.composite_scores)

    def test_does_not_touch_uids_outside_valid_models(self):
        """Eviction only looks at UIDs in valid_models — keeps composite
        rows for models that are temporarily failing precheck."""
        state = _FakeState()
        state.composite_scores = {
            "7": {"model": "miner/x", "revision": "main", "block": 100, "worst": 0.4},
        }
        state.evaluated_uids = {"7"}
        valid_models = {}  # UID 7 not currently valid
        evicted = single_eval.evict_stale_evaluated_uids(state, valid_models)
        self.assertEqual(evicted, [])
        self.assertIn("7", state.composite_scores)
        self.assertIn("7", state.evaluated_uids)


class SelectChallengersSingleEvalTests(unittest.TestCase):
    def setUp(self):
        os.environ["SINGLE_EVAL_MODE"] = "1"

    def tearDown(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_returns_only_never_composite_scored(self):
        from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
        state = _FakeState()
        state.composite_scores = {
            "1": {
                "model": "miner/a", "revision": "main", "block": 100,
                "worst": 0.5, "version": COMPOSITE_SHADOW_VERSION,
            },
        }
        state.evaluated_uids = {"1"}
        state.scores = {"1": 0.1}
        valid_models = {}
        valid_models.update(_commit(1, "miner/a", 100))
        valid_models.update(_commit(2, "miner/b", 200))
        valid_models.update(_commit(3, "miner/c", 300))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=1, king_kl=0.1, epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {2, 3})

    def test_forces_king_when_composite_version_stale(self):
        """v27: when the king's stored composite is from an older schema
        (composite_shadow_version), select_challengers must force the
        king back into the round so dethronement comparisons stay
        apples-to-apples with challengers evaluated under the new
        schema. This closes the prompt-variance unfairness that surfaced
        in the Discord thread on 2026-04-26."""
        from scripts.validator.composite import COMPOSITE_SHADOW_VERSION
        stale = max(0, int(COMPOSITE_SHADOW_VERSION) - 5)
        state = _FakeState()
        state.composite_scores = {
            "1": {
                "model": "miner/a", "revision": "main", "block": 100,
                "worst": 0.5, "version": stale,
            },
        }
        state.evaluated_uids = {"1"}
        state.scores = {"1": 0.1}
        valid_models = {}
        valid_models.update(_commit(1, "miner/a", 100))
        valid_models.update(_commit(2, "miner/b", 200))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=1, king_kl=0.1, epoch_count=1,
        )
        self.assertIn(1, challengers,
                      "stale-version king must be re-evaluated under v27")
        self.assertIn(2, challengers,
                      "non-king challengers should still be picked up")

    def test_picks_up_re_commits(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"model": "miner/a", "revision": "main", "block": 100, "worst": 0.5},
        }
        state.evaluated_uids = {"1"}
        valid_models = _commit(1, "miner/a", 200)  # new block = re-commit
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {1})
        self.assertNotIn("1", state.composite_scores)

    def test_skips_reference_uid(self):
        state = _FakeState()
        valid_models = _commit(-1, "Qwen/Qwen3.5-4B", 0, is_reference=True)
        valid_models.update(_commit(2, "miner/b", 200))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {2})

    def test_skips_permanently_bad(self):
        state = _FakeState()
        state.permanently_bad_models = {"miner/bad"}
        valid_models = _commit(2, "miner/bad", 200)
        valid_models.update(_commit(3, "miner/c", 300))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {3})

    def test_skips_evaluated_uid_without_score(self):
        """Strict no-re-eval: a UID in evaluated_uids never re-runs even if
        its score row is missing.

        Regression for Discord 2026-04-25, sebastian: validator was looping
        in 24-model rounds because precheck-DQ'd UIDs had been removed from
        ``state.scores`` while remaining in ``evaluated_uids`` — old filter
        let them slip back into the queue."""
        state = _FakeState()
        state.evaluated_uids = {"5"}
        state.scores = {}
        valid_models = _commit(5, "miner/x", 100)
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(challengers, {})

    def test_max_per_round_caps_with_fifo(self):
        """Cap kicks in only when we exceed the threshold; oldest commit
        wins, newest defers to next round."""
        state = _FakeState()
        valid_models = {}
        # 12 candidates, blocks 100..1200.
        for i in range(12):
            uid = i + 1
            valid_models.update(_commit(uid, f"miner/{uid}", 100 * (i + 1)))
        # Patch the cap to 5 for the duration of this test.
        with patch.object(single_eval, "SINGLE_EVAL_MAX_PER_ROUND", 5):
            challengers = ch_mod.select_challengers(
                valid_models, state, king_uid=None, king_kl=float("inf"),
                epoch_count=1,
            )
        # 5 oldest commit_blocks: 100, 200, 300, 400, 500 → UIDs 1..5
        self.assertEqual(set(challengers.keys()), {1, 2, 3, 4, 5})

    def test_max_per_round_zero_disables_cap(self):
        """SINGLE_EVAL_MAX_PER_ROUND=0 means uncapped (legacy behaviour)."""
        state = _FakeState()
        valid_models = {}
        for i in range(8):
            uid = i + 1
            valid_models.update(_commit(uid, f"miner/{uid}", 100 * (i + 1)))
        with patch.object(single_eval, "SINGLE_EVAL_MAX_PER_ROUND", 0):
            challengers = ch_mod.select_challengers(
                valid_models, state, king_uid=None, king_kl=float("inf"),
                epoch_count=1,
            )
        self.assertEqual(len(challengers), 8)

    def test_max_per_round_under_cap_no_truncation(self):
        """When pending < cap, all candidates advance unchanged."""
        state = _FakeState()
        valid_models = {}
        for i in range(3):
            uid = i + 1
            valid_models.update(_commit(uid, f"miner/{uid}", 100 * (i + 1)))
        with patch.object(single_eval, "SINGLE_EVAL_MAX_PER_ROUND", 10):
            challengers = ch_mod.select_challengers(
                valid_models, state, king_uid=None, king_kl=float("inf"),
                epoch_count=1,
            )
        self.assertEqual(set(challengers.keys()), {1, 2, 3})


class ReEvalHelpersAreNoopsTests(unittest.TestCase):
    def setUp(self):
        os.environ["SINGLE_EVAL_MODE"] = "1"

    def tearDown(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_top5_contenders_noop(self):
        state = _FakeState()
        state.top4_leaderboard["contenders"] = [
            {"uid": 7, "model": "miner/x", "h2h_kl": 0.05},
        ]
        valid_models = _commit(7, "miner/x", 100)
        challengers = {}
        ch_mod.add_top5_contenders(challengers, valid_models, state, king_uid=1)
        self.assertEqual(challengers, {})

    def test_dormant_rotation_noop(self):
        state = _FakeState()
        state.scores = {"5": 0.05, "6": 0.06}
        valid_models = _commit(5, "miner/x", 100)
        valid_models.update(_commit(6, "miner/y", 200))
        challengers = {}
        ch_mod.add_dormant_rotation(challengers, valid_models, state, king_uid=1, king_kl=0.10)
        self.assertEqual(challengers, {})

    def test_cap_challengers_noop(self):
        state = _FakeState()
        challengers = {i: {"model": f"m/{i}", "commit_block": i} for i in range(50)}
        ch_mod.cap_challengers(challengers, state, king_uid=1)
        self.assertEqual(len(challengers), 50)  # nothing truncated

    def test_assert_top_contenders_noop(self):
        state = _FakeState()
        state.top4_leaderboard["contenders"] = [
            {"uid": 9, "model": "miner/lb", "h2h_kl": 0.05},
        ]
        valid_models = _commit(9, "miner/lb", 100)
        challengers = {}
        # Should not warn, force, or evict — single-eval has no concept of
        # "must be present every round."
        ch_mod.assert_top_contenders_present(challengers, valid_models, state, king_uid=1)
        self.assertEqual(challengers, {})


class MergeCompositeScoresTests(unittest.TestCase):
    def test_merges_one_record_per_scored_row(self):
        state = _FakeState()
        h2h_results = [
            {
                "uid": 5, "model": "miner/a",
                "composite": {"worst": 0.42, "weighted": 0.55,
                              "axes": {"kl": 0.6, "capability": 0.4},
                              "present_count": 2},
            },
            {  # DQ row should be skipped
                "uid": 6, "model": "miner/dq", "disqualified": True,
                "composite": {"worst": 0.30},
            },
            {  # Reference row should be skipped
                "uid": -1, "model": "Qwen/Qwen3.5-4B", "is_reference": True,
                "composite": {"worst": 0.50},
            },
            {  # No composite payload — skipped
                "uid": 7, "model": "miner/missing", "composite": {"worst": None},
            },
        ]
        models_to_eval = {
            5: {"model": "miner/a", "revision": "main", "commit_block": 1234},
            6: {"model": "miner/dq", "revision": "main", "commit_block": 1235},
            -1: {"model": "Qwen/Qwen3.5-4B", "is_reference": True},
            7: {"model": "miner/missing", "revision": "main", "commit_block": 1236},
        }
        n = single_eval.merge_composite_scores(state, h2h_results, models_to_eval, current_block=2000)
        self.assertEqual(n, 1)
        self.assertIn("5", state.composite_scores)
        record = state.composite_scores["5"]
        self.assertEqual(record["model"], "miner/a")
        self.assertEqual(record["block"], 1234)
        self.assertAlmostEqual(record["worst"], 0.42)
        self.assertAlmostEqual(record["weighted"], 0.55)


class BootstrapCompositeFromH2HTests(unittest.TestCase):
    """Bootstrap covers latest H2H + every entry in h2h_history.

    Originally only seeded from ``state.h2h_latest`` (~8 UIDs/round). Without
    historical seeding, every UID that scored before single-eval-mode was
    flipped on would be treated as a new commitment after a state reset
    and re-evaluated."""

    def test_seeds_from_latest_only_when_history_empty(self):
        state = _FakeState()
        state.h2h_latest = {
            "block": 100,
            "results": [
                {"uid": 5, "model": "miner/a",
                 "composite": {"worst": 0.5, "weighted": 0.55}},
            ],
        }
        n = single_eval.bootstrap_composite_from_h2h(state)
        self.assertEqual(n, 1)
        self.assertIn("5", state.composite_scores)
        self.assertTrue(state.composite_scores["5"].get("_bootstrapped"))

    def test_seeds_from_history_too(self):
        """A historical round is processed even when h2h_latest is empty."""
        state = _FakeState()
        state.h2h_latest = {}
        state.h2h_history = [
            {
                "block": 50,
                "results": [
                    {"uid": 7, "model": "miner/x",
                     "composite": {"worst": 0.6, "weighted": 0.6}},
                ],
            },
            {
                "block": 100,
                "results": [
                    {"uid": 8, "model": "miner/y",
                     "composite": {"worst": 0.4, "weighted": 0.5}},
                ],
            },
        ]
        n = single_eval.bootstrap_composite_from_h2h(state)
        self.assertEqual(n, 2)
        self.assertIn("7", state.composite_scores)
        self.assertIn("8", state.composite_scores)

    def test_newer_history_wins_over_older(self):
        """When the same UID appears in two historical rounds, the most
        recent (highest block) record is the one that survives."""
        state = _FakeState()
        state.h2h_latest = {}
        state.h2h_history = [
            {
                "block": 100,
                "results": [
                    {"uid": 7, "model": "miner/x",
                     "composite": {"worst": 0.42, "weighted": 0.5}},
                ],
            },
            {
                "block": 50,
                "results": [
                    {"uid": 7, "model": "miner/x",
                     "composite": {"worst": 0.30, "weighted": 0.4}},
                ],
            },
        ]
        single_eval.bootstrap_composite_from_h2h(state)
        self.assertAlmostEqual(state.composite_scores["7"]["worst"], 0.42)
        self.assertEqual(state.composite_scores["7"]["block"], 100)

    def test_existing_records_preserved(self):
        """An existing composite_scores record (e.g. from merge after a
        completed round) is never overwritten by the bootstrap pass."""
        state = _FakeState()
        state.composite_scores = {
            "5": {"worst": 0.99, "weighted": 0.99, "model": "miner/a"},
        }
        state.h2h_latest = {
            "block": 100,
            "results": [
                {"uid": 5, "model": "miner/a",
                 "composite": {"worst": 0.10, "weighted": 0.10}},
            ],
        }
        single_eval.bootstrap_composite_from_h2h(state)
        self.assertAlmostEqual(state.composite_scores["5"]["worst"], 0.99)

    def test_persist_called_when_save_helper_present(self):
        """Bootstrap must trigger eager persistence so a second restart
        reads from disk instead of re-walking history."""
        state = _FakeState()
        state.h2h_latest = {
            "block": 100,
            "results": [
                {"uid": 5, "model": "miner/a",
                 "composite": {"worst": 0.5, "weighted": 0.55}},
            ],
        }
        calls = []
        state.save_composite_scores = lambda: calls.append("save")
        single_eval.bootstrap_composite_from_h2h(state)
        self.assertEqual(calls, ["save"])

    def test_persist_called_after_merge(self):
        """``merge_composite_scores`` also persists immediately so end-of-
        round results survive a crash before ``state.save()`` runs."""
        state = _FakeState()
        calls = []
        state.save_composite_scores = lambda: calls.append("save")
        h2h_results = [
            {"uid": 5, "model": "miner/a",
             "composite": {"worst": 0.5, "weighted": 0.55,
                           "axes": {}, "present_count": 0}},
        ]
        models_to_eval = {5: {"model": "miner/a", "revision": "main",
                              "commit_block": 1234}}
        single_eval.merge_composite_scores(state, h2h_results, models_to_eval, current_block=2000)
        self.assertEqual(calls, ["save"])


class SelectKingByCompositeTests(unittest.TestCase):
    def test_picks_highest_worst(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"worst": 0.5, "weighted": 0.55, "n_axes": 20},
            "2": {"worst": 0.7, "weighted": 0.65, "n_axes": 20},
            "3": {"worst": 0.6, "weighted": 0.62, "n_axes": 20},
        }
        valid_models = {1: {"model": "a"}, 2: {"model": "b"}, 3: {"model": "c"}}
        uid, rec = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 2)
        self.assertAlmostEqual(rec["worst"], 0.7)

    def test_ignores_dq_uids(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"worst": 0.5, "n_axes": 20},
            "2": {"worst": 0.9, "n_axes": 20},
        }
        # UID 2 is DQ'd at its current commit_block.
        state.dq_reasons = {"hk2:100": "anti-finetune detected"}
        valid_models = {
            1: {"model": "a", "hotkey": "hk1", "commit_block": 100},
            2: {"model": "b", "hotkey": "hk2", "commit_block": 100},
        }
        uid, rec = single_eval.select_king_by_composite(
            state, valid_models, uid_to_hotkey={1: "hk1", 2: "hk2"},
        )
        self.assertEqual(uid, 1)

    def test_returns_none_when_no_candidates(self):
        state = _FakeState()
        valid_models = {1: {"model": "a"}}
        uid, rec = single_eval.select_king_by_composite(state, valid_models)
        self.assertIsNone(uid)
        self.assertIsNone(rec)

    def test_prior_king_wins_noise_level_worst_tie(self):
        """Round 8045570 (2026-04-25): all eight bootstrapped UIDs tied at
        composite.worst=0 (AIME=0 across the board). Within-noise weighted
        differences (<3%) shouldn't flip the crown, otherwise the king
        would oscillate every round on procedural-prompt noise."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        # Tight cluster — challenger weighted 0.66 vs king 0.6491 = +1.7%
        # which is *below* the 3% dethrone margin → king holds.
        state.composite_scores = {
            "48":  {"worst": 0.0, "weighted": 0.6491, "n_axes": 20, "model": "ghost-94/sn97-solution-5"},
            "111": {"worst": 0.0, "weighted": 0.6600, "n_axes": 20, "model": "best26/sn97-best-w0p3"},
            "112": {"worst": 0.0, "weighted": 0.6580, "n_axes": 20, "model": "ncaagcc/7664460"},
            "101": {"worst": 0.0, "weighted": 0.6520, "n_axes": 20, "model": "weedyweed/k"},
        }
        valid_models = {48: {"model": "ghost-94/sn97-solution-5"},
                        111: {"model": "best26/sn97-best-w0p3"},
                        112: {"model": "ncaagcc/7664460"},
                        101: {"model": "weedyweed/k"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 48)

    def test_meaningfully_better_challenger_dethrones_at_saturated_floor(self):
        """Counterpart to the above: when both prior king and challenger sit
        at worst=0.0 *and* the challenger's weighted clears the
        SINGLE_EVAL_DETHRONE_MARGIN gate, the challenger correctly wins.
        Without this, a permanent worst=0.0 king blocks all replacements."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        state.composite_scores = {
            "48":  {"worst": 0.0, "weighted": 0.50, "n_axes": 20},
            "111": {"worst": 0.0, "weighted": 0.78, "n_axes": 20},  # +56% relative on weighted
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_prior_king_preserved_over_legacy_higher_worst(self):
        """Regression for the broader-bootstrap rollout (2026-04-25):
        h2h_history seeded a 3-axis legacy record for UID 25 with worst=0.73
        — way higher than the current king's 20-axis worst=0.0 — but UID 25
        had been disqualified months earlier and shouldn't have crowned at
        all. The fast-path prior-king check skips this entirely; the schema
        guard also blocks UID 25 if the prior king were ineligible."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        state.composite_scores = {
            "48": {"worst": 0.0, "weighted": 0.6491, "n_axes": 20, "model": "ghost-94/sn97-solution-5"},
            "25": {"worst": 0.73, "weighted": 0.85, "n_axes": 3, "model": "Crocodile0125/ste"},
        }
        valid_models = {48: {"model": "ghost-94/sn97-solution-5"},
                        25: {"model": "Crocodile0125/ste"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 48)

    def test_legacy_record_skipped_when_no_prior_king(self):
        """Without a prior king to fast-path through, the schema guard still
        excludes legacy 3-axis records. Should fall back through to the
        all-records pool only if no schema-current records exist."""
        state = _FakeState()
        state.h2h_latest = {}
        state.composite_scores = {
            "25": {"worst": 0.73, "weighted": 0.85, "n_axes": 3, "model": "old"},
            "48": {"worst": 0.10, "weighted": 0.55, "n_axes": 20, "model": "current"},
        }
        valid_models = {25: {"model": "old"}, 48: {"model": "current"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 48)

    def test_legacy_record_used_only_when_no_modern_alternatives(self):
        """If only legacy records exist, the planner crowns the legacy king
        rather than returning None — graceful degradation."""
        state = _FakeState()
        state.h2h_latest = {}
        state.composite_scores = {
            "25": {"worst": 0.73, "weighted": 0.85, "n_axes": 3, "model": "old"},
            "26": {"worst": 0.50, "weighted": 0.60, "n_axes": 3, "model": "older"},
        }
        valid_models = {25: {"model": "old"}, 26: {"model": "older"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 25)

    def test_prior_king_disqualified_falls_back_to_composite(self):
        """If h2h_latest's king is no longer eligible (DQ'd, removed from
        valid_models), the planner falls back to composite-worst pick from
        schema-current records."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        state.composite_scores = {
            "48":  {"worst": 0.30, "weighted": 0.40, "n_axes": 20},
            "111": {"worst": 0.50, "weighted": 0.60, "n_axes": 20},
        }
        valid_models = {111: {"model": "b"}}  # UID 48 absent → ineligible
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_higher_worst_dethrones_prior_king_when_meaningfully_better(self):
        """As of 2026-04-26 the prior-king preference is a stability *bias*,
        not a hard lock: a challenger with a measurably better composite
        (clears SINGLE_EVAL_DETHRONE_MARGIN) takes the crown. Previously
        the fast path returned the prior king unconditionally on
        eligibility, which silently locked the crown forever.

        Here UID 111 has worst=0.50 vs UID 48's worst=0.40 — a 25% relative
        win that clears the 3% margin. UID 48 is also dropped from
        valid_models for parity with the original test, but the dethrone
        check would fire either way."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        state.composite_scores = {
            "48":  {"worst": 0.40, "weighted": 0.65, "n_axes": 20},
            "111": {"worst": 0.50, "weighted": 0.62, "n_axes": 20},
        }
        valid_models = {111: {"model": "b"}}  # UID 48 not in valid_models
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_no_prior_king_uses_weighted_tiebreaker(self):
        """Pure first-boot scenario: h2h_latest empty/missing king. Tiebreaker
        falls through to weighted, then UID."""
        state = _FakeState()
        state.h2h_latest = {}
        state.composite_scores = {
            "48":  {"worst": 0.0, "weighted": 0.6491, "n_axes": 20},
            "111": {"worst": 0.0, "weighted": 0.6771, "n_axes": 20},
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_v_current_records_preferred_over_legacy(self):
        """2026-04-26 schema bump (v12 → v13: long_context_bench grader hardened).
        Pre-v13 records have lc_bench=1.0 because the lenient substring grader
        rewarded "dump every code" attacks; post-v13 records use the strict
        confuser-rejection grader. King selection must prefer v_current
        records when both tiers exist, otherwise the inflated lc/wgt scores
        of v12 records would block a competing v13 challenger.

        Setup mirrors the early-v13 transition: UID 48 has a higher worst
        and weighted under the OLD grader (v12), UID 111 has slightly lower
        scores under the NEW grader (v13). Without version filtering, UID 48
        wins. With it, UID 111 wins because v13 is the only tier with
        candidates that both sides can be fairly compared on."""
        state = _FakeState()
        state.h2h_latest = {}
        v_current = single_eval._KING_SELECTION_MIN_VERSION
        state.composite_scores = {
            "48":  {"worst": 0.50, "weighted": 0.70, "n_axes": 20, "version": v_current - 1},
            "111": {"worst": 0.40, "weighted": 0.60, "n_axes": 20, "version": v_current},
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_falls_through_when_no_v_current_records(self):
        """Transition window: after a schema bump, v_current records don't
        exist yet. Selector must gracefully fall through to schema-current
        legacy records (Tier 2) so we don't go kingless."""
        state = _FakeState()
        state.h2h_latest = {}
        v_current = single_eval._KING_SELECTION_MIN_VERSION
        state.composite_scores = {
            "48":  {"worst": 0.50, "weighted": 0.70, "n_axes": 20, "version": v_current - 1},
            "111": {"worst": 0.30, "weighted": 0.50, "n_axes": 20, "version": v_current - 2},
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 48)

    def test_prior_king_v12_gets_dethrone_check_against_v13_challenger(self):
        """Critical guarantee: when the only candidate-pool records are v13
        (after the schema bump fully takes hold) but the prior king is
        still v12, they MUST get a margin check before losing the crown.
        Otherwise a single weak v13 challenger steals the throne by
        default during the transition.

        Prior king UID 48 (v12, weighted=0.85) vs lone v13 challenger UID 111
        (weighted=0.60). UID 111 is in Tier 1 (the only candidate); UID 48
        isn't. But the dethrone gate compares them anyway and rejects UID
        111 because they're not measurably better."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        v_current = single_eval._KING_SELECTION_MIN_VERSION
        state.composite_scores = {
            "48":  {"worst": 0.40, "weighted": 0.85, "n_axes": 20, "version": v_current - 1},
            "111": {"worst": 0.40, "weighted": 0.60, "n_axes": 20, "version": v_current},
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 48, "prior king must get a margin check even when on a stale grader")

    def test_v13_challenger_dethrones_v12_king_when_clearly_better(self):
        """Counterpart to the above: the prior-king margin check should NOT
        permanently lock the crown to a v12 record. A v13 challenger that
        clears the dethrone gate (3% margin on worst or weighted) takes the
        crown — that's the whole point of the schema bump."""
        state = _FakeState()
        state.h2h_latest = {"king_uid": 48}
        v_current = single_eval._KING_SELECTION_MIN_VERSION
        state.composite_scores = {
            "48":  {"worst": 0.40, "weighted": 0.55, "n_axes": 20, "version": v_current - 1},
            "111": {"worst": 0.50, "weighted": 0.70, "n_axes": 20, "version": v_current},  # clear win
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 111)

    def test_records_without_version_field_treated_as_legacy(self):
        """Records stored before the version field was introduced have
        ``version=None`` (or the field missing entirely). They should be
        treated as legacy (i.e. not Tier-1 eligible)."""
        state = _FakeState()
        state.h2h_latest = {}
        v_current = single_eval._KING_SELECTION_MIN_VERSION
        state.composite_scores = {
            # UID 48: legacy, version field missing
            "48":  {"worst": 0.50, "weighted": 0.70, "n_axes": 20},
            # UID 111: explicit None version (also legacy)
            "111": {"worst": 0.55, "weighted": 0.75, "n_axes": 20, "version": None},
            # UID 222: schema-current
            "222": {"worst": 0.30, "weighted": 0.50, "n_axes": 20, "version": v_current},
        }
        valid_models = {48: {"model": "a"}, 111: {"model": "b"}, 222: {"model": "c"}}
        uid, _ = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 222, "v_current record (Tier 1) must beat legacy records (Tier 2)")


class ResolveDethroneTests(unittest.TestCase):
    def test_no_incumbent_accepts_any_positive(self):
        ch = {"worst": 0.4}
        self.assertTrue(single_eval.resolve_dethrone(None, None, 5, ch))
        self.assertFalse(single_eval.resolve_dethrone(None, None, 5, {"worst": 0}))

    def test_margin_required(self):
        inc = {"worst": 0.50}
        # 3% margin: 0.50 * 1.03 = 0.515. Anything <= 0.515 should not dethrone.
        self.assertFalse(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.51}, margin=0.03))
        self.assertFalse(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.515}, margin=0.03))
        self.assertTrue(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.52}, margin=0.03))

    def test_self_returns_true(self):
        # Same UID compared against itself is always "wins" — used in the
        # crown-retention branch where the king's own composite is the
        # benchmark and no challenger has surpassed it.
        inc = {"worst": 0.5}
        self.assertTrue(single_eval.resolve_dethrone(7, inc, 7, inc))

    def test_saturated_floor_uses_weighted_tiebreaker(self):
        """2026-04-26: ~45% of stored composite records have worst=0.0 (any
        single zero axis floors min). Without a saturated-floor tiebreaker,
        a never-ending stream of worst=0.0 challengers can never dethrone
        the worst=0.0 incumbent, even when their weighted scores are wildly
        different. Use weighted with the same relative margin in this case."""
        inc = {"worst": 0.0, "weighted": 0.50}
        # Weighted threshold: 0.50 * 1.03 = 0.515.
        # Below threshold → no dethrone.
        self.assertFalse(
            single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.0, "weighted": 0.51}, margin=0.03)
        )
        self.assertFalse(
            single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.0, "weighted": 0.515}, margin=0.03)
        )
        # Above threshold → dethrone.
        self.assertTrue(
            single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.0, "weighted": 0.78}, margin=0.03)
        )

    def test_worst_regression_blocks_weighted_tiebreaker(self):
        """If challenger's worst regressed past the margin compared to the
        incumbent's, the weighted tiebreaker MUST NOT save them — the
        no-axis-can-be-broken guarantee is what worst exists to enforce."""
        inc = {"worst": 0.30, "weighted": 0.50}
        ch = {"worst": 0.0, "weighted": 0.99}
        self.assertFalse(
            single_eval.resolve_dethrone(7, inc, 8, ch, margin=0.03)
        )

    def test_saturated_floor_falls_through_when_weighted_missing(self):
        """If either side lacks a weighted score, the tiebreaker can't
        decide and we hold the incumbent — fail-safe."""
        inc = {"worst": 0.0}  # no weighted
        ch = {"worst": 0.0, "weighted": 0.78}
        self.assertFalse(
            single_eval.resolve_dethrone(7, inc, 8, ch, margin=0.03)
        )

    def test_tied_non_saturated_worst_uses_weighted_tiebreaker(self):
        """2026-04-26 fix (Round 9 reproduction): when both UIDs sit at the
        same low-resolution-quantum worst (e.g. both 0.333 because n=3 on
        the worst axis), the weighted score must decide — otherwise a
        4%-better challenger gets silently rejected and the incumbent
        keeps the crown by default (the original Round 9 bug: UID 93
        weighted=0.6833 lost to UID 89 weighted=0.6567 on a worst tie)."""
        inc = {"worst": 0.333, "weighted": 0.6567}
        # Above weighted-margin (3% of 0.6567 = 0.0197 → threshold 0.6764).
        ch_above = {"worst": 0.333, "weighted": 0.6833}
        self.assertTrue(
            single_eval.resolve_dethrone(7, inc, 8, ch_above, margin=0.03)
        )
        # Below weighted-margin → still rejected.
        ch_below = {"worst": 0.333, "weighted": 0.6700}
        self.assertFalse(
            single_eval.resolve_dethrone(7, inc, 8, ch_below, margin=0.03)
        )

    def test_tied_worst_within_margin_uses_weighted(self):
        """Worst values within ±margin still defer to weighted, even when
        not exactly equal — the tied-region handling must be symmetric."""
        inc = {"worst": 0.50, "weighted": 0.60}
        # Within ±3% of 0.50: 0.485..0.515. Both sides land here.
        ch = {"worst": 0.49, "weighted": 0.70}  # weighted clears 0.60*1.03=0.618.
        self.assertTrue(
            single_eval.resolve_dethrone(7, inc, 8, ch, margin=0.03)
        )


class IsSingleEvalModeTests(unittest.TestCase):
    def test_default_off(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)
        self.assertFalse(single_eval.is_single_eval_mode())

    def test_explicit_on(self):
        with patch.dict(os.environ, {"SINGLE_EVAL_MODE": "1"}):
            self.assertTrue(single_eval.is_single_eval_mode())

    def test_other_values_off(self):
        for v in ("0", "true", "yes", "ok"):
            with patch.dict(os.environ, {"SINGLE_EVAL_MODE": v}):
                if v == "1":
                    continue
                self.assertFalse(single_eval.is_single_eval_mode(),
                                 f"value={v!r} should be off")


class BackwardCompatTests(unittest.TestCase):
    """With the flag OFF, the legacy planner path must remain unchanged."""

    def setUp(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_legacy_select_challengers_unchanged(self):
        state = _FakeState()
        state.scores = {"1": 0.1}
        state.evaluated_uids = {"1"}
        valid_models = _commit(1, "miner/a", 100)
        valid_models.update(_commit(2, "miner/b", 200))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=1, king_kl=0.1, epoch_count=1,
        )
        # In legacy mode, UID 1 is skipped (already in evaluated+scores) and
        # UID 2 is included as a P3 (never-evaluated). That's the existing
        # behavior; we want to confirm the SINGLE_EVAL gate didn't change it.
        self.assertEqual(set(challengers.keys()), {2})


if __name__ == "__main__":
    unittest.main()
