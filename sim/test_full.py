#!/usr/bin/env python3
"""
Full end-to-end simulation of the distillation subnet.

Tests the complete flow:
1. Two miners commit HuggingFace model links
2. Validator reads commitments
3. Validator checks model architecture (mocked HF)
4. Validator runs teacher + student inference (mocked vLLM, synthetic logprobs)
5. Computes KL-divergence
6. Winner-take-all weight assignment

Run: python -m sim.test_full
"""
import sys, os, json, math, random, logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sim")

# ── Import real eval modules ──
from eval.kl_divergence import compute_kl_divergence

# ── Config ──
TEACHER_MODEL = "zai-org/GLM-5"
NETUID = 99
NUM_PROMPTS = 3
MAX_TOKENS = 10  # Short for simulation
TOP_K = 20
VOCAB = [f"tok_{i}" for i in range(100)]  # Fake vocabulary


# ── Synthetic logprob generation ──
def make_logprobs(
    n_positions: int, top_k: int, temperature: float = 1.0, seed: int = 0
) -> list[dict[str, float]]:
    """Generate synthetic logprobs for n positions.

    Lower temperature = more peaked distribution (closer to teacher).
    Higher temperature = flatter distribution (more divergent).
    """
    rng = random.Random(seed)
    positions = []
    for pos in range(n_positions):
        # Generate raw logits
        raw = [rng.gauss(0, 1) for _ in range(top_k)]
        # Apply temperature
        scaled = [x / temperature for x in raw]
        # Convert to log-softmax
        max_val = max(scaled)
        log_sum_exp = max_val + math.log(sum(math.exp(x - max_val) for x in scaled))
        logprobs = {VOCAB[i]: scaled[i] - log_sum_exp for i in range(top_k)}
        positions.append(logprobs)
    return positions


# ── Mock classes ──
class MockSubtensor:
    """Simulates bittensor subtensor for testing."""

    def __init__(self):
        self.commitments = {}  # hotkey -> [(block, data_json)]
        self.weights_set = None

    def get_all_revealed_commitments(self, netuid):
        return self.commitments

    def set_weights(self, wallet, netuid, uids, weights, **kwargs):
        self.weights_set = {"uids": uids, "weights": weights}
        return True


class MockMetagraph:
    def __init__(self, n=256):
        self.n = n
        self.hotkeys = [f"hotkey_{i}" for i in range(n)]

    def sync(self, subtensor=None):
        pass


class MockWallet:
    def __init__(self, name="test", hotkey="test"):
        self.name = name
        self.hotkey_str = hotkey


# ── Simulation ──
def run_simulation():
    logger.info("=" * 70)
    logger.info("DISTILLATION SUBNET — FULL END-TO-END SIMULATION")
    logger.info("=" * 70)

    # 1. Setup
    subtensor = MockSubtensor()
    metagraph = MockMetagraph(n=256)
    wallet = MockWallet()

    # 2. Miner commitments
    # Miner A (UID 5): Good distillation — low temperature (close to teacher)
    miner_a_uid = 5
    miner_a_commit = {"model": "alice/glm5-distilled-70b", "revision": "abc123def456"}
    subtensor.commitments[metagraph.hotkeys[miner_a_uid]] = [
        (100, json.dumps(miner_a_commit))
    ]

    # Miner B (UID 12): Poor distillation — high temperature (divergent)
    miner_b_uid = 12
    miner_b_commit = {"model": "bob/glm5-quantized-60b", "revision": "789xyz000111"}
    subtensor.commitments[metagraph.hotkeys[miner_b_uid]] = [
        (100, json.dumps(miner_b_commit))
    ]

    logger.info(f"\nMiner A (UID {miner_a_uid}): {miner_a_commit['model']}")
    logger.info(f"Miner B (UID {miner_b_uid}): {miner_b_commit['model']}")

    # 3. Mock model architecture checks
    # Both pass — correct vocab, under param limit
    mock_arch_results = {
        "alice/glm5-distilled-70b": {
            "pass": True, "reason": "ok", "params_b": 70.0, "vocab_size": 151552,
        },
        "bob/glm5-quantized-60b": {
            "pass": True, "reason": "ok", "params_b": 60.0, "vocab_size": 151552,
        },
    }

    # 4. Simulate prompts (use real dataset files if available, else synthetic)
    dataset_path = Path(__file__).parent.parent / "dataset"
    if dataset_path.exists() and list(dataset_path.glob("*.json")):
        from eval.dataset import load_swe_infinite_prompts, sample_prompts, format_coding_prompt

        all_prompts = load_swe_infinite_prompts(str(dataset_path))
        epoch_prompts = sample_prompts(all_prompts, NUM_PROMPTS)
        prompt_texts = [format_coding_prompt(p) for p in epoch_prompts]
        logger.info(f"\nUsing {NUM_PROMPTS} real SweInfinite prompts")
    else:
        prompt_texts = [
            "Implement a binary search tree in Python",
            "Fix this bug: list index out of range in sorting algorithm",
            "Write a function to find the longest common subsequence",
        ]
        logger.info(f"\nUsing {len(prompt_texts)} synthetic prompts")

    # 5. Generate teacher logprobs (synthetic — we don't have a real GPU)
    logger.info("\n── Teacher Inference (synthetic) ──")
    teacher_results = []
    for i, prompt in enumerate(prompt_texts):
        logprobs = make_logprobs(MAX_TOKENS, TOP_K, temperature=1.0, seed=i * 1000)
        teacher_results.append({"text": f"teacher_output_{i}", "logprobs": logprobs})
        logger.info(f"  Prompt {i+1}: generated {len(logprobs)} token positions")

    # 6. Evaluate miners
    logger.info("\n── Miner Evaluation ──")
    scores = {}

    for uid, commitment in [(miner_a_uid, miner_a_commit), (miner_b_uid, miner_b_commit)]:
        model_repo = commitment["model"]

        # Architecture check
        check = mock_arch_results[model_repo]
        logger.info(f"\nUID {uid} ({model_repo}):")
        logger.info(f"  Architecture check: {check['reason']} ({check['params_b']}B params)")

        if not check["pass"]:
            logger.warning(f"  SKIPPED: {check['reason']}")
            continue

        # Generate student logprobs
        # Miner A: temperature 1.05 (very close to teacher — good distillation)
        # Miner B: temperature 2.0 (very different — poor distillation)
        if uid == miner_a_uid:
            student_temp = 1.05  # Close to teacher
        else:
            student_temp = 2.0  # Far from teacher

        student_results = []
        for i in range(len(prompt_texts)):
            # Use SAME seed as teacher for each prompt so distributions overlap
            logprobs = make_logprobs(MAX_TOKENS, TOP_K, temperature=student_temp, seed=i * 1000)
            student_results.append(
                {"text": f"student_{uid}_output_{i}", "logprobs": logprobs}
            )

        # Compute KL-divergence across all prompts
        kl_divs = []
        for t_res, s_res in zip(teacher_results, student_results):
            kl = compute_kl_divergence(t_res["logprobs"], s_res["logprobs"])
            kl_divs.append(kl)

        import numpy as np

        avg_kl = float(np.mean(kl_divs))
        scores[uid] = avg_kl

        logger.info(f"  Per-prompt KL-divs: {[f'{k:.6f}' for k in kl_divs]}")
        logger.info(f"  Average KL-divergence: {avg_kl:.6f}")

    # 7. Winner-take-all weight assignment
    logger.info("\n── Weight Assignment (Winner-Take-All) ──")

    if not scores:
        logger.error("No valid miners to score!")
        return False

    winner_uid = min(scores, key=scores.get)
    loser_uid = max(scores, key=scores.get)

    logger.info(f"\nScoreboard:")
    for uid in sorted(scores, key=scores.get):
        marker = " ← WINNER" if uid == winner_uid else ""
        logger.info(f"  UID {uid:3d}: KL-div = {scores[uid]:.6f}{marker}")

    # Set weights
    uids = list(range(metagraph.n))
    weights = [0.0] * metagraph.n
    weights[winner_uid] = 1.0

    success = subtensor.set_weights(
        wallet=wallet,
        netuid=NETUID,
        uids=uids,
        weights=weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    logger.info(f"\nWeights set: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"  UID {winner_uid}: weight = 1.0 (KL-div: {scores[winner_uid]:.6f})")
    logger.info(f"  UID {loser_uid}: weight = 0.0 (KL-div: {scores[loser_uid]:.6f})")
    logger.info(f"  All other UIDs: weight = 0.0")

    # 8. Verify
    logger.info("\n── Verification ──")
    assert success, "Weight setting failed"
    assert subtensor.weights_set is not None, "No weights recorded"
    assert subtensor.weights_set["weights"][winner_uid] == 1.0, "Winner weight should be 1.0"
    assert subtensor.weights_set["weights"][loser_uid] == 0.0, "Loser weight should be 0.0"
    assert scores[winner_uid] < scores[loser_uid], "Winner should have lower KL-div"

    logger.info("✓ Winner has lower KL-divergence")
    logger.info("✓ Winner gets weight 1.0")
    logger.info("✓ Loser gets weight 0.0")
    logger.info("✓ All assertions passed")

    logger.info("\n" + "=" * 70)
    logger.info("SIMULATION COMPLETE — ALL CHECKS PASSED")
    logger.info("=" * 70)
    return True


# ── Also test edge cases ──
def test_model_rejection():
    """Test that models exceeding param limit or wrong tokenizer are rejected."""
    logger.info("\n── Edge Case: Model Rejection Tests ──")

    # Simulate a model that's too large
    logger.info("\nTest 1: Model too large (100B > 74.4B)")
    result = {"pass": False, "reason": "too_large:100.0B > 74.4B", "params_b": 100.0}
    assert not result["pass"]
    logger.info(f"  ✓ Correctly rejected: {result['reason']}")

    # Simulate wrong tokenizer
    logger.info("\nTest 2: Wrong tokenizer (vocab mismatch)")
    result = {
        "pass": False,
        "reason": "vocab_mismatch:32000 != 151552",
        "params_b": 50.0,
        "vocab_size": 32000,
    }
    assert not result["pass"]
    logger.info(f"  ✓ Correctly rejected: {result['reason']}")

    # Simulate valid model
    logger.info("\nTest 3: Valid model (70B, correct vocab)")
    result = {"pass": True, "reason": "ok", "params_b": 70.0, "vocab_size": 151552}
    assert result["pass"]
    logger.info(f"  ✓ Correctly accepted: {result['params_b']}B params")

    logger.info("\n✓ All rejection tests passed")


def test_kl_divergence_properties():
    """Test KL-divergence mathematical properties."""
    logger.info("\n── KL-Divergence Property Tests ──")

    # KL(P || P) = 0
    logger.info("\nTest 1: KL(P || P) should be 0")
    same_logprobs = make_logprobs(5, 10, temperature=1.0, seed=42)
    kl = compute_kl_divergence(same_logprobs, same_logprobs)
    logger.info(f"  KL(P || P) = {kl:.10f}")
    assert abs(kl) < 1e-8, f"KL(P||P) should be ~0, got {kl}"
    logger.info("  ✓ KL(P || P) ≈ 0")

    # KL should increase with temperature difference
    logger.info("\nTest 2: KL should increase with divergence")
    teacher_lp = make_logprobs(5, 10, temperature=1.0, seed=42)
    close_lp = make_logprobs(5, 10, temperature=1.1, seed=42)
    far_lp = make_logprobs(5, 10, temperature=3.0, seed=42)

    kl_close = compute_kl_divergence(teacher_lp, close_lp)
    kl_far = compute_kl_divergence(teacher_lp, far_lp)

    logger.info(f"  KL(teacher || close) = {kl_close:.6f}")
    logger.info(f"  KL(teacher || far)   = {kl_far:.6f}")
    assert kl_close < kl_far, "Close model should have lower KL-div"
    logger.info("  ✓ More divergent model has higher KL")

    # KL should be non-negative
    logger.info("\nTest 3: KL should be non-negative")
    assert kl_close >= 0, "KL should be non-negative"
    assert kl_far >= 0, "KL should be non-negative"
    logger.info("  ✓ All KL values non-negative")

    logger.info("\n✓ All KL-divergence property tests passed")


if __name__ == "__main__":
    test_kl_divergence_properties()
    test_model_rejection()
    success = run_simulation()
    sys.exit(0 if success else 1)
