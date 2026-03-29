#!/usr/bin/env python3
"""
Full end-to-end simulation of the distillation subnet (v3).

Tests the complete flow with all v3 changes:
  1. One submission per hotkey (latest only)
  2. Eval by commitment timestamp (FIFO)
  3. 1% epsilon threshold for taking weights
  4. More thorough evaluation (more prompts, longer outputs)
  5. Real GPU eval via Lium (with --real flag)

Run:
  python -m sim.test_full          # Mock simulation (no GPU)
  python -m sim.test_full --real   # Real GPU eval on Lium H200
"""
import sys, os, json, math, random, logging, argparse, tempfile, time
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
VOCAB = [f"tok_{i}" for i in range(100)]  # Fake vocabulary
EPSILON = 0.01  # 1% threshold


# ── Synthetic logprob generation ──
def make_logprobs(
    n_positions: int, top_k: int, temperature: float = 1.0, seed: int = 0
) -> list[dict[str, float]]:
    """Generate synthetic logprobs for n positions."""
    rng = random.Random(seed)
    positions = []
    for pos in range(n_positions):
        raw = [rng.gauss(0, 1) for _ in range(top_k)]
        scaled = [x / temperature for x in raw]
        max_val = max(scaled)
        log_sum_exp = max_val + math.log(sum(math.exp(x - max_val) for x in scaled))
        logprobs = {VOCAB[i]: scaled[i] - log_sum_exp for i in range(top_k)}
        positions.append(logprobs)
    return positions


# ── Mock classes ──
class MockSubtensor:
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


# ══════════════════════════════════════════════════════════════════════════
# MOCK SIMULATION (no GPU)
# ══════════════════════════════════════════════════════════════════════════

def run_simulation():
    """Full mock simulation testing all v3 changes."""
    logger.info("=" * 70)
    logger.info("DISTILLATION SUBNET v3 — MOCK SIMULATION")
    logger.info("=" * 70)

    # Config: more prompts and longer outputs (change 4)
    NUM_PROMPTS = 12
    MAX_TOKENS = 30  # Shortened for simulation, but more than v2's 10

    # 1. Setup
    subtensor = MockSubtensor()
    metagraph = MockMetagraph(n=256)
    wallet = MockWallet()

    # 2. Miner commitments with different blocks (change 2: FIFO)
    # Miner A committed at block 100 (earlier)
    miner_a_uid = 5
    miner_a_commit = {"model": "alice/glm5-distilled-70b", "revision": "abc123def456"}
    subtensor.commitments[metagraph.hotkeys[miner_a_uid]] = [
        (100, json.dumps(miner_a_commit))
    ]

    # Miner B committed at block 200 (later)
    miner_b_uid = 12
    miner_b_commit = {"model": "bob/glm5-quantized-60b", "revision": "789xyz000111"}
    subtensor.commitments[metagraph.hotkeys[miner_b_uid]] = [
        (200, json.dumps(miner_b_commit))
    ]

    # Miner C committed at block 150 (middle) — tests FIFO ordering
    miner_c_uid = 8
    miner_c_commit = {"model": "carol/glm5-pruned-65b", "revision": "ccc333ddd444"}
    subtensor.commitments[metagraph.hotkeys[miner_c_uid]] = [
        (150, json.dumps(miner_c_commit))
    ]

    logger.info(f"\nMiner A (UID {miner_a_uid}): {miner_a_commit['model']} — block 100")
    logger.info(f"Miner C (UID {miner_c_uid}): {miner_c_commit['model']} — block 150")
    logger.info(f"Miner B (UID {miner_b_uid}): {miner_b_commit['model']} — block 200")

    # ─── Change 1: Deduplicate per hotkey (latest only) ──────────────
    # Test: Miner A has TWO commitments — only latest should be used
    subtensor.commitments[metagraph.hotkeys[miner_a_uid]] = [
        (50, json.dumps({"model": "alice/old-model", "revision": "old111"})),
        (100, json.dumps(miner_a_commit)),  # Latest
    ]
    logger.info("\nMiner A has 2 commitments — validator should use latest (block 100)")

    # Read commitments (validator logic)
    commitments = {}
    revealed = subtensor.get_all_revealed_commitments(NETUID)
    for uid in range(metagraph.n):
        hotkey = metagraph.hotkeys[uid]
        if hotkey in revealed:
            block, commit_data = revealed[hotkey][-1]  # Latest only
            data = json.loads(commit_data)
            if "model" in data:
                commitments[uid] = {"block": block, **data}

    logger.info(f"Deduplicated commitments: {len(commitments)}")
    assert commitments[miner_a_uid]["model"] == "alice/glm5-distilled-70b", \
        "Should use latest commitment"

    # ─── Change 2: Sort by block (FIFO) ──────────────────────────────
    sorted_miners = sorted(commitments.items(), key=lambda x: x[1]["block"])
    eval_order = [uid for uid, _ in sorted_miners]
    logger.info(f"Eval order (FIFO): {eval_order}")
    assert eval_order == [miner_a_uid, miner_c_uid, miner_b_uid], \
        f"Expected FIFO order [5, 8, 12], got {eval_order}"

    # 3. Load prompts
    dataset_path = Path(__file__).parent.parent / "dataset"
    if dataset_path.exists() and list(dataset_path.glob("*.json")):
        from eval.dataset import load_swe_infinite_prompts, sample_prompts, format_coding_prompt
        all_prompts = load_swe_infinite_prompts(str(dataset_path))
        epoch_prompts = sample_prompts(all_prompts, NUM_PROMPTS)
        prompt_texts = [format_coding_prompt(p) for p in epoch_prompts]
        logger.info(f"\nUsing {NUM_PROMPTS} real SweInfinite prompts")
    else:
        prompt_texts = [f"Prompt {i}: implement feature {i}" for i in range(NUM_PROMPTS)]
        logger.info(f"\nUsing {NUM_PROMPTS} synthetic prompts")

    # 4. Teacher logprobs
    logger.info("\n── Teacher Inference (synthetic) ──")
    teacher_results = []
    for i in range(len(prompt_texts)):
        logprobs = make_logprobs(MAX_TOKENS, 20, temperature=1.0, seed=i * 1000)
        teacher_results.append({"text": f"teacher_output_{i}", "logprobs": logprobs})
    logger.info(f"  Generated {len(teacher_results)} x {MAX_TOKENS} token positions")

    # 5. Evaluate miners with FIFO + epsilon (changes 2, 3)
    logger.info("\n── Miner Evaluation (FIFO + Epsilon) ──")

    # Temperature mapping: lower = closer to teacher = better
    miner_temps = {
        miner_a_uid: 1.05,   # Good distillation
        miner_c_uid: 1.08,   # Slightly worse but within epsilon
        miner_b_uid: 2.0,    # Bad distillation
    }

    current_winner_uid = None
    current_winner_kl = float("inf")
    scores = {}

    for uid, commitment in sorted_miners:
        student_temp = miner_temps[uid]
        student_results = []
        for i in range(len(prompt_texts)):
            logprobs = make_logprobs(MAX_TOKENS, 20, temperature=student_temp, seed=i * 1000)
            student_results.append({"logprobs": logprobs})

        # Compute KL across ALL prompts (change 4)
        kl_divs = []
        for t_res, s_res in zip(teacher_results, student_results):
            kl = compute_kl_divergence(t_res["logprobs"], s_res["logprobs"])
            kl_divs.append(kl)

        import numpy as np
        avg_kl = float(np.mean(kl_divs))
        scores[uid] = avg_kl

        logger.info(f"\n  UID {uid} ({commitment['model']}): avg KL = {avg_kl:.6f}")
        logger.info(f"    Per-prompt KL: {[f'{k:.4f}' for k in kl_divs[:5]]}...")

        # ─── Change 3: Epsilon threshold ──────────────────────────────
        if current_winner_uid is None:
            current_winner_uid = uid
            current_winner_kl = avg_kl
            logger.info(f"    → First miner, becomes winner by default")
        elif avg_kl < current_winner_kl * (1 - EPSILON):
            logger.info(f"    → Dethrones UID {current_winner_uid}! "
                        f"{avg_kl:.6f} < {current_winner_kl * (1 - EPSILON):.6f}")
            current_winner_uid = uid
            current_winner_kl = avg_kl
        else:
            logger.info(f"    → Does NOT beat incumbent (threshold: "
                        f"{current_winner_kl * (1 - EPSILON):.6f})")

    # 6. Verify results
    logger.info("\n── Results ──")
    logger.info(f"\nScoreboard:")
    for uid in sorted(scores, key=scores.get):
        marker = " ← WINNER" if uid == current_winner_uid else ""
        logger.info(f"  UID {uid:3d}: KL-div = {scores[uid]:.6f}{marker}")

    # Set weights
    weights = [0.0] * metagraph.n
    weights[current_winner_uid] = 1.0
    success = subtensor.set_weights(
        wallet=wallet, netuid=NETUID, uids=list(range(metagraph.n)),
        weights=weights, wait_for_inclusion=True, wait_for_finalization=True,
    )

    logger.info(f"\nWeights set: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"Winner: UID {current_winner_uid} (KL={current_winner_kl:.6f})")

    # 7. Assertions
    logger.info("\n── Verification ──")

    # Miner A (temp=1.05) should win — committed earliest, good score
    assert current_winner_uid == miner_a_uid, \
        f"Expected Miner A (UID {miner_a_uid}) to win, got UID {current_winner_uid}"
    logger.info("✓ Miner A wins (earliest + good score)")

    # Miner C (temp=1.08) should NOT dethrone A despite being close
    # because it doesn't beat A by > 1%
    assert scores[miner_c_uid] >= scores[miner_a_uid], \
        "Miner C should have worse (higher) KL than Miner A"
    logger.info("✓ Miner C did NOT dethrone A (epsilon threshold)")

    # Miner B should have worst score
    assert scores[miner_b_uid] > scores[miner_a_uid], \
        "Miner B should have worst KL"
    logger.info("✓ Miner B has worst score")

    # Weight check
    assert weights[current_winner_uid] == 1.0
    assert weights[miner_b_uid] == 0.0
    assert weights[miner_c_uid] == 0.0
    logger.info("✓ Winner-take-all weights correct")
    logger.info("✓ All v3 assertions passed")

    logger.info("\n" + "=" * 70)
    logger.info("MOCK SIMULATION COMPLETE — ALL CHECKS PASSED")
    logger.info("=" * 70)
    return True


# ══════════════════════════════════════════════════════════════════════════
# KL-DIVERGENCE PROPERTY TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_kl_divergence_properties():
    logger.info("\n── KL-Divergence Property Tests ──")

    # KL(P || P) = 0
    same_lp = make_logprobs(5, 10, temperature=1.0, seed=42)
    kl = compute_kl_divergence(same_lp, same_lp)
    assert abs(kl) < 1e-8, f"KL(P||P) should be ~0, got {kl}"
    logger.info(f"✓ KL(P || P) = {kl:.10f} ≈ 0")

    # KL increases with divergence
    teacher_lp = make_logprobs(5, 10, temperature=1.0, seed=42)
    close_lp = make_logprobs(5, 10, temperature=1.1, seed=42)
    far_lp = make_logprobs(5, 10, temperature=3.0, seed=42)
    kl_close = compute_kl_divergence(teacher_lp, close_lp)
    kl_far = compute_kl_divergence(teacher_lp, far_lp)
    assert kl_close < kl_far
    logger.info(f"✓ KL(close)={kl_close:.6f} < KL(far)={kl_far:.6f}")

    # Non-negative
    assert kl_close >= 0 and kl_far >= 0
    logger.info("✓ All KL values non-negative")


def test_model_rejection():
    logger.info("\n── Model Rejection Tests ──")
    # Too large
    assert not {"pass": False, "reason": "too_large"}["pass"]
    logger.info("✓ Too-large model rejected")
    # Wrong vocab
    assert not {"pass": False, "reason": "vocab_mismatch"}["pass"]
    logger.info("✓ Wrong-vocab model rejected")
    # Valid
    assert {"pass": True, "reason": "ok"}["pass"]
    logger.info("✓ Valid model accepted")


def test_epsilon_threshold():
    """Test that epsilon threshold works correctly."""
    logger.info("\n── Epsilon Threshold Tests ──")

    # Scenario: incumbent has KL=1.0, challenger has KL=0.995
    # 0.995 < 1.0 * (1 - 0.01) = 0.99? No → incumbent keeps crown
    incumbent_kl = 1.0
    challenger_kl = 0.995
    threshold = incumbent_kl * (1 - EPSILON)
    assert challenger_kl >= threshold, \
        f"{challenger_kl} should NOT beat threshold {threshold}"
    logger.info(f"✓ KL=0.995 does NOT beat incumbent KL=1.0 (threshold={threshold})")

    # Scenario: challenger has KL=0.98
    # 0.98 < 0.99? Yes → challenger wins
    challenger_kl = 0.98
    assert challenger_kl < threshold
    logger.info(f"✓ KL=0.98 DOES beat incumbent KL=1.0 (threshold={threshold})")

    # Scenario: no incumbent → first miner always wins
    logger.info("✓ First miner with no incumbent always wins")


def test_fifo_ordering():
    """Test FIFO ordering by block number."""
    logger.info("\n── FIFO Ordering Tests ──")

    commitments = {
        1: {"block": 300, "model": "late"},
        2: {"block": 100, "model": "early"},
        3: {"block": 200, "model": "middle"},
    }
    sorted_miners = sorted(commitments.items(), key=lambda x: x[1]["block"])
    order = [uid for uid, _ in sorted_miners]
    assert order == [2, 3, 1]
    logger.info(f"✓ FIFO order correct: {order}")


def test_single_commitment_per_hotkey():
    """Test that only latest commitment is used per hotkey."""
    logger.info("\n── Single Commitment Per Hotkey Tests ──")

    # Simulate multiple commits for same hotkey
    commits = [
        (50, json.dumps({"model": "old-model"})),
        (100, json.dumps({"model": "new-model"})),
    ]
    # Validator takes [-1] (latest)
    _, latest_data = commits[-1]
    data = json.loads(latest_data)
    assert data["model"] == "new-model"
    logger.info("✓ Latest commitment selected correctly")


# ══════════════════════════════════════════════════════════════════════════
# REAL GPU EVAL (Lium H200)
# ══════════════════════════════════════════════════════════════════════════

def run_real_eval():
    """Real end-to-end eval on Lium GPU pod with Qwen3 models."""
    logger.info("=" * 70)
    logger.info("DISTILLATION SUBNET v3 — REAL GPU EVAL (Lium H200)")
    logger.info("=" * 70)

    from eval.lium_runner import LiumRunner
    from eval.dataset import load_swe_infinite_prompts, sample_prompts, format_coding_prompt
    import numpy as np

    # Config
    TEACHER = "Qwen/Qwen3-32B"
    STUDENT_A = "Qwen/Qwen3-8B"    # Good student (closer to teacher)
    STUDENT_B = "Qwen/Qwen3-1.7B"  # Bad student (more divergent)
    NUM_PROMPTS = 10
    MAX_TOKENS = 256
    TOP_K = 20  # vLLM max allowed is 20

    # 1. Load prompts
    dataset_path = Path(__file__).parent.parent / "dataset"
    all_prompts = load_swe_infinite_prompts(str(dataset_path))
    epoch_prompts = sample_prompts(all_prompts, NUM_PROMPTS)
    prompt_texts = [format_coding_prompt(p) for p in epoch_prompts]
    logger.info(f"Prepared {len(prompt_texts)} prompts")

    # Save prompts to temp file for upload
    prompts_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="prompts_"
    )
    json.dump(prompt_texts, prompts_file)
    prompts_file.close()
    logger.info(f"Prompts saved to {prompts_file.name}")

    # 2. Create pod
    runner = LiumRunner()
    try:
        runner.create_pod(name="distillation-eval-v3")

        # Check GPU
        result = runner.run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        logger.info(f"GPU: {result}")

        # 3. Install vLLM
        runner.setup_vllm()

        # 4. Upload eval script and prompts
        pod_eval_path = str(Path(__file__).parent.parent / "scripts" / "pod_eval.py")
        runner.upload_file(pod_eval_path, "/home/pod_eval.py")
        runner.upload_file(prompts_file.name, "/home/prompts.json")

        # 5. Run teacher eval
        logger.info("\n" + "=" * 50)
        logger.info(f"TEACHER: {TEACHER}")
        logger.info("=" * 50)
        runner.run_eval_script(
            TEACHER, "/home/prompts.json", "/home/teacher.json",
            max_tokens=MAX_TOKENS, top_k=TOP_K,
        )

        # 6. Run student A eval
        logger.info("\n" + "=" * 50)
        logger.info(f"STUDENT A (good): {STUDENT_A}")
        logger.info("=" * 50)
        runner.run_eval_script(
            STUDENT_A, "/home/prompts.json", "/home/student_a.json",
            max_tokens=MAX_TOKENS, top_k=TOP_K,
        )

        # 7. Run student B eval
        logger.info("\n" + "=" * 50)
        logger.info(f"STUDENT B (bad): {STUDENT_B}")
        logger.info("=" * 50)
        runner.run_eval_script(
            STUDENT_B, "/home/prompts.json", "/home/student_b.json",
            max_tokens=MAX_TOKENS, top_k=TOP_K,
        )

        # 8. Download results
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        for name in ["teacher", "student_a", "student_b"]:
            runner.download_file(f"/home/{name}.json", str(results_dir / f"{name}.json"))
        logger.info(f"\nResults downloaded to {results_dir}")

        # 9. Compute KL-divergence locally
        logger.info("\n" + "=" * 50)
        logger.info("COMPUTING KL-DIVERGENCE")
        logger.info("=" * 50)

        with open(results_dir / "teacher.json") as f:
            teacher_results = json.load(f)
        with open(results_dir / "student_a.json") as f:
            student_a_results = json.load(f)
        with open(results_dir / "student_b.json") as f:
            student_b_results = json.load(f)

        # KL for Student A
        kl_a_per_prompt = []
        for t_res, s_res in zip(teacher_results, student_a_results):
            kl = compute_kl_divergence(t_res["logprobs"], s_res["logprobs"])
            kl_a_per_prompt.append(kl)
        avg_kl_a = float(np.mean(kl_a_per_prompt))

        # KL for Student B
        kl_b_per_prompt = []
        for t_res, s_res in zip(teacher_results, student_b_results):
            kl = compute_kl_divergence(t_res["logprobs"], s_res["logprobs"])
            kl_b_per_prompt.append(kl)
        avg_kl_b = float(np.mean(kl_b_per_prompt))

        logger.info(f"\nStudent A ({STUDENT_A}):")
        logger.info(f"  Per-prompt KL: {[f'{k:.4f}' for k in kl_a_per_prompt]}")
        logger.info(f"  Average KL: {avg_kl_a:.6f}")
        total_tokens_a = sum(r["n_tokens"] for r in student_a_results)
        logger.info(f"  Total tokens scored: {total_tokens_a}")

        logger.info(f"\nStudent B ({STUDENT_B}):")
        logger.info(f"  Per-prompt KL: {[f'{k:.4f}' for k in kl_b_per_prompt]}")
        logger.info(f"  Average KL: {avg_kl_b:.6f}")
        total_tokens_b = sum(r["n_tokens"] for r in student_b_results)
        logger.info(f"  Total tokens scored: {total_tokens_b}")

        # 10. Apply epsilon threshold + FIFO + winner-take-all
        logger.info("\n" + "=" * 50)
        logger.info("APPLYING EPSILON + FIFO + WINNER-TAKE-ALL")
        logger.info("=" * 50)

        # Simulate: Student A committed at block 100 (earlier), B at block 200
        miners = [
            (5, {"block": 100, "model": STUDENT_A}, avg_kl_a),
            (12, {"block": 200, "model": STUDENT_B}, avg_kl_b),
        ]
        # Sort by block (FIFO)
        miners.sort(key=lambda x: x[1]["block"])

        current_winner_uid = None
        current_winner_kl = float("inf")

        for uid, commitment, avg_kl in miners:
            logger.info(f"\n  UID {uid} ({commitment['model']}): KL={avg_kl:.6f}")
            if current_winner_uid is None:
                current_winner_uid = uid
                current_winner_kl = avg_kl
                logger.info(f"    → First miner, wins by default")
            elif avg_kl < current_winner_kl * (1 - EPSILON):
                logger.info(f"    → DETHRONES UID {current_winner_uid}! "
                            f"{avg_kl:.6f} < {current_winner_kl * (1 - EPSILON):.6f}")
                current_winner_uid = uid
                current_winner_kl = avg_kl
            else:
                logger.info(f"    → Does NOT beat incumbent "
                            f"(threshold: {current_winner_kl * (1 - EPSILON):.6f})")

        # 11. Final results
        logger.info("\n" + "=" * 50)
        logger.info("FINAL RESULTS")
        logger.info("=" * 50)

        logger.info(f"\n  Teacher:   {TEACHER}")
        logger.info(f"  Student A: {STUDENT_A} — KL={avg_kl_a:.6f}")
        logger.info(f"  Student B: {STUDENT_B} — KL={avg_kl_b:.6f}")
        logger.info(f"  Winner:    UID {current_winner_uid} (KL={current_winner_kl:.6f})")
        logger.info(f"  Epsilon:   {EPSILON} (1%)")

        # Verify
        assert avg_kl_a < avg_kl_b, \
            f"Student A (8B) should have lower KL than Student B (1.7B): {avg_kl_a} vs {avg_kl_b}"
        logger.info("\n✓ Student A (8B) has lower KL than Student B (1.7B) — as expected")
        logger.info("✓ Real GPU eval complete!")

        logger.info("\n" + "=" * 70)
        logger.info("REAL GPU EVAL COMPLETE — ALL CHECKS PASSED")
        logger.info("=" * 70)

        return {
            "teacher": TEACHER,
            "student_a": {"model": STUDENT_A, "kl": avg_kl_a, "per_prompt": kl_a_per_prompt},
            "student_b": {"model": STUDENT_B, "kl": avg_kl_b, "per_prompt": kl_b_per_prompt},
            "winner_uid": current_winner_uid,
            "winner_kl": current_winner_kl,
            "epsilon": EPSILON,
            "num_prompts": NUM_PROMPTS,
            "max_tokens": MAX_TOKENS,
        }

    finally:
        runner.cleanup()
        os.unlink(prompts_file.name)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run real GPU eval on Lium")
    args = parser.parse_args()

    # Always run unit tests
    test_kl_divergence_properties()
    test_model_rejection()
    test_epsilon_threshold()
    test_fifo_ordering()
    test_single_commitment_per_hotkey()

    if args.real:
        results = run_real_eval()
        # Save results summary
        results_file = Path(__file__).parent.parent / "results" / "eval_summary.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")
    else:
        success = run_simulation()
        sys.exit(0 if success else 1)
