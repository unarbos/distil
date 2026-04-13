# Concurrent vLLM for Student Scoring — Design Document

**Author:** Arbos  
**Date:** 2026-04-13  
**Status:** DESIGN ONLY — not yet implemented

## Problem

Phase 3 (student scoring) currently loads each student model via HuggingFace
`AutoModelForCausalLM`, runs a forward pass on each prompt, then unloads.
This is slow because:
- HF model loading is unoptimized (no tensor parallelism, no continuous batching)
- Forward passes are sequential per-prompt
- No KV cache reuse between prompts

## Proposed Solution

Replace HF forward passes with vLLM prompt-mode inference for student scoring.
Use vLLM's `/v1/completions` endpoint with `max_tokens=0` and `prompt_logprobs=True`
to extract logprobs at teacher-generated token positions without any autoregressive
generation.

## Architecture

```
Current (Phase 3):
┌──────────────────────────────────────────────────────┐
│  For each student:                                    │
│    1. HF load_model(student) → ~8-10GB VRAM          │
│    2. For each prompt (300x):                         │
│       student(full_sequence).logits → KL vs teacher   │
│    3. del student; free_gpu()                         │
│  Time: ~2-3 min/student × N students                  │
└──────────────────────────────────────────────────────┘

Proposed (Phase 3 with vLLM):
┌──────────────────────────────────────────────────────┐
│  For each student:                                    │
│    1. start_vllm_server(student, gpu_util=0.15)       │
│       → ~10-12GB VRAM (model + vLLM overhead)         │
│    2. POST /v1/completions (batched):                 │
│       {                                               │
│         "model": "student",                           │
│         "prompt": <full_sequence_token_ids>,          │
│         "max_tokens": 0,                              │
│         "prompt_logprobs": 128,                       │
│         "temperature": 1.0                            │
│       }                                               │
│    3. Extract logprobs at continuation positions      │
│    4. Compute KL vs teacher sparse logprobs           │
│    5. stop_vllm_server()                              │
│  Time: ~30-60s/student (estimated)                    │
└──────────────────────────────────────────────────────┘
```

## Key Insight: Prompt Mode (No Generation)

vLLM supports `prompt_logprobs` — when you send a prompt, vLLM returns the
log-probability of each token in the prompt (conditioned on all preceding tokens).
By setting `max_tokens=0`, vLLM does a single forward pass over the entire
sequence without generating any new tokens.

This is functionally identical to `model(input_ids).logits` but benefits from:
- vLLM's optimized attention kernels (PagedAttention)
- Continuous batching (multiple prompts processed concurrently)
- Better GPU utilization

## API Calls

### Start vLLM for student

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model <student_name> \
    --revision <revision> \
    --port 8100 \                          # Different port from teacher (8000)
    --gpu-memory-utilization 0.15 \        # ~28GB on B200 (student is only ~10GB)
    --max-model-len 16384 \
    --disable-log-requests \
    --dtype auto
```

### Score prompts (batched)

```python
import requests

# Batch all 300 prompts into a single request (or chunk into batches of 50)
responses = []
for batch in chunk(full_sequences, batch_size=50):
    resp = requests.post("http://localhost:8100/v1/completions", json={
        "model": student_name,
        "prompt": [seq.tolist() for seq in batch],  # token ID lists
        "max_tokens": 0,
        "prompt_logprobs": 128,   # top-128 logprobs per position
        "temperature": 1.0,
    })
    responses.append(resp.json())
```

### Extract KL from prompt_logprobs

```python
for choice in response["choices"]:
    prompt_logprobs = choice["prompt_logprobs"]
    # prompt_logprobs[i] = {token_str: logprob, ...} for position i
    # We only care about positions [prompt_len-1 : end-1] (continuation)
    # These correspond to teacher's generated tokens
    
    for pos in range(prompt_len, len(prompt_logprobs)):
        student_top_k = prompt_logprobs[pos]  # dict of token→logprob
        teacher_top_k = teacher_logits_list[prompt_idx]  # sparse indices+values
        kl = compute_kl_from_sparse(teacher_top_k, student_top_k)
```

## VRAM Budget (B200 = 192GB)

| Component | Current VRAM | Proposed VRAM |
|-----------|-------------|---------------|
| Teacher logits (CPU) | ~2GB CPU RAM | ~2GB CPU RAM |
| King model (HF, stays loaded) | ~8GB | 0 (not kept in VRAM)* |
| Student model (HF) | ~8-10GB | — |
| Student vLLM server | — | ~12-15GB |
| vLLM overhead (PagedAttention, KV cache) | — | ~5-10GB |
| **Total GPU** | **~18GB** | **~15-25GB** |

*Note: With vLLM approach, king doesn't stay in VRAM between students.
Each model (including king) gets its own vLLM lifecycle. This is a tradeoff —
we lose the "king stays loaded" optimization but gain vLLM's faster inference.

**Alternative:** Keep king in HF while running challenger via vLLM:
- King HF: ~8GB + Student vLLM: ~15GB = ~23GB total
- Adds complexity but preserves the king-stays-loaded optimization
- Recommendation: Start without this, add later if king reload time is significant

## Estimated Speedup

| Phase | Current | Proposed | Speedup |
|-------|---------|----------|---------|
| Student load | ~15-30s (HF download+load) | ~20-40s (vLLM startup) | ~0.7x (slower) |
| 300 prompt forward passes | ~90-120s | ~20-40s (batched) | **3-5x** |
| Student unload | ~2s (del + gc) | ~3s (kill vLLM) | ~0.7x |
| **Per-student total** | **~120-150s** | **~50-80s** | **~2x** |
| **5 students total** | **~10-12 min** | **~5-7 min** | **~2x** |

The main win is batched inference — vLLM processes multiple prompts concurrently
with continuous batching, while HF processes them sequentially.

Early stopping reduces actual prompts scored (often 7-50 instead of 300 for
non-competitive models), so the speedup is most significant for competitive
models that need all 300 prompts.

## Risks and Edge Cases

### 1. vLLM startup overhead
- vLLM takes 15-40s to start per model (loading weights, compiling kernels)
- For models that early-stop at 7 prompts, vLLM startup may be slower than HF
- **Mitigation:** Fall back to HF for models expected to early-stop (if previous round data exists)

### 2. prompt_logprobs token mapping
- vLLM returns logprobs as `{token_string: logprob}` not `{token_id: logprob}`
- Need `token_to_id` mapping (already implemented for teacher in `vllm_logprobs_to_sparse()`)
- **Risk:** Special tokens, whitespace tokens may have ambiguous string representations
- **Mitigation:** Reuse existing `_build_token_to_id_map()` function; all students share Qwen tokenizer

### 3. Numerical differences
- vLLM uses different attention implementations (PagedAttention vs HF's FlashAttention)
- KL scores may differ slightly between vLLM and HF forward passes
- **Mitigation:** Run A/B comparison on 5 models, verify KL differences < 1e-4
- **Critical:** This changes scoring behavior — must validate that rankings are preserved

### 4. vLLM port conflicts
- Teacher vLLM runs on port 8000, student would need different port (8100)
- But teacher vLLM is killed before Phase 3, so port 8000 is also available
- **Decision:** Use port 8000 for simplicity (same as teacher)

### 5. VRAM fragmentation
- Repeated vLLM start/stop may fragment GPU memory
- **Mitigation:** Run `torch.cuda.empty_cache()` and `gc.collect()` between models
- vLLM manages its own memory pool, should be clean on shutdown

### 6. Model compatibility
- vLLM must support `Qwen3_5ForConditionalGeneration` architecture
- Currently works for teacher (Qwen3.5-35B-A3B) — students use same arch
- **Risk:** Custom student model configs that vLLM doesn't handle
- **Mitigation:** Fall back to HF on vLLM load failure (same pattern as teacher fallback)

### 7. prompt_logprobs batch size limits
- Very long sequences (16K tokens) × 300 prompts × 128 logprobs = large response
- **Mitigation:** Batch in chunks of 50 prompts, stream responses

### 8. King model optimization lost
- Current: king stays in VRAM across all rounds, saving ~15-30s per round
- Proposed: king gets its own vLLM lifecycle like any other student
- **Impact:** ~20-40s extra per round for king vLLM startup
- **Mitigation:** Accept the tradeoff; the per-prompt speedup compensates

## Implementation Plan

### Step 1: Validate prompt_logprobs accuracy (1-2 hours)
- Write a test script that scores one student model both ways (HF and vLLM prompt mode)
- Compare KL values across all 300 prompts
- Accept if max absolute difference < 1e-4

### Step 2: Implement `score_student_via_vllm()` function (2-3 hours)
- New function alongside existing HF scoring loop
- Handles: vLLM startup, prompt batching, logprob extraction, KL computation
- Reuses `vllm_logprobs_to_sparse()` for token mapping
- Includes timeout and fallback to HF on failure

### Step 3: Integrate into Phase 3 loop (1-2 hours)
- Replace the per-student HF forward pass with `score_student_via_vllm()`
- Preserve: early stopping, live progress updates, incremental result saving
- Preserve: VRAM fraud check (check model size before scoring)
- Preserve: activation fingerprinting (may need separate HF load or skip)

### Step 4: Handle activation fingerprinting (1 hour)
- `compute_activation_fingerprint()` needs direct model access (HF)
- Options:
  a. Load HF model briefly just for fingerprint, then unload and use vLLM for scoring
  b. Compute fingerprint via vLLM hidden states (if exposed)
  c. Skip fingerprint for vLLM path (not recommended — needed for copy detection)
- **Recommendation:** Option (a) — fingerprint is fast (~1-2s), load/unload overhead is acceptable

### Step 5: A/B testing on prod (1 round)
- Run one full eval round with `--use-vllm-scoring` flag (off by default)
- Compare results against HF scoring from previous round
- Verify: rankings preserved, KL values within tolerance, no crashes

### Step 6: Default enable + cleanup (30 min)
- Make vLLM scoring the default
- Keep `--no-vllm-scoring` flag for fallback
- Update documentation and progress reporting

## Open Questions

1. **Should king keep the "stays in VRAM" optimization?**
   If so, king scores via HF (fast, already loaded) while challengers score via vLLM.
   Adds code complexity but saves ~30s per round.

2. **Can we batch ALL students into one vLLM instance?**
   Not easily — vLLM serves one model at a time. Would need model swapping
   or multiple vLLM instances (VRAM constraints make this impractical for >1 student).

3. **Is the activation fingerprint worth the HF load overhead?**
   If fingerprinting is critical for copy detection, the HF load is unavoidable.
   Consider caching fingerprints across rounds for known models.
