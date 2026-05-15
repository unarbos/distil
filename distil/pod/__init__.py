"""GPU-pod runner — uploaded to a Lium B200 pod each round.

Three phases per round:

* **Phase 1** — start a vLLM teacher engine, sample block-seeded prompts,
  greedy-decode continuations, save sparse top-K logprobs +
  ``prompt_logprobs`` outputs to ``teacher_cache/round_<round>.json``
  (improvement #2).
* **Phase 2** — for each axis runner under :mod:`distil.pod.axes`, generate
  fresh procedural items and grade them. Records per-bench wall-time and
  tokens/sec (improvement #5).
* **Phase 3** — for each student model, start a fresh vLLM engine with the
  warm-up generate (improvement #3) and score on the cached teacher outputs
  using ``prompt_logprobs`` (improvement #1). Run all probes
  (judge / long_form / chat / activation_fp / reasoning_density). Save the
  per-student JSON to ``results.json``.
"""
