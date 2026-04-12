# GPU Splitting Bug Report — Lium RTX 4090

**Date**: 2026-04-12 ~02:00-02:30 UTC  
**Reporter**: Arbos (SN97 Distil validator team)  
**Executor**: `zesty-eagle-77` (2×RTX4090)  
**Pod**: `zesty-eagle-e9`  
**GPU Split Config**: `gpu_count: 1` (rented 1 of 2 GPUs)  

## Summary

GPU splitting on a 2×RTX4090 executor results in a CUDA misaligned address error during vLLM inference, crashing the server after the first request. The same model + vLLM version works flawlessly on a standard (non-split) 1×RTX4090 pod.

## Steps to Reproduce

1. Rent a GPU-split pod via REST API:
   ```python
   payload = {
       "pod_name": "chat-bench",
       "template_id": "<pytorch template>",
       "user_public_key": "<ssh_key>",
       "gpu_count": 1,  # Split: only rent 1 of 2 GPUs
   }
   resp = client._request("POST", f"/executors/zesty-eagle-77/rent", json=payload)
   ```

2. Pod comes up, `nvidia-smi` shows 1×RTX4090 (GPU-0), 24GB VRAM — looks correct.

3. Install vLLM 0.19.0 + serve a 4B parameter model:
   ```bash
   pip install vllm
   python3 -m vllm.entrypoints.openai.api_server \
     --model /root/king-model \
     --port 8100 --host 0.0.0.0 \
     --dtype bfloat16 --max-model-len 32768 \
     --trust-remote-code --enforce-eager \
     --gpu-memory-utilization 0.90
   ```

4. Model loads successfully. Encoder cache profiles fine. Server starts responding to `/v1/models`.

5. Send a single chat completion request → **crashes** with:
   ```
   torch.AcceleratorError: CUDA error: misaligned address
   ```

6. vLLM EngineCore dies, server shuts down. Cannot recover without full restart, which also crashes on first request.

## Error Log (verbatim)

```
(EngineCore pid=1309) torch.AcceleratorError: CUDA error: misaligned address
(EngineCore pid=1309) Search for `cudaErrorMisalignedAddress' in 
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
(EngineCore pid=1309) CUDA kernel errors might be asynchronously reported at some 
other API call, so the stacktrace below might be incorrect.

(APIServer pid=757) vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered 
an issue. See stack trace (above) for the root cause.
```

Stack trace points to:
- `gpu_model_runner.py:3485` → `self.prepare_inputs_event.synchronize()`
- `uniproc_executor.py:84` → `collective_rpc`

## Attempted second start

After the crash, restarting vLLM on the same pod immediately fails:
```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

GPU shows 0% utilization and 280 MiB VRAM (no model loaded), but the engine refuses to start.

## Working configuration (control)

The exact same model, vLLM version, and flags work perfectly on:
- `cosmic-lion-e2` (1×RTX4090, standard non-split pod, `golden-fox-f8` executor)
- `brave-comet-f0` (1×RTX4090, standard non-split pod)
- `calm-wolf-ca` (1×H200, standard pod)

## Environment

- **vLLM**: 0.19.0
- **torch**: 2.10.0+cu128
- **Model**: ncaagcc/sn97-q8rn (4.38B params, Qwen3.5 architecture, bfloat16)
- **CUDA**: 12.8
- **GPU**: RTX 4090 (24GB)
- **Pod template**: PyTorch (Custom)
- **OS in container**: Ubuntu 24.04.2 LTS

## Hypothesis

The `cudaErrorMisalignedAddress` error suggests the GPU memory addressing may be affected by the GPU splitting mechanism. Possible causes:
1. **MIG partitioning issues**: If the executor uses MIG or similar GPU partitioning, memory alignment constraints may differ from a full GPU.
2. **CUDA context isolation**: GPU splitting may use cgroups/namespace-level GPU isolation that affects memory mapping alignment.
3. **Driver-level memory mapping**: The split GPU's memory pages may not be aligned to the boundaries vLLM's custom CUDA kernels expect.

## Impact

GPU splitting would save ~30% on cost ($0.14/hr vs $0.20/hr for RTX 4090), but is currently unusable for vLLM inference workloads.

## Workaround

Use standard (non-split) 1×RTX4090 pods instead.
