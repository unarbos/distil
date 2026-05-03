#!/usr/bin/env python3
"""
Remote-teacher distillation shim (HTTP)
======================================

**Feasibility** — Training a student while the **teacher stays on another host** serving Kimi‑K2.6 via
your own inference stack (**vLLM**, **SGLang**, Ray, …) **is workable**, but raw OpenAI
``/v1/chat/completions`` APIs do **not** expose arbitrary per‑position teacher logits across the prompt.
You expose a tiny **teacher sidecar HTTP route** next to your engine.

This module documents a practical contract and ships:

1. ``TeacherTopKGatherClient`` — training‑node HTTP client fetching **top‑K logits + log‑sum‑exp**.
2. ``forward_kl_topk_gather`` — forward KL \\( \\mathbb E_{\\text{teacher \\, top‑}K}[ \\log q_S ] \\) style
   term that matches distilled objectives when \\(K\\) is large (**tail mass neglected** unless you widen K).
3. ``logits_to_wire_dict`` + ``build_example_fastapi_teacher_app`` — encoder + **reference FastAPI** app wired
   to HF ``AutoModelForCausalLM``. Replace ``model.forward`` internals with calls into your **vLLM / SGLang**
   Python bindings while keeping **the JSON wire unchanged**.

Recommended wire (**POST /teacher/topk_logits_gather**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Request JSON::

    {
      "top_k": 512,
      "assume_causal_lm": true,
      "sequences": [
        {"input_ids_row": [...], "attention_mask_row": [...] },
        ...
      ],
      "kd_start_positions": [128, …]            # informational (logging / masking hints)
    }

Response blobs (numpy little‑endian):

* ``gathered_indices_b64`` + meta ``int32 [B,Lt,K]``
* ``gathered_logits_b64`` + meta ``float16 [B,Lt,K]`` — raw logits for those indices (**before softmax**).
* ``position_logsumexp_b64`` + meta ``float32 [B,Lt]`` — \\( \\log\\sum_v \\exp(\\text{logits}_{t,v}) \\) over **full** vocab rows.
  True teacher log‑prob gather: \\( \\text{gathered_logits} − \\text{position_logsumexp} \\).
* Optional ``valid_mask_b64`` ``uint8 [B,Lt]``.

The server trims each row to ``sum(attention_mask)`` tokens before padding internally. Logits slices are taken
after the usual HF causal forward ``logits[..., :-1, :]``.

Examples::

    # Reference server on teacher box (GPT‑2 sanity check; substitute Kimi + vLLM hook)
    python examples/remote_teacher_kd.py serve --model gpt2 --port 8787

    # Local smoke‑test without HTTP
    python examples/remote_teacher_kd.py dry_run_roundtrip
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "TeacherTopKGatherClient",
    "TeacherTopKBatched",
    "TeacherTopKBatched_decode",
    "forward_kl_topk_gather",
    "logits_to_wire_dict",
    "build_example_fastapi_teacher_app",
]


def _b64_tensor(buf: bytes, *, dtype: str, shape: Sequence[int]) -> torch.Tensor:
    dt = getattr(np, dtype)
    arr = np.frombuffer(buf, dtype=dt).reshape(tuple(shape)).copy()
    return torch.from_numpy(arr)


def _tensor_b16(x: torch.Tensor) -> tuple[str, dict[str, Any]]:
    blob = x.detach().cpu().to(torch.float16).numpy().tobytes(order="C")
    return base64.standard_b64encode(blob).decode("ascii"), {
        "dtype": "float16",
        "shape": list(x.shape),
    }


def _tensor_f32(x: torch.Tensor) -> tuple[str, dict[str, Any]]:
    blob = x.detach().cpu().to(torch.float32).numpy().tobytes(order="C")
    return base64.standard_b64encode(blob).decode("ascii"), {
        "dtype": "float32",
        "shape": list(x.shape),
    }


def _tensor_i32(x: torch.Tensor) -> tuple[str, dict[str, Any]]:
    blob = x.detach().cpu().to(torch.int32).numpy().tobytes(order="C")
    return base64.standard_b64encode(blob).decode("ascii"), {
        "dtype": "int32",
        "shape": list(x.shape),
    }


def _tensor_u8_mask(x: torch.Tensor) -> tuple[str, dict[str, Any]]:
    blob = x.detach().cpu().numpy().astype(np.uint8).tobytes(order="C")
    return base64.standard_b64encode(blob).decode("ascii"), {
        "dtype": "uint8",
        "shape": list(x.shape),
    }


@dataclass
class TeacherTopKBatched:
    gathered_indices: torch.Tensor  # int64/long [B, Lt, K] CPU
    gathered_logits: torch.Tensor  # float32 [B, Lt, K] CPU
    position_logsumexp: torch.Tensor  # float32 [B, Lt] CPU
    valid_mask: torch.Tensor | None  # bool [B, Lt]


def TeacherTopKBatched_decode(resp: dict[str, Any]) -> TeacherTopKBatched:
    raw_ix = base64.standard_b64decode(resp["gathered_indices_b64"].encode("ascii"))
    ix_meta = resp["gathered_indices_meta"]
    gi = _b64_tensor(raw_ix, dtype=ix_meta["dtype"], shape=ix_meta["shape"]).long()

    raw_g = base64.standard_b64decode(resp["gathered_logits_b64"].encode("ascii"))
    g_meta = resp["gathered_logits_meta"]
    gl = _b64_tensor(raw_g, dtype=g_meta["dtype"], shape=g_meta["shape"]).float()

    raw_lse = base64.standard_b64decode(resp["position_logsumexp_b64"].encode("ascii"))
    l_meta = resp["position_logsumexp_meta"]
    plse = _b64_tensor(raw_lse, dtype=l_meta["dtype"], shape=l_meta["shape"]).float()

    if resp.get("valid_mask_b64") and resp.get("valid_mask_meta"):
        raw_m = base64.standard_b64decode(resp["valid_mask_b64"].encode("ascii"))
        m_meta = resp["valid_mask_meta"]
        m_u8 = _b64_tensor(raw_m, dtype=m_meta["dtype"], shape=m_meta["shape"])
        mb = m_u8.to(torch.bool)
        if mb.shape[-1] != plse.shape[-1]:
            raise ValueError(f"mask width {mb.shape[-1]} ≠ logits axis {plse.shape[-1]}")
    else:
        mb = torch.ones(plse.shape, dtype=torch.bool, device=plse.device)

    return TeacherTopKBatched(gi, gl, plse, mb)


class TeacherTopKGatherClient:
    """Trainer-side HTTP shim for POST ``/teacher/topk_logits_gather``."""

    def __init__(
        self,
        *,
        url: str,
        timeout_sec: float = 240.0,
        top_k: int = 512,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.top_k = int(top_k)
        self._sess = session or requests.Session()

    def fetch_for_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kl_start_positions: Sequence[int],
    ) -> TeacherTopKBatched:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        toks_cpu = input_ids.detach().cpu().long()
        mask_cpu = attention_mask.detach().cpu().long() if attention_mask is not None else None
        seq_lens: list[int] = []
        Lpad = int(toks_cpu.shape[1])
        for b in range(toks_cpu.shape[0]):
            seq_lens.append(int(mask_cpu[b].sum().item()) if mask_cpu is not None else Lpad)
        sequences: list[dict[str, Any]] = [{"input_ids_row": toks_cpu[b, : sl].tolist()} for b, sl in enumerate(seq_lens)]
        if mask_cpu is not None:
            for b, sl in enumerate(seq_lens):
                sequences[b]["attention_mask_row"] = mask_cpu[b, :sl].tolist()
        payload = {
            "top_k": self.top_k,
            "assume_causal_lm": True,
            "sequences": sequences,
            "kd_start_positions": [int(x) for x in kl_start_positions],
        }
        rsp = self._sess.post(f"{self.url}/teacher/topk_logits_gather", json=payload, timeout=self.timeout_sec)
        rsp.raise_for_status()
        return TeacherTopKBatched_decode(rsp.json())

    def health(self, suffix: str = "/healthz") -> bool:
        uri = self.url.rstrip("/") + (suffix if suffix.startswith("/") else f"/{suffix}")
        try:
            r = self._sess.get(uri, timeout=min(15.0, self.timeout_sec))
            return r.status_code == 200
        except Exception:
            return False


def logits_to_wire_dict(
    logits_no_last_timestep: torch.Tensor,
    valid_row_mask: torch.Tensor | None,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Turn teacher logits **[B,Lt,V]** slices (already ``[:, :-1, :]``) into JSON payloads."""
    if logits_no_last_timestep.ndim != 3:
        raise ValueError("logits tensor must be [B,Lt,V]")
    Vs = logits_no_last_timestep.shape[-1]
    kk = max(1, min(int(top_k), Vs))
    l32 = logits_no_last_timestep.float()
    lse = torch.logsumexp(l32, dim=-1)
    vals, ix = torch.topk(l32, kk, dim=-1)
    if valid_row_mask is None:
        valid_row_mask = torch.ones(lse.shape, dtype=torch.bool, device=l32.device)

    out: dict[str, Any] = {}
    blob, meta = _tensor_i32(ix.to(torch.int32).cpu())
    out["gathered_indices_b64"], out["gathered_indices_meta"] = blob, meta
    blob, meta = _tensor_b16(vals.cpu())
    out["gathered_logits_b64"], out["gathered_logits_meta"] = blob, meta
    blob, meta = _tensor_f32(lse.cpu())
    out["position_logsumexp_b64"], out["position_logsumexp_meta"] = blob, meta
    blob, meta = _tensor_u8_mask(valid_row_mask.to(torch.uint8).cpu())
    out["valid_mask_b64"], out["valid_mask_meta"] = blob, meta
    return out


def forward_kl_topk_gather(
    *,
    student: nn.Module,
    teacher_pack: TeacherTopKBatched,
    input_ids_student: torch.Tensor,
    attention_mask_student: torch.Tensor | None,
    kd_loss_mask_logits: torch.Tensor,
    kd_start_pos: int,
    temperature_student: float = 1.0,
    temperature_teacher: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Masked forward‑KL surrogate on gathered teacher masses.

    For each timestep ``t`` and gathered index ``v``::

        contrib = \(\exp (\log p_T(v)) \, (\log p_T(v) - \log q_S(v))\)

    summed over gathered ``v`` **and truncated tail outside top‑K** is omitted.
    Multiply by ``kd_loss_mask_logits[..., t]``.
    """
    sdev = input_ids_student.device
    logits_u = student(
        input_ids_student,
        attention_mask=attention_mask_student,
    ).logits.float() / float(temperature_student)

    Lt = int(teacher_pack.gathered_logits.shape[1])
    if logits_u.shape[1] < Lt:
        raise ValueError(
            f"Student seq shorter than teacher logits ({logits_u.shape[1]} < {Lt}); "
            "truncate teacher or lengthen student IDs."
        )
    lg = logits_u[:, :Lt, :]

    Vs = lg.shape[-1]
    ks = teacher_pack.gathered_indices.to(sdev, dtype=torch.long).clamp(0, Vs - 1)

    s_lp_full = F.log_softmax(lg, dim=-1)
    gathered_s_lp = torch.gather(s_lp_full, dim=-1, index=ks)

    tg = teacher_pack.gathered_logits.to(sdev, dtype=torch.float32) / float(temperature_teacher)
    plse = teacher_pack.position_logsumexp.to(sdev, dtype=torch.float32).unsqueeze(-1) / float(temperature_teacher)
    lp_t = torch.clamp(tg - plse, min=-120.0, max=2.0)
    p_t = torch.exp(lp_t)

    kd_m = kd_loss_mask_logits.to(sdev, dtype=torch.float32)[:, :Lt].clone()
    if teacher_pack.valid_mask is not None:
        kd_m = kd_m * teacher_pack.valid_mask.to(sdev, dtype=torch.float32)[:, :Lt]
    kd_m[:, : max(0, int(kd_start_pos))] = 0.0

    per_elem = kd_m.unsqueeze(-1) * p_t * (lp_t - gathered_s_lp)
    denom = kd_m.sum().clamp(min=1e-6)
    return per_elem.sum() / denom, {"tail_mass_approx_ignored_note": True}


def build_example_fastapi_teacher_app(model: nn.Module, tokenizer):
    """FastAPI shim—wire ``model()`` to HF today, swap for vLLM ``LLM()`` forward later."""
    try:
        from fastapi import FastAPI
    except ImportError as e:  # pragma: no cover
        raise ImportError("`fastapi` required for HTTP teacher server") from e

    tok = tokenizer
    mdl = model
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = FastAPI(title="Teacher top-K logits relay")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.post("/teacher/topk_logits_gather")
    def route_gather(payload: dict[str, Any]) -> dict[str, Any]:
        top_k = max(1, int(payload.get("top_k") or 512))
        specs = payload.get("sequences") or []
        pad_id = int(getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None) or 0)

        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        max_len_ids = 0
        for spec in specs:
            ids_li = list(spec.get("input_ids_row") or [])
            am_li = spec.get("attention_mask_row")
            ids_t = torch.tensor(ids_li, dtype=torch.long)
            if am_li is not None:
                m = torch.tensor(am_li, dtype=torch.bool)
                keep = int(m.sum().item())
                ids_t = ids_t[:keep]
            rows.append(ids_t)
            masks.append(torch.ones_like(ids_t, dtype=torch.bool))
            max_len_ids = max(max_len_ids, int(ids_t.shape[0]))

        if not rows:
            raise ValueError("empty batch")

        B = len(rows)
        inp_ids = torch.full((B, max_len_ids), pad_id, dtype=torch.long, device=dev)
        attn_pad = torch.zeros((B, max_len_ids), dtype=torch.long, device=dev)
        for bi, rr in enumerate(rows):
            ln = rr.shape[0]
            inp_ids[bi, :ln].copy_(rr.to(device=dev))
            attn_pad[bi, :ln] = 1

        mdl.eval()
        with torch.no_grad():
            out = mdl(input_ids=inp_ids, attention_mask=attn_pad)
            logits = getattr(out, "logits", out).float()[:, :-1, :]

        Lt_local = logits.shape[1]
        pad_valid = torch.zeros((B, Lt_local), dtype=torch.bool, device=logits.device)
        for bi, rr in enumerate(rows):
            sl = rr.shape[0]
            pad_valid[bi, : max(sl - 1, 0)] = True if sl >= 2 else False

        return logits_to_wire_dict(logits.detach().cpu(), pad_valid.cpu(), top_k=top_k)

    return app


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


class _SmokeStudent(nn.Module):
    def __init__(self, template: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tpl", template.clone(), persistent=False)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.tpl)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("serve", help="Minimal HF-backed teacher relay (prototype)")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--model", default="gpt2", help="HF repo id locally; Kimi swaps this for your engine hook")
    sp.add_argument("--revision", default=None)

    dq = sub.add_parser("dry_run_roundtrip")
    dq.add_argument("--B", type=int, default=2)
    dq.add_argument("--Lt", type=int, default=6)
    dq.add_argument("--Vs", type=int, default=64)
    dq.add_argument("--top_k", type=int, default=16)

    args = p.parse_args(argv)

    if args.cmd == "dry_run_roundtrip":
        _configure_logging()
        B = int(args.B)
        Lt = int(args.Lt)
        Vs = int(args.Vs)
        kk = max(1, min(int(args.top_k), Vs))
        torch.manual_seed(1)
        ref_logits = torch.randn(B, Lt, Vs)
        valid = torch.ones(B, Lt, dtype=torch.bool)
        wire_json = logits_to_wire_dict(ref_logits, valid, top_k=kk)
        pack = TeacherTopKBatched_decode(wire_json)
        pupil = _SmokeStudent(ref_logits).to(torch.device("cpu"))
        ids = torch.randint(0, min(Vs - 1, 1023), (B, Lt))
        kd_mask = torch.ones(B, Lt, dtype=torch.float32)
        loss, dbg = forward_kl_topk_gather(
            student=pupil,
            teacher_pack=pack,
            input_ids_student=ids,
            attention_mask_student=torch.ones_like(ids),
            kd_loss_mask_logits=kd_mask,
            kd_start_pos=0,
        )
        loss_val = float(loss.detach().cpu().item())

        pupil2 = _SmokeStudent(torch.zeros(B, Lt, Vs)).eval()
        loss2, _ = forward_kl_topk_gather(
            student=pupil2,
            teacher_pack=pack,
            input_ids_student=ids,
            attention_mask_student=torch.ones_like(ids),
            kd_loss_mask_logits=kd_mask,
            kd_start_pos=0,
        )

        print("kl_same_template", round(loss_val, 8))
        print("kl_zero_student", round(float(loss2.item()), 8))
        print("diag", dbg)
        return

    if args.cmd == "serve":
        _configure_logging()
        import uvicorn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_u = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        if tok_u.pad_token is None:
            tok_u.pad_token = tok_u.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        mfu = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            mfu = mfu.to(torch.device("cpu"))
        uvicorn.run(build_example_fastapi_teacher_app(mfu, tok_u), host=args.host, port=int(args.port))

        return

    p.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
