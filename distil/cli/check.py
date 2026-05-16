"""``distil check`` — validate a HuggingFace model against the SN97 precheck."""

from __future__ import annotations


def run(model_repo: str, revision: str = "") -> int:
    from distil.eval.precheck import precheck

    print(f"Checking {model_repo}@{revision or 'latest'} …")
    result = precheck(model_repo, revision)
    if result.ok:
        meta = result.metadata or {}
        print("OK")
        if cfg := meta.get("config"):
            print(f"  arch         : {cfg.get('architectures')}")
        if meta.get("total_params"):
            print(f"  total_params : {meta['total_params']:,}")
        if meta.get("active_params"):
            print(f"  active_params: {meta['active_params']:,}")
        print(f"  is_moe       : {meta.get('is_moe', False)}")
        if meta.get("hf_sha"):
            print(f"  hf_sha       : {meta['hf_sha']}")
        return 0
    print(f"FAIL — {result.reason}")
    return 1
