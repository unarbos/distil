"""``distil miner`` — submit your model commitment on chain."""

from __future__ import annotations


def run(
    model_repo: str,
    revision: str = "",
    wallet_name: str | None = None,
    hotkey_name: str | None = None,
    *,
    dry_run: bool = False,
) -> int:
    from distil.chain.commitments import commit_self
    from distil.eval.precheck import precheck
    from distil.settings import settings

    if wallet_name:
        settings.wallet_name = wallet_name
    if hotkey_name:
        settings.hotkey_name = hotkey_name

    print(f"Pre-checking {model_repo}@{revision or 'latest'} …")
    pre = precheck(model_repo, revision)
    if not pre.ok:
        print(f"FAIL — would be DQ'd at validator: {pre.reason}")
        return 1
    meta = pre.metadata or {}
    if meta.get("total_params"):
        print(f"  total_params : {meta['total_params']:,}")
    if meta.get("active_params"):
        print(f"  active_params: {meta['active_params']:,}")
    print(f"  is_moe       : {meta.get('is_moe', False)}")
    print(f"  hf_sha       : {meta.get('hf_sha')}")

    if dry_run:
        print("DRY RUN — not submitting on-chain.")
        return 0

    import bittensor as bt

    wallet = bt.wallet(name=settings.wallet_name, hotkey=settings.hotkey_name)
    print(f"Wallet hotkey: {wallet.hotkey.ss58_address}")
    ok, block, msg = commit_self(wallet, model_repo, revision)
    print(msg)
    if ok:
        print(f"Committed at block {block}.")
    return 0 if ok else 1
