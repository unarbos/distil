#!/opt/distil/venv/bin/python
"""Post a tweet (or thread reply) via the X v2 API as @arbos_born.

Usage:
    post_tweet.py --text "tweet content"
    post_tweet.py --text "reply content" --reply-to <tweet_id>
    post_tweet.py --thread-file thread.json   # post a sequence

Reads OAuth1 credentials from /root/.distil-secrets/x.env. Saves the
posted tweet IDs (and any reply chain) to /opt/distil/repo/state/
posted_tweets.jsonl as one JSON object per line.

Scheduled posts work via the system ``at`` command — see
``schedule_tweet.sh`` (sibling).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ENV_PATH = Path("/root/.distil-secrets/x.env")
LOG_PATH = Path("/opt/distil/repo/state/posted_tweets.jsonl")


def _load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def post_one(text: str, reply_to: str | None = None) -> dict:
    import tweepy

    env = _load_env()
    client = tweepy.Client(
        consumer_key=env["X_CONSUMER_KEY"],
        consumer_secret=env["X_CONSUMER_SECRET"],
        access_token=env["X_ACCESS_TOKEN"],
        access_token_secret=env["X_ACCESS_SECRET"],
    )
    kwargs: dict = {"text": text}
    if reply_to:
        kwargs["in_reply_to_tweet_id"] = reply_to
    resp = client.create_tweet(**kwargs)
    tweet_id = (resp.data or {}).get("id")
    record = {
        "tweet_id": tweet_id,
        "text": text,
        "reply_to": reply_to,
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Posted: id={tweet_id}  url=https://x.com/arbos_born/status/{tweet_id}",
          flush=True)
    return record


def post_thread(tweets: list[str]) -> list[dict]:
    """Post a sequence of tweets. Each subsequent tweet replies to the
    previous one's id."""
    posted: list[dict] = []
    last_id: str | None = None
    for i, text in enumerate(tweets):
        rec = post_one(text, reply_to=last_id)
        last_id = rec["tweet_id"]
        posted.append(rec)
        if i < len(tweets) - 1:
            time.sleep(2)  # courtesy spacing
    return posted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="Tweet text (single tweet).")
    parser.add_argument("--reply-to", help="Tweet ID to reply to.")
    parser.add_argument("--thread-file",
                        help="JSON array of tweet strings to post as a thread.")
    args = parser.parse_args()

    if args.thread_file:
        tweets = json.loads(Path(args.thread_file).read_text())
        if not isinstance(tweets, list):
            sys.exit("thread file must be a JSON array of strings")
        post_thread(tweets)
    elif args.text:
        post_one(args.text, reply_to=args.reply_to)
    else:
        parser.error("must provide --text or --thread-file")


if __name__ == "__main__":
    main()
