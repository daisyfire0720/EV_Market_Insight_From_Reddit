#!/usr/bin/env python3
"""Build the comments v2 label file: a balanced draw with the parent post as context.

Follows ``COMMENTS_V2_PLAN.md``. Bodies and parent text go to the gitignored
``data/raw``; the committed files carry ids, times, subreddits, and counts.
Nothing here calls a model. From ``v2/``::

    python build_comments_v2.py            # draw, backfill bodies, fetch parents, write files
    python build_comments_v2.py --offline  # use only what is already cached
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.comments_v2 import (  # noqa: E402
    CAP_PER_SIDE,
    SEED,
    assign_candidates,
    balanced_draw,
    is_usable_body,
    ordered_sides,
    parent_fields,
    prefix_ids,
)
from ev_v2.after_llm import parent_key  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
COMMENT_META = OUT / "comment_meta.csv"
FRAMING = OUT / "framing_labels.csv"
COMMENT_CACHE = RAW / "comments.jsonl"
PARENT_CACHE = RAW / "comment_parents.jsonl"
LABEL_POSTS = RAW / "label_posts.csv"
POSTS_OUT = RAW / "comments_v2_posts.csv"
META_OUT = OUT / "comments_v2_meta.csv"
COUNTS_OUT = OUT / "comments_v2_draw_counts.csv"
POST_IDS = "https://arctic-shift.photon-reddit.com/api/posts/ids"
PREFIX_STEPS = (1.6, 3.0, 6.0)


def _load_jsonl(path: Path) -> dict[str, dict]:
    have: dict[str, dict] = {}
    if not path.exists():
        return have
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("id"):
            prior = have.get(str(row["id"]), {})
            merged = dict(prior)
            merged.update({k: v for k, v in row.items() if v not in (None, "") or k not in prior})
            have[str(row["id"])] = merged
    return have


def _usable_ids(bodies: dict[str, dict]) -> set[str]:
    return {doc_id for doc_id, row in bodies.items() if is_usable_body(row.get("body"))}


def fetch_parents(link_ids: list[str]) -> dict[str, dict]:
    from fetch_comments import _get

    have = _load_jsonl(PARENT_CACHE)
    missing = [doc_id for doc_id in link_ids if doc_id not in have]
    print(f"parent fetch wanted {len(link_ids)} missing {len(missing)}", flush=True)
    with PARENT_CACHE.open("a", encoding="utf-8") as handle:
        for start in range(0, len(missing), 100):
            chunk = missing[start : start + 100]
            url = POST_IDS + "?" + urllib.parse.urlencode({"ids": ",".join(chunk), "fields": "id,title,selftext,subreddit"})
            payload = _get(url)
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
            returned = set()
            for row in rows:
                doc_id = str(row.get("id") or "")
                if not doc_id:
                    continue
                title, excerpt = parent_fields(row.get("title"), row.get("selftext"))
                stored = {"id": doc_id, "title": title, "excerpt": excerpt, "subreddit": str(row.get("subreddit") or "")}
                have[doc_id] = stored
                handle.write(json.dumps(stored) + "\n")
                returned.add(doc_id)
            if payload is not None:
                for doc_id in chunk:
                    if doc_id not in returned:
                        stored = {"id": doc_id, "title": "", "excerpt": "", "subreddit": "", "not_found": True}
                        have[doc_id] = stored
                        handle.write(json.dumps(stored) + "\n")
            handle.flush()
            print(f"  parents {min(start + 100, len(missing))}/{len(missing)}", flush=True)
            time.sleep(0.3)
    return have


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Balanced comments v2 draw with parent titles.")
    parser.add_argument("--cap", type=int, default=CAP_PER_SIDE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--offline", action="store_true", help="Do not call the archive; use cached bodies and parents only.")
    args = parser.parse_args(argv)

    meta = pd.read_csv(COMMENT_META)
    framing = pd.read_csv(FRAMING, usecols=["id", "audience"])
    parent_audience = dict(zip(framing["id"].astype(str), framing["audience"].astype(str)))
    candidates = assign_candidates(meta, parent_audience)
    print(f"comment meta {len(meta)} candidates in windows {len(candidates)}", flush=True)
    order = ordered_sides(candidates, seed=args.seed)

    bodies = _load_jsonl(COMMENT_CACHE)
    for factor in PREFIX_STEPS:
        k = int(round(args.cap * factor))
        wanted = prefix_ids(order, k)
        if not args.offline:
            ids_path = RAW / "comments_v2_backfill_ids.csv"
            missing = [i for i in wanted if not str((bodies.get(i) or {}).get("body") or "").strip()]
            if missing:
                pd.DataFrame({"id": missing}).to_csv(ids_path, index=False)
                from fetch_comments import backfill_bodies

                backfill_bodies(ids_path)
                bodies = _load_jsonl(COMMENT_CACHE)
        usable = _usable_ids(bodies) & set(wanted)
        rows, counts = balanced_draw(order, usable, cap=args.cap)
        short = counts[counts["n_drawn_per_side"] < counts[["n_candidates_left", "n_candidates_right"]].min(axis=1).clip(upper=args.cap)]
        print(f"prefix {k}: drawn {len(rows)}; cells short of what candidates allow {len(short)}", flush=True)
        if short.empty:
            break

    drawn = pd.DataFrame(rows, columns=["id", "draw_cutoff_id", "audience", "draw_side"])
    info = candidates.drop(columns=["audience"]).drop_duplicates("id").set_index("id")
    drawn = drawn.join(info[[c for c in ("author_hash", "created_utc", "subreddit", "link_id", "score", "delta_days", "audience_source") if c in info.columns]], on="id")
    drawn["text_type"] = "comment"
    drawn["parent_id"] = drawn["link_id"].map(parent_key)

    local = pd.read_csv(LABEL_POSTS, usecols=["id", "title", "selftext"])
    local["id"] = local["id"].astype(str)
    local_map = {row.id: parent_fields(row.title, row.selftext) for row in local.itertuples(index=False)}
    need = sorted({p for p in drawn["parent_id"] if p and p not in local_map})
    fetched = {} if args.offline else fetch_parents(need)
    if args.offline:
        fetched = _load_jsonl(PARENT_CACHE)
    titles, excerpts, sources = [], [], []
    for parent in drawn["parent_id"]:
        if parent in local_map:
            title, excerpt = local_map[parent]
            src = "label_posts"
        elif parent in fetched and (fetched[parent].get("title") or fetched[parent].get("excerpt")):
            title, excerpt = str(fetched[parent].get("title") or ""), str(fetched[parent].get("excerpt") or "")
            src = "archive"
        else:
            title, excerpt, src = "", "", "missing"
        titles.append(title)
        excerpts.append(excerpt)
        sources.append(src)
    drawn["parent_title"] = titles
    drawn["parent_excerpt"] = excerpts
    drawn["parent_source"] = sources
    drawn["body"] = drawn["id"].map(lambda i: str((bodies.get(i) or {}).get("body") or "")[:800])

    # Shuffle the label order once, so a smoke test on the first rows is a mix of cells.
    drawn = drawn.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    drawn.to_csv(POSTS_OUT, index=False)
    meta_cols = ["id", "author_hash", "created_utc", "subreddit", "link_id", "score", "text_type", "audience", "audience_source", "draw_cutoff_id", "draw_side", "delta_days", "parent_source"]
    drawn[[c for c in meta_cols if c in drawn.columns]].to_csv(META_OUT, index=False)
    counts.to_csv(COUNTS_OUT, index=False)
    print(f"wrote {POSTS_OUT} rows={len(drawn)}", flush=True)
    print(f"parent source: {drawn['parent_source'].value_counts().to_dict()}", flush=True)
    print(counts[["cutoff_id", "audience", "n_candidates_left", "n_candidates_right", "n_usable_left", "n_usable_right", "n_drawn_per_side"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
