#!/usr/bin/env python3
"""Fetch title, body, author, and time for the labeled ids.

The cache is gitignored. Usernames stay in that cache. Committed outputs
store a hash only. The script resumes from the cache.

From ``v2/``::

    python fetch_label_text.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

IDS_URL = "https://arctic-shift.photon-reddit.com/api/posts/ids"
USER_AGENT = "ev-reddit-research/0.2 (academic sample; Arctic Shift API)"
RAW = ROOT / "data" / "raw"
CACHE = RAW / "label_text.jsonl"
LABELS = ROOT / "outputs" / "framing_labels.csv"
AUTHORS = ROOT / "outputs" / "post_author_hashes.csv"
FIELDS = "id,author,created_utc,title,selftext"
EXCERPT = 800


def _hash_author(author: str) -> str:
    if author is None or str(author).strip() in {"", "[deleted]", "[removed]"}:
        return ""
    return hashlib.sha256(f"ev-reddit-v2|{author}".encode()).hexdigest()[:16]


def _load_cache() -> dict[str, dict]:
    have: dict[str, dict] = {}
    if not CACHE.exists():
        return have
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        have[str(row["id"])] = row
    return have


def _local_rows() -> dict[str, dict]:
    """Posts already on disk, so the API is not asked for them again."""
    found: dict[str, dict] = {}
    for path in (RAW / "thin_baseline_posts.csv", RAW / "mainstream_2026.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            doc_id = str(getattr(row, "id", "") or "")
            if not doc_id:
                continue
            title = str(getattr(row, "title", "") or "")
            body = str(getattr(row, "selftext", "") or "")
            text = f"{title}\n{body}".strip()
            found[doc_id] = {
                "id": doc_id,
                "author_hash": _hash_author(str(getattr(row, "author", "") or "")),
                "created_utc": getattr(row, "created_utc", None),
                "title": title[:300],
                "excerpt": text[:EXCERPT],
            }
    return found


def _fetch(ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(ids), 200):
        chunk = ids[start : start + 200]
        url = IDS_URL + "?ids=" + ",".join(chunk) + "&fields=" + FIELDS
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        payload = None
        last_error = "no response"
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                wait = min(90, 3 + attempt * 8)
                print(f"  retry {start} ({last_error}) sleep {wait}s", flush=True)
                time.sleep(wait)
        if payload is None:
            print(f"  gave up at {start} ({last_error})", flush=True)
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.extend(data.values())
        print(f"  ids {min(start + 200, len(ids))}/{len(ids)} stored {len(rows)}", flush=True)
        time.sleep(0.4)
    return rows


def _slim(row: dict) -> dict:
    title = str(row.get("title") or "")
    body = str(row.get("selftext") or "")
    if body.lower() in {"[deleted]", "[removed]", "nan"}:
        body = ""
    text = f"{title}\n{body}".strip()
    created = row.get("created_utc")
    return {
        "id": str(row.get("id") or ""),
        "author_hash": _hash_author(str(row.get("author") or "")),
        "created_utc": created,
        "title": title[:300],
        "excerpt": text[:EXCERPT],
    }


def write_author_table(cache: dict[str, dict]) -> None:
    rows = [
        {"id": doc_id, "author_hash": row.get("author_hash") or "", "created_utc": row.get("created_utc")}
        for doc_id, row in cache.items()
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(AUTHORS, index=False)
    print(f"wrote {AUTHORS} rows={len(frame)}", flush=True)


def main() -> int:
    labels = pd.read_csv(LABELS)
    wanted = [str(item) for item in labels["id"].tolist()]
    have = _load_cache()
    local = _local_rows()
    RAW.mkdir(parents=True, exist_ok=True)
    added = 0
    with CACHE.open("a") as handle:
        for doc_id, row in local.items():
            if doc_id in have or doc_id not in set(wanted):
                continue
            have[doc_id] = row
            handle.write(json.dumps(row) + "\n")
            added += 1
        handle.flush()
        missing = [doc_id for doc_id in wanted if doc_id not in have]
        print(f"cache {len(have)} local_added {added} missing {len(missing)}", flush=True)
        if missing:
            for row in _fetch(missing):
                slim = _slim(row)
                if not slim["id"] or slim["id"] in have:
                    continue
                have[slim["id"]] = slim
                handle.write(json.dumps(slim) + "\n")
                added += 1
                if added % 200 == 0:
                    handle.flush()
    write_author_table(have)
    print(f"covered {sum(1 for doc_id in wanted if doc_id in have)} of {len(wanted)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
