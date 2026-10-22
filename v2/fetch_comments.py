#!/usr/bin/env python3
"""Comments by repeat authors, and comments on sampled submissions near the cutoffs.

Usernames stay in the gitignored cache. The committed files are hashes,
times, subreddits, and scores. Bodies are not committed.

The pull is resumable. From ``v2/``::

    python fetch_comments.py
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.comment_coverage import coverage_table  # noqa: E402
from ev_v2.rdit import CUTOFFS  # noqa: E402
from fetch_label_text import USER_AGENT, _hash_author  # noqa: E402

RAW = ROOT / "data" / "raw"
LABELS = ROOT / "outputs" / "framing_labels.csv"
AUTHORS = ROOT / "outputs" / "post_author_hashes.csv"
SCORES = ROOT / "outputs" / "post_scores.csv"
COMMENT_META = ROOT / "outputs" / "comment_meta.csv"
COVERAGE = ROOT / "outputs" / "comment_cutoff_coverage.csv"
PULL_STATUS = ROOT / "outputs" / "comment_pull_status.csv"
NAME_CACHE = RAW / "author_lookup.jsonl"
COMMENT_CACHE = RAW / "comments.jsonl"
POST_URL = "https://arctic-shift.photon-reddit.com/api/posts/ids"
COMMENT_SEARCH = "https://arctic-shift.photon-reddit.com/api/comments/search"
_FETCH_STATS = {"ok": 0, "timeout": 0}
SUBREDDITS = {
    "electricvehicles",
    "teslamotors",
    "boltev",
    "rivian",
    "ioniq5",
    "kiaev6",
    "evcharging",
    "cars",
    "autos",
    "carbuying",
}
WINDOW_DAYS = 180
MAX_PAGES = 20


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = "no response"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last = str(exc)
            wait = min(90, 3 + attempt * 8)
            print(f"  retry ({last}) sleep {wait}s", flush=True)
            time.sleep(wait)
    print(f"  gave up ({last})", flush=True)
    return None


def _load_jsonl(path: Path) -> dict[str, dict]:
    have: dict[str, dict] = {}
    if not path.exists():
        return have
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            have[str(row["id"])] = row
    return have


def _clusters() -> list[tuple[float, float]]:
    """Union of the ±180-day windows. The gap between 2023 and 2025 is not searched."""
    intervals = []
    for spec in CUTOFFS:
        day = spec["cutoff"]
        cutoff = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        intervals.append((cutoff - timedelta(days=WINDOW_DAYS), cutoff + timedelta(days=WINDOW_DAYS)))
    intervals.sort()
    merged: list[list[datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start.timestamp(), end.timestamp()) for start, end in merged]


def _in_clusters(created: float, clusters: list[tuple[float, float]]) -> bool:
    return any(start <= created < end for start, end in clusters)


def fetch_scores_and_names(labels: pd.DataFrame) -> dict[str, dict]:
    """id -> {author_hash, author (gitignored), score, created_utc}. Resumes."""
    RAW.mkdir(parents=True, exist_ok=True)
    have = _load_jsonl(NAME_CACHE)
    # Seed from local text files that already store the author.
    for path in (RAW / "thin_baseline_posts.csv", RAW / "mainstream_2026.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "author" not in frame.columns:
            continue
        with NAME_CACHE.open("a") as handle:
            for row in frame.itertuples(index=False):
                doc_id = str(getattr(row, "id", "") or "")
                if not doc_id or doc_id in have:
                    continue
                stored = {
                    "id": doc_id,
                    "author": str(getattr(row, "author", "") or ""),
                    "author_hash": _hash_author(str(getattr(row, "author", "") or "")),
                    "score": getattr(row, "score", None),
                    "created_utc": getattr(row, "created_utc", None),
                }
                have[doc_id] = stored
                handle.write(json.dumps(stored) + "\n")
    missing = [doc_id for doc_id in labels["id"].astype(str) if doc_id not in have]
    print(f"author/score cache {len(have)} missing {len(missing)}", flush=True)
    with NAME_CACHE.open("a") as handle:
        for start in range(0, len(missing), 200):
            chunk = missing[start : start + 200]
            url = POST_URL + "?ids=" + ",".join(chunk) + "&fields=id,author,score,created_utc"
            payload = _get(url)
            if not payload:
                continue
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
            for row in rows:
                doc_id = str(row.get("id") or "")
                if not doc_id or doc_id in have:
                    continue
                stored = {
                    "id": doc_id,
                    "author": str(row.get("author") or ""),
                    "author_hash": _hash_author(str(row.get("author") or "")),
                    "score": row.get("score"),
                    "created_utc": row.get("created_utc"),
                }
                have[doc_id] = stored
                handle.write(json.dumps(stored) + "\n")
            handle.flush()
            print(f"  posts {min(start + 200, len(missing))}/{len(missing)}", flush=True)
            time.sleep(0.4)
    scores = pd.DataFrame(
        [{"id": doc_id, "score": row.get("score")} for doc_id, row in have.items() if doc_id in set(labels["id"].astype(str))]
    )
    scores.to_csv(SCORES, index=False)
    print(f"wrote {SCORES} rows={len(scores)}", flush=True)
    return have


def _repeat_names(labels: pd.DataFrame, names: dict[str, dict], authors: pd.DataFrame) -> dict[str, str]:
    """hash -> one username. Deleted accounts are dropped."""
    merged = labels.merge(authors, on="id", how="left")
    merged["author_hash"] = merged["author_hash"].fillna("").astype(str)
    counts = merged.loc[merged["author_hash"] != ""].groupby("author_hash").size()
    repeat = set(counts[counts >= 2].index)
    found: dict[str, str] = {}
    for row in names.values():
        digest = str(row.get("author_hash") or "")
        author = str(row.get("author") or "")
        if digest in repeat and author and author not in {"[deleted]", "[removed]"} and digest not in found:
            found[digest] = author
    return found


def _keep_comment(row: dict, clusters: list[tuple[float, float]]) -> dict | None:
    subreddit = str(row.get("subreddit") or "")
    if subreddit.casefold() not in SUBREDDITS:
        return None
    created = row.get("created_utc")
    try:
        created_f = float(created)
    except (TypeError, ValueError):
        return None
    if not _in_clusters(created_f, clusters):
        return None
    author = str(row.get("author") or "")
    body = str(row.get("body") or "")
    if body.lower() in {"[removed]", "[deleted]"}:
        body = ""
    body = body[:800]
    link = str(row.get("link_id") or "")
    if link.startswith("t3_"):
        link = link[3:]
    return {
        "id": str(row.get("id") or ""),
        "author_hash": _hash_author(author),
        "created_utc": created_f,
        "subreddit": subreddit,
        "link_id": link,
        "score": row.get("score"),
        "text_type": "comment",
        "body": body,
    }


def _fetch_comment_page(params: dict) -> dict | str | None:
    """One comment-search page. A server timeout comes back as the string ``timeout`` so the caller can split the window."""
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(COMMENT_SEARCH + "?" + query, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                _FETCH_STATS["ok"] += 1
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code == 422 and "Timeout" in body:
                _FETCH_STATS["timeout"] += 1
                return "timeout"
            wait = 5 + attempt * 5
            print(f"  retry ({exc.code}) sleep {wait}s", flush=True)
            time.sleep(wait)
        except TimeoutError:
            _FETCH_STATS["timeout"] += 1
            return "timeout"
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError) or "timed out" in str(exc).lower():
                _FETCH_STATS["timeout"] += 1
                return "timeout"
            wait = 5 + attempt * 5
            print(f"  retry ({exc}) sleep {wait}s", flush=True)
            time.sleep(wait)
        except json.JSONDecodeError as exc:
            wait = 5 + attempt * 5
            print(f"  retry ({exc}) sleep {wait}s", flush=True)
            time.sleep(wait)
    return None


def needs_subreddit_split(payload, n_rows: int) -> bool:
    """True only for a server timeout. A full page is not fanned out across every subreddit."""
    del n_rows
    return payload == "timeout"


def _fetch_with_retry(params: dict, tries: int = 3) -> dict | str | None:
    """Repeat a timeout on the same query. A timeout is not a reason to hit every subreddit."""
    payload: dict | str | None = None
    for attempt in range(tries):
        payload = _fetch_comment_page(params)
        if payload != "timeout":
            return payload
        time.sleep(0.4 * (attempt + 1))
    return payload


def _comment_params(
    author: str,
    after: float,
    before: float,
    subreddit: str | None = None,
    sort: str = "asc",
    limit: int = 100,
) -> dict:
    params = {
        "author": author,
        "limit": limit,
        "sort": sort,
        "after": int(after),
        "before": int(before),
        "fields": "id,author,created_utc,subreddit,score,link_id",
    }
    if subreddit:
        params["subreddit"] = subreddit
    return params


def _absorb_page(payload: dict, seen: set[str], clusters: list[tuple[float, float]]) -> tuple[list[dict], int, float]:
    data = payload.get("data")
    batch = data if isinstance(data, list) else []
    rows: list[dict] = []
    last = 0.0
    for row in batch:
        doc_id = str(row.get("id") or "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            kept = _keep_comment(row, clusters)
            if kept and kept["id"]:
                rows.append(kept)
        try:
            last = max(last, float(row.get("created_utc") or 0))
        except (TypeError, ValueError):
            pass
    return rows, len(batch), last


SLICE_SECONDS = 45 * 86400


def _time_slices(start: float, end: float, step: float = SLICE_SECONDS):
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + step)
        if nxt <= cursor:
            break
        yield cursor, nxt
        cursor = nxt


def _search_range(
    author: str,
    start_utc: float,
    end_utc: float,
    seen: set[str],
    clusters: list[tuple[float, float]],
    subreddit: str | None = None,
    allow_slice: bool = True,
) -> tuple[list[dict], bool]:
    """Page one window. A timeout becomes 45-day slices, then one subreddit pass."""
    rows: list[dict] = []
    after = start_utc - 1
    for _page in range(MAX_PAGES):
        payload = _fetch_comment_page(_comment_params(author, after, end_utc, subreddit))
        if payload == "timeout":
            left = max(after, start_utc)
            if allow_slice and end_utc - left > SLICE_SECONDS:
                truncated = False
                for slice_start, slice_end in _time_slices(left, end_utc):
                    got, cut = _search_range(author, slice_start, slice_end, seen, clusters, subreddit, allow_slice=False)
                    rows.extend(got)
                    truncated = truncated or cut
                return rows, truncated
            if subreddit is None:
                truncated = False
                for name in sorted(SUBREDDITS):
                    got, cut = _search_range(author, left, end_utc, seen, clusters, name, allow_slice=False)
                    rows.extend(got)
                    truncated = truncated or cut
                    time.sleep(0.05)
                return rows, truncated
            return rows, True
        if not isinstance(payload, dict):
            return rows, True
        kept, n_batch, last = _absorb_page(payload, seen, clusters)
        rows.extend(kept)
        if n_batch < 100:
            return rows, False
        if last <= after:
            return rows, True
        after = last
        time.sleep(0.1)
    return rows, True


def _search_span(author: str, start_utc: float, end_utc: float, seen: set[str], clusters: list[tuple[float, float]]) -> tuple[list[dict], bool]:
    return _search_range(author, start_utc, end_utc, seen, clusters)


def _home_subreddits(labels: pd.DataFrame, authors: pd.DataFrame) -> dict[str, list[str]]:
    """Study subreddits where each hashed author has a submission. Most authors have one."""
    merged = labels.merge(authors[["id", "author_hash"]], on="id", how="left")
    merged["author_hash"] = merged["author_hash"].fillna("").astype(str)
    found: dict[str, set[str]] = {}
    for digest, subreddit in zip(merged["author_hash"], merged["subreddit"]):
        name = str(subreddit or "").casefold()
        if digest and name in SUBREDDITS:
            found.setdefault(digest, set()).add(name)
    return {digest: sorted(names) for digest, names in found.items()}


def _search_side(
    author: str,
    start_utc: float,
    end_utc: float,
    sort: str,
    seen: set[str],
    clusters: list[tuple[float, float]],
    home_subs: list[str] | None = None,
) -> tuple[list[dict], bool]:
    """Closest 100 comments on one side of a cutoff.

    A short page is the whole side. A full page that already includes a study
    subreddit is kept as the closest 100. A timeout, or a full page that is
    entirely outside the study subreddits, is retried on the subreddits where
    this author submitted.
    """
    home_subs = list(home_subs or [])
    payload = _fetch_with_retry(_comment_params(author, start_utc, end_utc, sort=sort))
    if isinstance(payload, dict):
        kept, n_batch, _last = _absorb_page(payload, seen, clusters)
        if not (n_batch >= 100 and not kept):
            return kept, False
    elif payload != "timeout":
        return [], True
    if not home_subs:
        return [], True
    rows: list[dict] = []
    failed = False
    for subreddit in home_subs:
        page = _fetch_with_retry(
            _comment_params(author, start_utc, end_utc, subreddit=subreddit, sort=sort, limit=100),
            tries=2,
        )
        if isinstance(page, dict):
            kept, _n_batch, _last = _absorb_page(page, seen, clusters)
            rows.extend(kept)
        else:
            failed = True
        time.sleep(0.05)
    return rows, failed


def _search_author(
    author: str,
    clusters: list[tuple[float, float]],
    home_subs: list[str] | None = None,
) -> tuple[list[dict], bool]:
    """Comments inside each cutoff window, closest to the cutoff first. ``clusters`` keeps the row filter."""
    del clusters
    from ev_v2.comment_coverage import WINDOW_DAYS, _cutoff_utc

    rows: list[dict] = []
    truncated = False
    seen: set[str] = set()
    windows = _clusters()
    homes = list(home_subs or [])
    day = WINDOW_DAYS * 86400.0
    for spec in CUTOFFS:
        cutoff = _cutoff_utc(spec["cutoff"])
        for start_utc, end_utc, sort in (
            (cutoff - day, cutoff, "desc"),
            (cutoff, cutoff + day, "asc"),
        ):
            got, failed = _search_side(author, start_utc, end_utc, sort, seen, windows, homes)
            rows.extend(got)
            truncated = truncated or failed
    return rows, truncated


def _search_link(link_id: str, clusters: list[tuple[float, float]]) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "link_id": link_id,
            "limit": 100,
            "sort": "asc",
            "fields": "id,author,created_utc,subreddit,score,link_id",
        }
    )
    payload = _get(COMMENT_SEARCH + "?" + query)
    if not payload:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    batch = data if isinstance(data, list) else []
    kept = []
    for row in batch:
        item = _keep_comment(row, clusters)
        if item and item["id"]:
            kept.append(item)
    return kept


def _finished_authors(done_path: Path) -> set[str]:
    """Hashes with a clean done line. A truncated line is retried."""
    if not done_path.exists():
        return set()
    clean: set[str] = set()
    truncated: set[str] = set()
    for line in done_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if "truncated" in parts:
            truncated.add(parts[0])
        else:
            clean.add(parts[0])
    return clean


def _write_outputs(
    submissions: pd.DataFrame,
    n_repeat: int | None = None,
    n_authors_searched: int | None = None,
    complete: bool = False,
    threads_complete: bool = False,
) -> None:
    """Reload the cache so one shard cannot overwrite another shard's rows."""
    lock = _lock()
    try:
        comments = _load_jsonl(COMMENT_CACHE)
    finally:
        lock.close()
    clusters = _clusters()
    meta = pd.DataFrame(
        [
            {key: row.get(key) for key in ("id", "author_hash", "created_utc", "subreddit", "link_id", "score", "text_type")}
            for row in comments.values()
            if _in_clusters(float(row.get("created_utc") or 0), clusters)
        ]
    )
    if meta.empty:
        meta = pd.DataFrame(columns=["id", "author_hash", "created_utc", "subreddit", "link_id", "score", "text_type"])
    meta.to_csv(COMMENT_META, index=False)
    events = pd.concat(
        [
            submissions.assign(text_type="submission")[["author_hash", "created_utc", "text_type"]],
            meta[["author_hash", "created_utc", "text_type"]],
        ],
        ignore_index=True,
    )
    coverage_table(events).to_csv(COVERAGE, index=False)
    if n_repeat is not None and n_authors_searched is not None:
        pd.DataFrame(
            [
                {
                    "n_repeat_authors": int(n_repeat),
                    "n_authors_searched": int(n_authors_searched),
                    "n_comments": int(len(meta)),
                    "complete": int(bool(complete)),
                    "threads_complete": int(bool(threads_complete)),
                    "cap_per_side": 100,
                }
            ]
        ).to_csv(PULL_STATUS, index=False)
    print(f"wrote {COMMENT_META} rows={len(meta)} and {COVERAGE}", flush=True)


def _lock():
    RAW.mkdir(parents=True, exist_ok=True)
    handle = (RAW / "comment_fetch.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


COMMENT_IDS = "https://arctic-shift.photon-reddit.com/api/comments/ids"


def backfill_bodies(id_csv: Path) -> int:
    """Fill empty bodies in the gitignored cache. Usernames are not written."""
    wanted = pd.read_csv(id_csv)
    if "id" not in wanted.columns:
        raise SystemExit("The id file needs an id column.")
    ids = [str(item) for item in wanted["id"].tolist()]
    have = _load_jsonl(COMMENT_CACHE)
    missing = [doc_id for doc_id in ids if not str((have.get(doc_id) or {}).get("body") or "").strip()]
    print(f"body backfill wanted {len(ids)} missing {len(missing)}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    filled = 0
    clusters = _clusters()
    with COMMENT_CACHE.open("a") as handle:
        for start in range(0, len(missing), 100):
            chunk = missing[start : start + 100]
            url = COMMENT_IDS + "?" + urllib.parse.urlencode({"ids": ",".join(chunk), "fields": "id,author,body,created_utc,score,subreddit,link_id"})
            payload = _get(url)
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
            for row in rows:
                doc_id = str(row.get("id") or "")
                if not doc_id:
                    continue
                kept = _keep_comment(row, clusters)
                base = dict(have.get(doc_id) or kept or {})
                if not base.get("id"):
                    continue
                body = str(row.get("body") or "")
                if body.lower() in {"[removed]", "[deleted]"}:
                    body = ""
                base["body"] = body[:800]
                base.pop("author", None)
                have[doc_id] = base
                handle.write(json.dumps(base) + "\n")
                filled += 1
            print(f"  bodies {min(start + 100, len(missing))}/{len(missing)}", flush=True)
            time.sleep(0.3)
    print(f"bodies filled {filled}", flush=True)
    return filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull comments for the cutoff windows.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--phase", choices=("authors", "links", "all"), default="all")
    parser.add_argument("--backfill", type=Path, default=None, help="CSV of comment ids whose bodies should be filled in the gitignored cache.")
    args = parser.parse_args(argv)
    if args.backfill:
        backfill_bodies(args.backfill)
        return 0
    labels = pd.read_csv(LABELS)
    labels["id"] = labels["id"].astype(str)
    authors = pd.read_csv(AUTHORS)
    authors["id"] = authors["id"].astype(str)
    names = fetch_scores_and_names(labels)
    repeat = _repeat_names(labels, names, authors)
    homes = _home_subreddits(labels, authors)
    clusters = _clusters()
    print(f"repeat authors with a username {len(repeat)} shard {args.shard}/{args.shards} clusters {len(clusters)}", flush=True)
    comments = _load_jsonl(COMMENT_CACHE)
    done_path = RAW / "comment_authors_done.txt"
    done = _finished_authors(done_path)
    submissions = labels.merge(authors[["id", "author_hash", "created_utc"]], on="id", how="left")
    mine = [
        (digest, author)
        for digest, author in repeat.items()
        if int(digest[:2], 16) % args.shards == args.shard and digest not in done
    ]
    if args.phase in {"authors", "all"}:
        with COMMENT_CACHE.open("a") as handle, done_path.open("a") as done_handle:
            for index, (digest, author) in enumerate(mine, start=1):
                rows, truncated = _search_author(author, clusters, homes.get(digest, []))
                lock = _lock()
                try:
                    for row in rows:
                        if row["id"] in comments:
                            continue
                        comments[row["id"]] = row
                        handle.write(json.dumps(row) + "\n")
                    done_handle.write(digest + (" truncated\n" if truncated else "\n"))
                    done_handle.flush()
                    handle.flush()
                finally:
                    lock.close()
                if index % 10 == 0 or index == len(mine):
                    print(
                        f"  shard {args.shard} authors {index}/{len(mine)} comments {len(comments)} "
                        f"ok={_FETCH_STATS['ok']} timeout={_FETCH_STATS['timeout']}",
                        flush=True,
                    )
                if index % 100 == 0:
                    _write_outputs(
                        submissions,
                        n_repeat=len(repeat),
                        n_authors_searched=len(_finished_authors(done_path)),
                        complete=False,
                    )
                time.sleep(0.12)
        searched_now = len(_finished_authors(done_path))
        _write_outputs(
            submissions,
            n_repeat=len(repeat),
            n_authors_searched=searched_now,
            complete=searched_now >= len(repeat),
        )
    if args.phase in {"links", "all"}:
        created = pd.to_numeric(submissions["created_utc"], errors="coerce")
        mask = created.map(lambda value: _in_clusters(float(value), clusters) if pd.notna(value) else False)
        in_window = submissions[mask]
        link_done_path = RAW / "comment_links_done.txt"
        link_done = set(link_done_path.read_text().split()) if link_done_path.exists() else set()
        pending = [
            doc_id
            for doc_id in in_window["id"].astype(str)
            if doc_id not in link_done and sum(ord(char) for char in doc_id) % args.shards == args.shard
        ]
        print(f"submission threads in the windows {len(in_window)} still to fetch {len(pending)}", flush=True)
        with COMMENT_CACHE.open("a") as handle, link_done_path.open("a") as link_handle:
            for index, doc_id in enumerate(pending, start=1):
                rows = _search_link(doc_id, clusters)
                lock = _lock()
                try:
                    for row in rows:
                        if row["id"] in comments:
                            continue
                        comments[row["id"]] = row
                        handle.write(json.dumps(row) + "\n")
                    link_handle.write(doc_id + "\n")
                    handle.flush()
                    link_handle.flush()
                finally:
                    lock.close()
                if index % 50 == 0 or index == len(pending):
                    print(f"  threads {index}/{len(pending)} comments {len(comments)}", flush=True)
                time.sleep(0.25)
        searched = len(_finished_authors(done_path))
        finished_links = set(link_done_path.read_text().split()) if link_done_path.exists() else set()
        threads_done = set(in_window["id"].astype(str)) <= finished_links
        _write_outputs(
            submissions,
            n_repeat=len(repeat),
            n_authors_searched=searched,
            complete=searched >= len(repeat),
            threads_complete=threads_done,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
