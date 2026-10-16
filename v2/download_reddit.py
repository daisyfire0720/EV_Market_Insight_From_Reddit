#!/usr/bin/env python3
"""Download a reproducible EV-Reddit sample from the Arctic Shift API.

Arctic Shift (https://arctic-shift.photon-reddit.com) is the source. Pushshift
no longer updates. The official Reddit API needs OAuth credentials this
environment does not have. Academic Torrents hosts historical dumps, but the
API is what can be queried through the latest archived month without
shipping a multi-gigabyte dump.

What this script writes
-----------------------
- ``v2/data/raw/`` (gitignored): request cache, full sample of submissions
- ``v2/data_card/monthly_post_counts.csv``: census of submissions per
  subreddit-month for the core communities, from the aggregate endpoint
- ``v2/data_card/download_log.json``: source, range, filters, failures
- ``v2/sample/posts_sample.csv``: a small de-identified sample safe to commit

The aggregate endpoint labels months in Europe/Amsterdam. See ``ev_v2.arctic``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.arctic import (  # noqa: E402
    COMMENTS_SEARCH,
    CORE_SUBREDDITS,
    GENERAL_KEYWORDS,
    GENERAL_SUBREDDITS,
    POSTS_AGGREGATE,
    POSTS_SEARCH,
    bucket_month,
)

USER_AGENT = "ev-reddit-research/0.2 (academic sample; Arctic Shift API)"
RAW_DIR = ROOT / "data" / "raw"
CARD_DIR = ROOT / "data_card"
SAMPLE_DIR = ROOT / "sample"
CACHE_PATH = RAW_DIR / "request_cache.json"
RANGE_START = date(2015, 1, 1)

POST_FIELDS = ",".join(
    [
        "id",
        "created_utc",
        "author",
        "title",
        "selftext",
        "score",
        "num_comments",
        "subreddit",
        "url",
        "link_flair_text",
    ]
)


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _sleep_for(headers, attempt: int, status: int | None = None) -> None:
    # 422 on this API is a query timeout, not a rate limit. The reset header
    # is still sent, and sleeping for it turns every timeout into a ~40s wait.
    if status == 429 and headers is not None:
        raw = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
        if raw:
            try:
                time.sleep(min(float(raw) + 0.5, 90))
                return
            except ValueError:
                pass
    time.sleep(min(1.5 * (attempt + 1), 8))


class ArcticClient:
    def __init__(self, cache_path: Path, min_interval: float = 0.8):
        self.cache_path = cache_path
        self.min_interval = min_interval
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self.n_network = 0
        self.n_cache = 0
        self._last = 0.0

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.cache_path)

    def get(self, url: str, params: dict) -> dict:
        key = url + "?" + urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
        if key in self.cache:
            self.n_cache += 1
            return self.cache[key]
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        last_error = "no response"
        for attempt in range(3):
            full = key
            req = urllib.request.Request(
                full,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = resp.read()
                    headers = resp.headers
                    status = resp.status
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                headers = exc.headers
                status = exc.code
            except Exception as exc:  # noqa: BLE001 - network errors are retried
                last_error = f"{type(exc).__name__}: {exc}"
                _sleep_for(None, attempt)
                continue
            self._last = time.time()
            self.n_network += 1
            try:
                data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                last_error = f"HTTP {status} non-json: {payload[:200]!r}"
                _sleep_for(headers, attempt, status)
                continue
            if status == 429 or (isinstance(data, dict) and data.get("error")):
                last_error = f"HTTP {status}: {data.get('error') if isinstance(data, dict) else payload[:200]!r}"
                print(f"  retry {attempt + 1} {last_error}", flush=True)
                _sleep_for(headers, attempt, status)
                continue
            if not isinstance(data, dict):
                last_error = f"unexpected payload type {type(data)}"
                _sleep_for(headers, attempt, status)
                continue
            self.cache[key] = data
            if self.n_network % 25 == 0:
                self.save()
            return data
        raise RuntimeError(f"Arctic Shift request failed: {key} ({last_error})")


def _month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            m = 1
            y += 1


def _split_range(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]] | None:
    if (end - start).days < 32:
        return None
    mid = start + (end - start) / 2
    mid = date(mid.year, mid.month, 1)
    if mid <= start or mid >= end:
        return None
    return (start, mid), (mid, end)


def fetch_monthly_counts(client: ArcticClient, subreddit: str, start: date, end: date) -> list[dict]:
    """Monthly submission counts for [start, end). Splits the span on timeout."""
    params = {
        "subreddit": subreddit,
        "after": start.isoformat(),
        "before": end.isoformat(),
        "aggregate": "created_utc",
        "frequency": "month",
    }
    try:
        data = client.get(POSTS_AGGREGATE, params)
    except RuntimeError as exc:
        pieces = _split_range(start, end)
        if pieces is None:
            print(f"  GIVING UP {subreddit} {start}..{end}: {exc}", flush=True)
            return [
                {
                    "subreddit": subreddit,
                    "month": month,
                    "n_posts": None,
                    "count_status": "timeout",
                }
                for month in _month_iter(start, end - timedelta(days=1))
            ]
        print(f"  split {subreddit} {start}..{end}", flush=True)
        left, right = pieces
        return fetch_monthly_counts(client, subreddit, *left) + fetch_monthly_counts(
            client, subreddit, *right
        )
    rows = []
    seen = set()
    for item in data.get("data") or []:
        month = bucket_month(item["created_utc"])
        if month < start.strftime("%Y-%m") or month > (end - timedelta(days=1)).strftime("%Y-%m"):
            continue
        rows.append(
            {
                "subreddit": subreddit,
                "month": month,
                "n_posts": int(item["count"]),
                "count_status": "ok",
            }
        )
        seen.add(month)
    for month in _month_iter(start, end - timedelta(days=1)):
        if month not in seen:
            rows.append(
                {
                    "subreddit": subreddit,
                    "month": month,
                    "n_posts": 0,
                    "count_status": "ok_empty",
                }
            )
    return rows


def _quarter_window(subreddit: str, year: int, quarter: int, today: date) -> tuple[date, date] | None:
    start_month = (quarter - 1) * 3 + 1
    q_start = date(year, start_month, 1)
    if start_month == 10:
        q_end = date(year + 1, 1, 1)
    else:
        q_end = date(year, start_month + 3, 1)
    if q_start > today:
        return None
    digest = hashlib.sha256(f"v1|{subreddit}|{year}|Q{quarter}".encode()).hexdigest()
    offset = int(digest[:8], 16) % 70
    win_start = q_start + timedelta(days=offset)
    win_end = win_start + timedelta(days=7)
    if win_end > q_end:
        win_end = q_end
        win_start = max(q_start, win_end - timedelta(days=7))
    if win_start >= today:
        return None
    if win_end > today + timedelta(days=1):
        win_end = today + timedelta(days=1)
    if win_end <= win_start:
        return None
    return win_start, win_end


def fetch_posts(
    client: ArcticClient,
    subreddit: str,
    after: date,
    before: date,
    limit: int,
    query: str | None = None,
    query_field: str = "query",
) -> list[dict]:
    params = {
        "subreddit": subreddit,
        "after": after.isoformat(),
        "before": before.isoformat(),
        "limit": str(limit),
        "sort": "asc",
        "fields": POST_FIELDS,
    }
    if query:
        params[query_field] = query
    data = client.get(POSTS_SEARCH, params)
    rows = data.get("data") or []
    for row in rows:
        row["query"] = query or ""
        row["sample_after"] = after.isoformat()
        row["sample_before"] = before.isoformat()
        row["sample_truncated"] = len(rows) >= limit
    return rows


def _records_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "created_utc",
                "author",
                "title",
                "selftext",
                "score",
                "num_comments",
                "subreddit",
                "url",
                "link_flair_text",
                "query",
                "sample_after",
                "sample_before",
                "sample_truncated",
                "community_group",
            ]
        )
    frame = pd.DataFrame(rows)
    for col in ["selftext", "title", "author", "link_flair_text", "url", "query"]:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str)
    frame["created_utc"] = pd.to_numeric(frame["created_utc"], errors="coerce")
    frame["created_date"] = pd.to_datetime(frame["created_utc"], unit="s", utc=True)
    frame["month"] = frame["created_date"].dt.strftime("%Y-%m")
    return frame


def _hash_author(author: str) -> str:
    if author is None or author in {"", "[deleted]", "[removed]"}:
        return ""
    return hashlib.sha256(f"ev-reddit-v2|{author}".encode()).hexdigest()[:12]


def write_public_sample(frame: pd.DataFrame, path: Path, per_sub_year: int = 2) -> pd.DataFrame:
    if frame.empty:
        path.write_text("id,subreddit,month\n")
        return frame
    work = frame.copy()
    work["year"] = work["created_date"].dt.year
    work = work.sort_values(["subreddit", "created_utc", "id"])
    parts = []
    for (_, _), group in work.groupby(["subreddit", "year"], sort=True):
        take = min(per_sub_year, len(group))
        # Spread across the year instead of taking only January posts.
        if len(group) <= take:
            parts.append(group)
        else:
            indexes = [round(i * (len(group) - 1) / (take - 1)) for i in range(take)] if take > 1 else [0]
            parts.append(group.iloc[indexes])
    sample = pd.concat(parts, ignore_index=True)
    sample["author_hash"] = sample["author"].map(_hash_author)
    sample["selftext_excerpt"] = sample["selftext"].str.replace("\n", " ", regex=False).str.slice(0, 400)
    sample["title"] = sample["title"].str.replace("\n", " ", regex=False).str.slice(0, 300)
    keep = [
        "id",
        "subreddit",
        "community_group",
        "created_utc",
        "month",
        "score",
        "num_comments",
        "link_flair_text",
        "title",
        "selftext_excerpt",
        "author_hash",
        "query",
    ]
    public = sample[keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(path, index=False)
    return public


def _probe_comment(client: ArcticClient) -> dict:
    """One comment query so the data card can say whether comments are reachable."""
    try:
        data = client.get(
            COMMENTS_SEARCH,
            {
                "subreddit": "BoltEV",
                "after": "2024-06-01",
                "before": "2024-06-08",
                "limit": "2",
                "sort": "asc",
                "fields": "id,created_utc,subreddit",
            },
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    rows = data.get("data") or []
    return {"ok": True, "n_returned": len(rows), "example_id": rows[0]["id"] if rows else None}


def write_data_card(counts: pd.DataFrame, log: dict) -> None:
    """Write the data card from the files this run just produced."""
    lines = [
        "# Data card: EV Reddit submissions",
        "",
        "This card is generated by `v2/download_reddit.py` from the Arctic Shift responses. Counts are not estimated.",
        "",
        "## Source",
        "",
        "- Archive: [Arctic Shift](https://arctic-shift.photon-reddit.com), API documented in [api/README.md](https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md).",
        "- Endpoint for counts: `GET /api/posts/search/aggregate?aggregate=created_utc&frequency=month`.",
        "- Endpoint for text: `GET /api/posts/search`.",
        "- Retrieved at: `{}`.".format(log["retrieved_at_utc"]),
        "- Requested range: {} inclusive through {} exclusive.".format(log["range_start"], log["range_end_exclusive"]),
        "- Unit: Reddit submissions. A one-week comment probe is recorded in `download_log.json`. Comments are not part of the census.",
        "",
        "Pushshift, which the original project and Ruan and Lv (2022) used, no longer provides a current dump. Reddit's official API needs OAuth credentials that are not available here. Academic Torrents distributes historical Reddit dumps, which are the wrong tool for a selective, up-to-date pull.",
        "",
        "## Filters",
        "",
        "- Core communities (every submission counted): {}.".format(", ".join(f"r/{s}" for s in log["core_subreddits"])),
        "- Mainstream communities, keyword sample only, not a census: {}.".format(", ".join(f"r/{s}" for s in log["general_subreddits"])),
        "- Keyword field chosen per subreddit after a probe: `{}`.".format(json.dumps(log.get("general_keyword_field", {}))),
        "- Keywords, issued separately and de-duplicated by post id: {}.".format(", ".join(log["general_keywords"])),
        "- Bare `ev` is not a keyword. It matches inside unrelated words.",
        "- No score filter and no bot filter are applied at download time. Both change the population and are applied, if at all, in the analysis script where the removal counts are logged.",
        "- Arctic Shift's own note: `score` and `num_comments` stay at 0 or 1 until a post is about 36 hours old.",
        "",
        "## Sampling",
        "",
        "Monthly counts for the core communities are a census of submissions in the archive, one month per request. The text files are not a simple random sample. For each subreddit-quarter the script draws one deterministic 7-day window (SHA-256 of the stratum, offset inside the quarter) and keeps up to 25 posts. It also keeps up to 40 posts from a fixed week around each coded policy event. Mainstream posts use the same windows with the keyword filter and a limit of 12.",
        "",
        "The committed file `v2/sample/posts_sample.csv` is a further excerpt (up to two posts per subreddit-year), with author names replaced by a hash and selftext cut at 400 characters. The fuller text sample stays in `v2/data/raw/`, which git ignores.",
        "",
        "## Counts",
        "",
    ]
    if counts.empty:
        lines.append("No count rows were returned.")
    else:
        ok = counts[counts["count_status"].isin(["ok", "ok_empty"])].copy()
        ok["n_posts"] = pd.to_numeric(ok["n_posts"], errors="coerce").fillna(0).astype(int)
        # The last calendar month of the pull is incomplete.
        end_month = (date.fromisoformat(log["range_end_exclusive"]) - timedelta(days=1)).strftime("%Y-%m")
        complete = ok[ok["month"] < end_month]
        partial = ok[ok["month"] == end_month]
        lines.append(f"Complete months in this pull run through the month before `{end_month}`. `{end_month}` is partial because the archive was queried during that month.")
        lines.append("")
        lines.append("| Subreddit | Complete-month posts | Months with a count | Partial last month | Timeouts |")
        lines.append("|---|---:|---:|---:|---:|")
        for sub, group in counts.groupby("subreddit", sort=False):
            g_ok = ok[ok["subreddit"] == sub]
            g_complete = g_ok[g_ok["month"] < end_month]
            g_partial = g_ok[g_ok["month"] == end_month]
            n_timeout = int((group["count_status"] == "timeout").sum())
            partial_n = int(g_partial["n_posts"].sum()) if len(g_partial) else 0
            first = g_ok["month"].min() if len(g_ok) else ""
            last = g_ok["month"].max() if len(g_ok) else ""
            lines.append(
                f"| r/{sub} | {int(g_complete['n_posts'].sum())} | {g_ok['month'].nunique()} ({first} to {last}) | {partial_n} | {n_timeout} |"
            )
        lines.append("")
        n_timeout_total = int((counts["count_status"] == "timeout").sum())
        if n_timeout_total:
            lines.append(
                f"Sum of posts in months the API returned, excluding the partial last month: **{int(complete['n_posts'].sum())}**. "
                f"{n_timeout_total} subreddit-months timed out and are omitted, so the sum is a lower bound."
            )
        else:
            lines.append(
                f"Sum of posts in complete months: **{int(complete['n_posts'].sum())}**. "
                "Every requested core subreddit-month returned a count. The last calendar month is partial and is not in that sum."
            )
        lines.append("")
        lines.append("Year totals (complete months only) are in `yearly_post_counts.csv`. The month table is `monthly_post_counts.csv`.")
        yearly = complete.copy()
        yearly["year"] = yearly["month"].str.slice(0, 4)
        pivot = yearly.pivot_table(index="year", columns="subreddit", values="n_posts", aggfunc="sum", fill_value=0)
        pivot["total"] = pivot.sum(axis=1)
        pivot.to_csv(CARD_DIR / "yearly_post_counts.csv")
        lines.append("")
        lines.append("Downloaded text sample (de-duplicated posts): **{}**. By subreddit: `{}`.".format(
            log["downloaded_posts_deduped"], json.dumps(log["downloaded_posts_by_subreddit"])
        ))
        lines.append("")
        lines.append(f"Public excerpt rows: {log['public_sample_rows']}.")
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "Request failures in the first pass: {}. The second mainstream pass recorded {} further failures. Both lists are in `download_log.json`.".format(
            len(log.get("failures") or []),
            len((log.get("mainstream_fill") or {}).get("failures") or []),
        ),
            "",
            "## Ethics",
            "",
            "Posts on these subreddits are public. The committed excerpt hashes usernames and truncates body text. The local analysis file still has usernames so the bot filter can run; it is gitignored. This pull does not try to identify anyone. Arctic Shift states that it offers no uptime or performance guarantee.",
            "",
        ]
    )
    (CARD_DIR / "DATA_CARD.md").write_text("\n".join(lines))


def retry_timeouts(client: ArcticClient) -> None:
    path = CARD_DIR / "monthly_post_counts.csv"
    counts = pd.read_csv(path)
    pending = counts[counts["count_status"] == "timeout"]
    print(f"retrying {len(pending)} timeout months", flush=True)
    for _, row in pending.iterrows():
        start = date.fromisoformat(f"{row['month']}-01")
        if start.month == 12:
            end = date(start.year + 1, 1, 1)
        else:
            end = date(start.year, start.month + 1, 1)
        try:
            fresh = fetch_monthly_counts(client, row["subreddit"], start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  still failing {row['subreddit']} {row['month']}: {exc}", flush=True)
            continue
        for item in fresh:
            if item["month"] != row["month"] or item["count_status"] == "timeout":
                continue
            mask = (counts["subreddit"] == row["subreddit"]) & (counts["month"] == row["month"])
            counts.loc[mask, "n_posts"] = item["n_posts"]
            counts.loc[mask, "count_status"] = item["count_status"]
            print(f"  filled {row['subreddit']} {row['month']} = {item['n_posts']}", flush=True)
        client.save()
    counts.to_csv(path, index=False)
    log_path = CARD_DIR / "download_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log["count_timeouts_after_retry"] = int((counts["count_status"] == "timeout").sum())
    log_path.write_text(json.dumps(log, indent=2))
    write_data_card(counts, log if log.get("retrieved_at_utc") else {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_start": RANGE_START.isoformat(),
        "range_end_exclusive": _today_utc().isoformat(),
        "core_subreddits": list(CORE_SUBREDDITS),
        "general_subreddits": list(GENERAL_SUBREDDITS),
        "general_keywords": list(GENERAL_KEYWORDS),
        "general_keyword_field": {},
        "downloaded_posts_deduped": 0,
        "downloaded_posts_by_subreddit": {},
        "public_sample_rows": 0,
        "failures": [],
    })


def _merge_posts(new_rows: list[dict]) -> pd.DataFrame:
    full_path = RAW_DIR / "posts_downloaded.csv"
    new = _records_to_frame(new_rows)
    if full_path.exists():
        old = pd.read_csv(full_path)
        posts = pd.concat([old, new], ignore_index=True)
    else:
        posts = new
    if not posts.empty:
        posts = posts.drop_duplicates("id")
        posts["created_utc"] = pd.to_numeric(posts["created_utc"], errors="coerce")
        posts["created_date"] = pd.to_datetime(posts["created_utc"], unit="s", utc=True)
        posts["month"] = posts["created_date"].dt.strftime("%Y-%m")
    posts.to_csv(full_path, index=False)
    write_public_sample(posts, SAMPLE_DIR / "posts_sample.csv")
    return posts


def fill_mainstream(client: ArcticClient) -> None:
    """Keyword sample that keeps going after a timeout.

    The first pass aborted r/cars, r/Autos, and r/carbuying after three
    timeouts in 2015, so later years were never requested. This pass asks
    for two keywords in a fixed week of each half-year from 2018 on, and
    for the coded event weeks.
    """
    keywords = ("tesla", "electric vehicle")
    end = _today_utc() + timedelta(days=1)
    rows: list[dict] = []
    failures: list[dict] = []
    windows: list[tuple[str, date, date]] = []
    for year in range(2018, end.year + 1):
        for half, month, day in ((1, 2, 10), (2, 8, 10)):
            start = date(year, month, day)
            stop = start + timedelta(days=7)
            if start >= end:
                continue
            windows.append((f"{year}H{half}", start, min(stop, end)))
    windows.extend(
        [
            ("ira_2022", date(2022, 8, 14), date(2022, 8, 21)),
            ("tesla_price_cut_2023", date(2023, 1, 12), date(2023, 1, 19)),
            ("nacs_ford", date(2023, 5, 24), date(2023, 5, 31)),
            ("nacs_gm", date(2023, 6, 7), date(2023, 6, 14)),
            ("credit_law_2025", date(2025, 7, 3), date(2025, 7, 10)),
            ("credit_deadline_2025", date(2025, 9, 28), date(2025, 10, 5)),
        ]
    )
    for sub in GENERAL_SUBREDDITS:
        print(f"mainstream fill {sub}", flush=True)
        for stratum, start, stop in windows:
            if start >= end:
                continue
            for keyword in keywords:
                try:
                    got = fetch_posts(
                        client, sub, start, min(stop, end), limit=15, query=keyword, query_field="query"
                    )
                except RuntimeError as exc:
                    failures.append({"subreddit": sub, "keyword": keyword, "stratum": stratum, "error": str(exc)})
                    print(f"  fail {sub} {stratum} {keyword}", flush=True)
                    continue
                for row in got:
                    row["community_group"] = "mainstream_car"
                    row["sample_stratum"] = stratum
                rows.extend(got)
                print(f"  {sub} {stratum} {keyword}: {len(got)}", flush=True)
        client.save()
    posts = _merge_posts(rows)
    by_sub = posts.groupby("subreddit").size().astype(int).to_dict() if not posts.empty else {}
    log_path = CARD_DIR / "download_log.json"
    log = json.loads(log_path.read_text())
    log["mainstream_fill"] = {
        "keywords": list(keywords),
        "new_rows_before_dedupe": len(rows),
        "failures": failures,
    }
    log["downloaded_posts_deduped"] = int(len(posts))
    log["downloaded_posts_by_subreddit"] = by_sub
    log["public_sample_rows"] = int(len(pd.read_csv(SAMPLE_DIR / "posts_sample.csv")))
    log_path.write_text(json.dumps(log, indent=2))
    write_data_card(pd.read_csv(CARD_DIR / "monthly_post_counts.csv"), log)
    print(f"mainstream fill done, posts now {len(posts)}, failures {len(failures)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=None, help="Exclusive end date YYYY-MM-DD (default: tomorrow UTC)")
    parser.add_argument("--min-interval", type=float, default=1.1)
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Re-request months already marked count_status=timeout in the counts CSV, then rewrite the data card.",
    )
    parser.add_argument(
        "--fill-mainstream",
        action="store_true",
        help="Add a 2018-2026 keyword sample for r/cars, r/Autos, and r/carbuying. Does not abort a subreddit after early timeouts.",
    )
    args = parser.parse_args()
    client = ArcticClient(CACHE_PATH, min_interval=args.min_interval)
    if args.retry_timeouts:
        retry_timeouts(client)
    if args.fill_mainstream:
        fill_mainstream(client)
    if args.retry_timeouts or args.fill_mainstream:
        return
    today = _today_utc()
    end = date.fromisoformat(args.end) if args.end else today + timedelta(days=1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    client = ArcticClient(CACHE_PATH, min_interval=args.min_interval)
    failures: list[dict] = []

    print(f"Counting core subreddits {RANGE_START} .. {end}", flush=True)
    # A quarter or a year times out on active subreddits. One calendar month
    # returned in a few seconds for r/electricvehicles and r/teslamotors
    # (checked 16 Oct 2026). Every count request is therefore one month.
    count_rows: list[dict] = []
    for sub in CORE_SUBREDDITS:
        print(f"counts {sub}", flush=True)
        cursor = RANGE_START
        while cursor < end:
            if cursor.month == 12:
                span_end = date(cursor.year + 1, 1, 1)
            else:
                span_end = date(cursor.year, cursor.month + 1, 1)
            span_end = min(span_end, end)
            try:
                count_rows.extend(fetch_monthly_counts(client, sub, cursor, span_end))
            except Exception as exc:  # noqa: BLE001
                failures.append({"subreddit": sub, "span": [cursor.isoformat(), span_end.isoformat()], "error": str(exc)})
                print(f"  FAILED {sub} {cursor}: {exc}", flush=True)
            if cursor.month == 1 or cursor.month == 7:
                print(f"  {sub} reached {cursor.isoformat()}", flush=True)
                client.save()
            cursor = span_end
        client.save()

    counts = pd.DataFrame(count_rows)
    if not counts.empty:
        counts = counts.drop_duplicates(["subreddit", "month"], keep="last")
        counts = counts.sort_values(["subreddit", "month"])
    counts_path = CARD_DIR / "monthly_post_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"wrote {counts_path} rows={len(counts)}", flush=True)

    print("Sampling core submissions", flush=True)
    post_rows: list[dict] = []
    # Quarterly 7-day windows, plus a fixed week around each coded event.
    event_windows = [
        ("ira_2022", date(2022, 8, 14), date(2022, 8, 21)),
        ("tesla_price_cut_2023", date(2023, 1, 12), date(2023, 1, 19)),
        ("nacs_ford", date(2023, 5, 24), date(2023, 5, 31)),
        ("nacs_gm", date(2023, 6, 7), date(2023, 6, 14)),
        ("credit_law_2025", date(2025, 7, 3), date(2025, 7, 10)),
        ("credit_deadline_2025", date(2025, 9, 28), date(2025, 10, 5)),
    ]
    for sub in CORE_SUBREDDITS:
        print(f"sample {sub}", flush=True)
        for year in range(RANGE_START.year, end.year + 1):
            for quarter in (1, 2, 3, 4):
                window = _quarter_window(sub, year, quarter, today)
                if window is None:
                    continue
                try:
                    got = fetch_posts(client, sub, window[0], window[1], limit=25)
                except RuntimeError as exc:
                    failures.append({"subreddit": sub, "window": [window[0].isoformat(), window[1].isoformat()], "error": str(exc)})
                    continue
                for row in got:
                    row["community_group"] = "ev_community"
                    row["sample_stratum"] = f"{year}Q{quarter}"
                post_rows.extend(got)
        for event_id, after, before in event_windows:
            if after >= end:
                continue
            try:
                got = fetch_posts(client, sub, after, min(before, end), limit=40)
            except RuntimeError as exc:
                failures.append({"subreddit": sub, "event": event_id, "error": str(exc)})
                continue
            for row in got:
                row["community_group"] = "ev_community"
                row["sample_stratum"] = event_id
            post_rows.extend(got)
        client.save()

    print("Sampling keyword-filtered mainstream subs", flush=True)
    keyword_fields: dict[str, str] = {}
    probe_after, probe_before = date(2024, 6, 3), date(2024, 6, 6)
    for sub in GENERAL_SUBREDDITS:
        print(f"sample {sub}", flush=True)
        chosen = None
        for field in ("query", "title"):
            try:
                probe = fetch_posts(
                    client, sub, probe_after, probe_before, limit=5, query="tesla", query_field=field
                )
            except RuntimeError as exc:
                failures.append({"subreddit": sub, "probe_field": field, "error": str(exc)})
                print(f"  probe {field} failed: {exc}", flush=True)
                continue
            chosen = field
            for row in probe:
                row["community_group"] = "mainstream_car"
                row["sample_stratum"] = "probe_2024Q2"
            post_rows.extend(probe)
            break
        keyword_fields[sub] = chosen or "unavailable"
        if chosen is None:
            print(f"  no working keyword field for {sub}", flush=True)
            continue
        print(f"  using {chosen}=", flush=True)
        timeouts = 0
        for year in range(RANGE_START.year, end.year + 1):
            if timeouts >= 3:
                failures.append({"subreddit": sub, "error": "stopped keyword sampling after 3 timeouts"})
                break
            for quarter in (1, 2, 3, 4):
                window = _quarter_window(sub, year, quarter, today)
                if window is None or timeouts >= 3:
                    continue
                for keyword in GENERAL_KEYWORDS:
                    try:
                        got = fetch_posts(
                            client,
                            sub,
                            window[0],
                            window[1],
                            limit=12,
                            query=keyword,
                            query_field=chosen,
                        )
                    except RuntimeError as exc:
                        failures.append(
                            {
                                "subreddit": sub,
                                "keyword": keyword,
                                "field": chosen,
                                "window": [window[0].isoformat(), window[1].isoformat()],
                                "error": str(exc),
                            }
                        )
                        if "Timeout" in str(exc) or "429" in str(exc):
                            timeouts += 1
                            break
                        continue
                    for row in got:
                        row["community_group"] = "mainstream_car"
                        row["sample_stratum"] = f"{year}Q{quarter}"
                    post_rows.extend(got)
        client.save()

    comment_probe = _probe_comment(client)
    client.save()

    posts = _records_to_frame(post_rows)
    if not posts.empty:
        posts = posts.drop_duplicates("id")
    full_path = RAW_DIR / "posts_downloaded.csv"
    posts.to_csv(full_path, index=False)
    public = write_public_sample(posts, SAMPLE_DIR / "posts_sample.csv")

    by_sub = {}
    if not posts.empty:
        by_sub = posts.groupby("subreddit").size().astype(int).to_dict()
    log = {
        "source": "Arctic Shift API",
        "api_base": "https://arctic-shift.photon-reddit.com",
        "api_docs": "https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_start": RANGE_START.isoformat(),
        "range_end_exclusive": end.isoformat(),
        "latest_observed_created_utc": None if posts.empty else int(posts["created_utc"].max()),
        "core_subreddits": list(CORE_SUBREDDITS),
        "general_subreddits": list(GENERAL_SUBREDDITS),
        "general_keywords": list(GENERAL_KEYWORDS),
        "general_keyword_field": keyword_fields,
        "unit": "reddit submission (post). Comments were probed, not censused.",
        "comment_probe": comment_probe,
        "count_rows": int(len(counts)),
        "count_timeouts": int((counts["count_status"] == "timeout").sum()) if not counts.empty else None,
        "downloaded_posts_deduped": int(len(posts)),
        "downloaded_posts_by_subreddit": by_sub,
        "public_sample_rows": int(len(public)),
        "network_requests": client.n_network,
        "cache_hits": client.n_cache,
        "failures": failures,
        "sampling": {
            "core_counts": "aggregate created_utc frequency=month, one calendar month per request",
            "core_text": "one deterministic 7-day window per subreddit-quarter, limit 25, plus a fixed week around each coded event, limit 40",
            "mainstream_text": "same windows, separate keyword queries, limit 15, de-duplicated by id",
            "not_a_simple_random_sample": True,
        },
        "score_note": "Arctic Shift documents that score and num_comments stay at 0 or 1 until the post is about 36 hours old.",
    }
    (CARD_DIR / "download_log.json").write_text(json.dumps(log, indent=2))
    write_data_card(counts, log)
    print(json.dumps({k: log[k] for k in ["downloaded_posts_deduped", "downloaded_posts_by_subreddit", "count_rows", "count_timeouts", "network_requests"]}, indent=2)[:4000])
    print(f"failures={len(failures)} public_sample={len(public)}", flush=True)


if __name__ == "__main__":
    main()
