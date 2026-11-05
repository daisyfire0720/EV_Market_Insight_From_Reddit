"""Keyword census and text pull for mainstream car subreddits.

Counts use one combined Arctic Shift query per month:

    tesla OR rivian OR ioniq OR "electric vehicle" OR "electric car"

On 16 October 2026 that query returned 80 posts for r/cars in February 2016,
against 53 for ``tesla`` alone, so the ``OR`` is applied. A year-long aggregate
on the same subreddit timed out. Windows therefore start at one calendar month
and split on a real query timeout. Fast 422 responses are retried as rate limits
inside the client.

Text does not use that combined query. On 16 October 2026 a search for the
combined query on r/cars in January 2015 returned HTTP 200 and zero posts,
while the aggregate for the same query returned 71. Search is therefore one
keyword at a time (``tesla``, ``rivian``, ``ioniq``, ``electric vehicle``,
``electric car``). Empty count months are not searched again.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ev_v2.arctic import GENERAL_KEYWORDS, GENERAL_SUBREDDITS, POSTS_AGGREGATE, POSTS_SEARCH
from ev_v2.retry import Checkpoint, FetchError, resolve_window

MAINSTREAM_QUERY = 'tesla OR rivian OR ioniq OR "electric vehicle" OR "electric car"'


def text_queries() -> tuple[str, ...]:
    """Keywords for the text pull. The combined OR query is counts-only."""
    return GENERAL_KEYWORDS
MAINSTREAM_START = date(2015, 1, 1)
MAINSTREAM_END = date(2026, 10, 1)  # exclusive: through September 2026
DUMP_INDEX = "https://github.com/ArthurHeitmann/arctic_shift/blob/master/download_links.md"

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


def month_spans(start: date, end: date) -> list[tuple[datetime, datetime]]:
    spans = []
    cursor = date(start.year, start.month, 1)
    while cursor < end:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        stop = min(nxt, end)
        spans.append((_dt(cursor), _dt(stop)))
        cursor = nxt
    return spans


def _dt(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def api_time(moment: datetime) -> str:
    moment = moment.astimezone(timezone.utc)
    if moment.hour == moment.minute == moment.second == moment.microsecond == 0:
        return moment.date().isoformat()
    return str(int(moment.timestamp()))


def count_from_aggregate(payload: dict) -> int:
    total = 0
    for item in payload.get("data") or []:
        total += int(item["count"])
    return total


def run_census(client, checkpoint: Checkpoint, retry_failed: bool = False) -> pd.DataFrame:
    rows = []
    spans = month_spans(MAINSTREAM_START, MAINSTREAM_END)
    for sub in GENERAL_SUBREDDITS:
        print(f"mainstream census {sub} ({len(spans)} months)", flush=True)
        for start, end in spans:
            def fetch(win_start, win_end, subreddit=sub):
                payload = client.get(
                    POSTS_AGGREGATE,
                    {
                        "subreddit": subreddit,
                        "query": MAINSTREAM_QUERY,
                        "after": api_time(win_start),
                        "before": api_time(win_end),
                        "aggregate": "created_utc",
                        "frequency": "month",
                    },
                )
                return count_from_aggregate(payload)

            outcome = resolve_window(
                kind="agg",
                subreddit=sub,
                query=MAINSTREAM_QUERY,
                start=start,
                end=end,
                fetch=fetch,
                checkpoint=checkpoint,
                min_span=timedelta(days=1),
                retry_failed=retry_failed,
            )
            month = start.strftime("%Y-%m")
            print(f"  {sub} {month} {outcome.status} n={outcome.n}", flush=True)
            rows.append(
                {
                    "subreddit": sub,
                    "month": month,
                    "year": start.year,
                    "n_posts": outcome.n,
                    "count_status": outcome.status,
                    "query": MAINSTREAM_QUERY,
                }
            )
        client.save()
    frame = pd.DataFrame(rows)
    return frame


def _page_posts(client, subreddit: str, start: datetime, end: datetime, query: str, limit: int = 100) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        payload = client.get(
            POSTS_SEARCH,
            {
                "subreddit": subreddit,
                "query": query,
                "after": api_time(cursor),
                "before": api_time(end),
                "limit": str(limit),
                "sort": "asc",
                "fields": POST_FIELDS,
            },
        )
        page = payload.get("data") or []
        for row in page:
            row["query"] = query
            row["sample_after"] = api_time(start)
            row["sample_before"] = api_time(end)
            row["community_group"] = "mainstream_car"
            row["sample_stratum"] = f"keyword_{start.strftime('%Y-%m')}"
            row["sample_truncated"] = False
        rows.extend(page)
        if len(page) < limit:
            return rows
        last = max(float(row["created_utc"]) for row in page)
        nxt = datetime.fromtimestamp(last + 1, tz=timezone.utc)
        if nxt <= cursor or nxt >= end:
            raise FetchError("full page and cursor did not advance", kind="timeout")
        cursor = nxt
    return rows


def run_text(
    client,
    checkpoint: Checkpoint,
    counts: pd.DataFrame,
    jsonl_path: Path,
    retry_failed: bool = False,
) -> int:
    """Download keyword posts for months with a positive count. Returns new row count.

    The combined OR query is not sent to the search endpoint. Each keyword is
    its own checkpointed window.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stale = [key for key in list(checkpoint.windows) if key.startswith("search|") and MAINSTREAM_QUERY in key]
    for key in stale:
        del checkpoint.windows[key]
    if stale:
        checkpoint.save()
        print(f"dropped {len(stale)} combined-query search checkpoints", flush=True)
    wanted = counts[counts["count_status"].isin(["ok", "partial"])].copy()
    wanted["n_posts"] = pd.to_numeric(wanted["n_posts"], errors="coerce").fillna(0)
    wanted = wanted[wanted["n_posts"] > 0]
    written = 0
    handle = jsonl_path.open("a", encoding="utf-8")
    try:
        for sub in GENERAL_SUBREDDITS:
            sub_rows = wanted[wanted["subreddit"] == sub]
            print(f"mainstream text {sub} ({len(sub_rows)} months with posts)", flush=True)
            for keyword in text_queries():
                print(f"  keyword {keyword}", flush=True)
                for _, row in sub_rows.iterrows():
                    start = _dt(date.fromisoformat(f"{row['month']}-01"))
                    if start.month == 12:
                        end = _dt(date(start.year + 1, 1, 1))
                    else:
                        end = _dt(date(start.year, start.month + 1, 1))
                    fetched_now = {"n": 0}

                    def fetch(win_start, win_end, subreddit=sub, counter=fetched_now, term=keyword):
                        posts = _page_posts(client, subreddit, win_start, win_end, term)
                        for post in posts:
                            handle.write(json.dumps(post, default=str) + "\n")
                        handle.flush()
                        counter["n"] += len(posts)
                        return len(posts)

                    outcome = resolve_window(
                        kind="search",
                        subreddit=sub,
                        query=keyword,
                        start=start,
                        end=end,
                        fetch=fetch,
                        checkpoint=checkpoint,
                        min_span=timedelta(days=1),
                        retry_failed=retry_failed,
                    )
                    written += fetched_now["n"]
                    if outcome.status != "ok_empty" or fetched_now["n"]:
                        print(
                            f"  {sub} {row['month']} {keyword} {outcome.status} fetched_now={fetched_now['n']} window_n={outcome.n}",
                            flush=True,
                        )
            client.save()
    finally:
        handle.close()
    return written


def yearly_table(counts: pd.DataFrame) -> pd.DataFrame:
    work = counts.copy()
    work["n_posts"] = pd.to_numeric(work["n_posts"], errors="coerce")
    grouped = (
        work.groupby(["subreddit", "year"], as_index=False)
        .agg(
            n_posts=("n_posts", lambda s: s.sum(min_count=1)),
            months=("month", "nunique"),
            months_ok=("count_status", lambda s: int(s.isin(["ok", "ok_empty"]).sum())),
            months_partial=("count_status", lambda s: int((s == "partial").sum())),
            months_failed=("count_status", lambda s: int((s == "failed").sum())),
            missing_months=("n_posts", lambda s: int(s.isna().sum())),
        )
    )
    return grouped


def render_census_markdown(counts: pd.DataFrame, failures: list[dict], text_note: str) -> str:
    yearly = yearly_table(counts) if not counts.empty else pd.DataFrame()
    lines = [
        "## Mainstream keyword census",
        "",
        "Submissions in r/cars, r/Autos, and r/carbuying whose title or selftext matches the combined query "
        f"`{MAINSTREAM_QUERY}`. Range: {MAINSTREAM_START.isoformat()} inclusive through {MAINSTREAM_END.isoformat()} exclusive "
        "(through September 2026). This is a count of matching submissions, not a sum of separate keyword queries.",
        "",
        "Each month is one aggregate request. A fast HTTP 422 waits out `X-RateLimit-Reset` and retries. A slow HTTP 422 is a query timeout: the window is split down to one day. Failed windows are checkpointed and listed below.",
        "",
        "The text pull does not reuse this combined query. A search for it on r/cars, January 2015, returned HTTP 200 and zero posts on 16 October 2026, while the aggregate returned 71. Text is requested once per keyword (`tesla`, `rivian`, `ioniq`, `electric vehicle`, `electric car`), then de-duplicated by post id.",
        "",
        text_note,
        "",
    ]
    if counts.empty:
        lines.append("No mainstream count rows were written.")
        lines.append("")
    else:
        n_fail = int((counts["count_status"] == "failed").sum())
        n_partial = int((counts["count_status"] == "partial").sum())
        known = pd.to_numeric(counts["n_posts"], errors="coerce")
        lines.append(
            f"Subreddit-months requested: {len(counts)}. Failed months: {n_fail}. Partial months (a lower bound from the splits that returned): {n_partial}. "
            f"Sum of returned counts, with failed months omitted: **{int(known.fillna(0).sum())}**."
        )
        lines.append("")
        lines.append("| Subreddit | Year | Posts | Months ok | Partial | Failed |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        if not yearly.empty:
            for _, row in yearly.sort_values(["subreddit", "year"]).iterrows():
                lines.append(
                    f"| r/{row['subreddit']} | {int(row['year'])} | {int(row['n_posts']) if pd.notna(row['n_posts']) else ''} | {int(row['months_ok'])} | {int(row['months_partial'])} | {int(row['months_failed'])} |"
                )
        lines.append("")
    lines.append(f"Failed windows in the checkpoint: {len(failures)}.")
    if failures:
        lines.append("")
        lines.append("If a month is still failed after the split, Arctic Shift did not serve that keyword window. The maintainer's own note is to use the monthly dumps for large pulls. The dump index, which lists Academic Torrents files through 2026-09, is "
                      f"[download_links.md]({DUMP_INDEX}). Reddit's official API is the other legitimate route and needs OAuth credentials this project does not have.")
    else:
        lines.append("")
        lines.append("Every requested mainstream month returned a count or a counted split. The text file can still be short of that census where search pages timed out; those windows are in the same failure list when the text pass has run.")
    lines.append("")
    return "\n".join(lines)


def upsert_census_section(card_path: Path, section: str) -> None:
    start = "<!-- mainstream-census-start -->"
    end = "<!-- mainstream-census-end -->"
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    existing = card_path.read_text() if card_path.exists() else ""
    if start in existing and end in existing:
        head, rest = existing.split(start, 1)
        _, tail = rest.split(end, 1)
        card_path.write_text(head.rstrip() + "\n\n" + block + tail.lstrip("\n"))
        return
    card_path.write_text(existing.rstrip() + "\n\n" + block)
