"""Sampled-week blocks for the enthusiast text file.

The downloader keeps one 7-day window per subreddit-quarter, plus a fixed
week around each coded event. Posts inside an event week use that week as
the block, including when the post also falls in a quarterly window.
Everyone else is assigned the quarterly window that contains the timestamp.
A timestamp that matches neither is blocked by subreddit and month so it
still has a cluster, and the caller can count those fallbacks.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

# Same windows as the sampling loop in download_reddit.py. The end date is
# exclusive, matching Arctic Shift ``before``.
SAMPLE_EVENT_WINDOWS: tuple[tuple[str, date, date], ...] = (
    ("ira_2022", date(2022, 8, 14), date(2022, 8, 21)),
    ("tesla_price_cut_2023", date(2023, 1, 12), date(2023, 1, 19)),
    ("nacs_ford", date(2023, 5, 24), date(2023, 5, 31)),
    ("nacs_gm", date(2023, 6, 7), date(2023, 6, 14)),
    ("credit_law_2025", date(2025, 7, 3), date(2025, 7, 10)),
    ("credit_deadline_2025", date(2025, 9, 28), date(2025, 10, 5)),
)

# The week sample was drawn on this day. Later dates would clip 2026 differently.
SAMPLE_AS_OF = date(2026, 10, 16)


def parse_created_utc(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp):
        return None
    if stamp > 10_000_000_000:
        stamp = stamp / 1000.0
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def assign_week_block(subreddit: str, created_utc, as_of: date = SAMPLE_AS_OF) -> str:
    """Block id for one enthusiast submission."""
    moment = parse_created_utc(created_utc)
    if moment is None:
        return f"{subreddit}|unknown"
    day = moment.date()
    for event_id, start, end in SAMPLE_EVENT_WINDOWS:
        if start <= day < end:
            return f"{subreddit}|{event_id}|{start.isoformat()}"
    quarter = (day.month - 1) // 3 + 1
    window = _quarter_window(subreddit, day.year, quarter, as_of)
    if window is not None and window[0] <= day < window[1]:
        return f"{subreddit}|{day.year}Q{quarter}|{window[0].isoformat()}"
    return f"{subreddit}|month|{day.strftime('%Y-%m')}"


def _quarter_window(subreddit: str, year: int, quarter: int, today: date) -> tuple[date, date] | None:
    """Copy of the downloader's window rule, kept local to avoid import cycles."""
    start_month = (quarter - 1) * 3 + 1
    q_start = date(year, start_month, 1)
    if start_month == 10:
        q_end = date(year + 1, 1, 1)
    else:
        q_end = date(year, start_month + 3, 1)
    if q_start > today:
        return None
    # Imported lazily so a circular import through download_reddit cannot start.
    import hashlib

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
