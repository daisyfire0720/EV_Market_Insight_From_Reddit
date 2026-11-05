"""Helpers for the Arctic Shift API that do not need the network.

Aggregate buckets are labeled at the start of each month in
Europe/Amsterdam (UTC+1 in winter, UTC+2 in summer). The two examples
checked on 16 Oct 2026 were ``2023-12-31T23:00:00Z`` for January 2024 and
``2024-05-31T22:00:00Z`` for June 2024.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

AMSTERDAM = ZoneInfo("Europe/Amsterdam")

API_BASE = "https://arctic-shift.photon-reddit.com"
POSTS_SEARCH = f"{API_BASE}/api/posts/search"
POSTS_AGGREGATE = f"{API_BASE}/api/posts/search/aggregate"
COMMENTS_SEARCH = f"{API_BASE}/api/comments/search"

# Dedicated EV communities: every submission is in scope.
CORE_SUBREDDITS = (
    "electricvehicles",
    "teslamotors",
    "BoltEV",
    "Rivian",
    "Ioniq5",
    "KiaEV6",
    "evcharging",
)

# Mainstream car communities: kept only when a keyword search hits.
GENERAL_SUBREDDITS = (
    "cars",
    "Autos",
    "carbuying",
)

# Post keyword search on Arctic Shift is a simple keyword search and times
# out on very active subreddits. These phrases are issued as separate
# queries and de-duplicated by post id. Bare "ev" is intentionally absent.
GENERAL_KEYWORDS = (
    "tesla",
    "rivian",
    "ioniq",
    "electric vehicle",
    "electric car",
)


def bucket_month(created_utc: str) -> str:
    """Map an aggregate ``created_utc`` label to ``YYYY-MM``."""
    stamp = created_utc.replace("Z", "+00:00")
    dt = datetime.fromisoformat(stamp).astimezone(AMSTERDAM)
    return f"{dt.year:04d}-{dt.month:02d}"
