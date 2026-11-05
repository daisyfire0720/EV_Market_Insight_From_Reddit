"""Budget and request list for the thin enthusiast baseline fill.

Round 5 left four enthusiast months short: August 2021 (the IRA year-ago
cell, 23 posts), June 2022 (missing from the IRA prior-three and the NACS
year-ago window), December 2022 (missing from the Tesla price-cut
prior-three), and September 2024 (missing from the credit-deadline
year-ago window). This module only describes the pull. It does not call
the network.

The cap is three 7-day windows per month, 40 posts per subreddit-window,
seven subreddits. That is 3,360 posts at most. At about 140 posts a minute
on CPU the labeling step is under half an hour, and the script stops at
90 minutes even if the pace is slower. The full months are about 24,000
posts and are not pulled.
"""

from __future__ import annotations

from datetime import date

from ev_v2.arctic import CORE_SUBREDDITS
from ev_v2.blocks import FILL_MONTHS, month_fill_windows

# Same page cap the event weeks used.
FILL_LIMIT = 40
LABEL_BUDGET_MINUTES = 90


def planned_windows() -> list[tuple[str, str, date, date]]:
    """(subreddit, month, start, end) in a stable order. End is exclusive."""
    rows = []
    for month in FILL_MONTHS:
        year_s, month_s = month.split("-")
        year, mon = int(year_s), int(month_s)
        for start, end in month_fill_windows(year, mon):
            for subreddit in CORE_SUBREDDITS:
                rows.append((subreddit, month, start, end))
    return rows


def max_posts() -> int:
    """Upper bound on posts the API is asked to return."""
    return len(planned_windows()) * FILL_LIMIT
