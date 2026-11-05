"""How many authors sit on both sides of a cutoff once comments are added.

A submission-only count is the robustness check. Comments are a separate
text type. This module does not download anything and does not invent text.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ev_v2.rdit import CUTOFFS

WINDOW_DAYS = 180


def _cutoff_utc(day) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def authors_on_both_sides(
    events: pd.DataFrame,
    cutoff_utc: float,
    window_days: int = WINDOW_DAYS,
    text_types: tuple[str, ...] | None = None,
) -> set[str]:
    """Authors with at least one item before the cutoff and one after, inside the window."""
    if events.empty or "author_hash" not in events.columns or "created_utc" not in events.columns:
        return set()
    work = events.copy()
    work["author_hash"] = work["author_hash"].fillna("").astype(str)
    work["created_utc"] = pd.to_numeric(work["created_utc"], errors="coerce")
    work = work[(work["author_hash"] != "") & work["created_utc"].notna()]
    if text_types is not None and "text_type" in work.columns:
        work = work[work["text_type"].isin(text_types)]
    window = window_days * 86400.0
    work = work[(work["created_utc"] >= cutoff_utc - window) & (work["created_utc"] < cutoff_utc + window)]
    before = set(work.loc[work["created_utc"] < cutoff_utc, "author_hash"])
    after = set(work.loc[work["created_utc"] >= cutoff_utc, "author_hash"])
    return before & after


def coverage_table(events: pd.DataFrame, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """One row per cutoff. ``n_gained`` is authors who appear on both sides only after comments are included."""
    rows = []
    if "text_type" not in events.columns:
        events = events.copy()
        events["text_type"] = "submission"
    for spec in CUTOFFS:
        cutoff = _cutoff_utc(spec["cutoff"])
        submissions = authors_on_both_sides(events, cutoff, window_days, ("submission",))
        both = authors_on_both_sides(events, cutoff, window_days, None)
        rows.append(
            {
                "cutoff_id": spec["cutoff_id"],
                "cutoff": spec["cutoff"].isoformat(),
                "window_days": window_days,
                "n_authors_submissions_only": len(submissions),
                "n_authors_submissions_and_comments": len(both),
                "n_gained": len(both - submissions),
                "validated": False,
            }
        )
    return pd.DataFrame(rows)
