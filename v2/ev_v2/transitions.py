"""Stage-to-stage transitions for authors who post more than once.

The current small-model file has one ownership stage. Onboarding and
long-term ownership are not separated here. A forward move is a later post
at a higher rung of the mapped ladder. Repeat-author coverage is the
feasibility check: a week sample will contain few people twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_v2.taxonomy import SMALL_MODEL_STAGE

# The small model stops at a single ownership rung, after purchase.
MAPPED_ORDER = {
    "awareness": 1,
    "consideration": 2,
    "evaluation": 3,
    "purchase": 4,
    "ownership": 5,
}


def map_stage(stage_label: str | None) -> str:
    return SMALL_MODEL_STAGE.get(str(stage_label or ""), "unclear")


def author_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per audience, plus an all-rows row. Empty hashes are dropped."""
    work = frame.copy()
    work["author_hash"] = work["author_hash"].fillna("").astype(str)
    work = work[work["author_hash"] != ""]
    rows = []
    groups = [("all", work)]
    if "audience" in work.columns:
        for audience, block in work.groupby("audience"):
            groups.append((str(audience), block))
    for name, block in groups:
        if block.empty:
            rows.append(_coverage_row(name, block))
            continue
        counts = block.groupby("author_hash").size()
        repeat_authors = counts[counts >= 2]
        rows.append(
            {
                "audience": name,
                "n_posts": int(len(block)),
                "n_authors": int(counts.size),
                "n_repeat_authors": int(repeat_authors.size),
                "n_posts_from_repeat_authors": int(repeat_authors.sum()) if len(repeat_authors) else 0,
                "share_posts_from_repeat_authors": float(repeat_authors.sum() / len(block)) if len(block) else 0.0,
                "max_posts_per_author": int(counts.max()) if len(counts) else 0,
                "median_posts_per_author": float(counts.median()) if len(counts) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _coverage_row(name: str, block: pd.DataFrame) -> dict:
    return {
        "audience": name,
        "n_posts": int(len(block)),
        "n_authors": 0,
        "n_repeat_authors": 0,
        "n_posts_from_repeat_authors": 0,
        "share_posts_from_repeat_authors": 0.0,
        "max_posts_per_author": 0,
        "median_posts_per_author": 0.0,
    }


def consecutive_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per pair of successive posts by the same author."""
    work = frame.copy()
    work["author_hash"] = work["author_hash"].fillna("").astype(str)
    work = work[work["author_hash"] != ""]
    work["created_utc"] = pd.to_numeric(work["created_utc"], errors="coerce")
    work = work.dropna(subset=["created_utc"])
    work["stage"] = work["stage_label"].map(map_stage) if "stage" not in work.columns else work["stage"]
    work["rung"] = work["stage"].map(lambda value: MAPPED_ORDER.get(str(value), np.nan))
    work = work.sort_values(["author_hash", "created_utc", "id"] if "id" in work.columns else ["author_hash", "created_utc"])
    rows = []
    for author, block in work.groupby("author_hash", sort=False):
        if len(block) < 2:
            continue
        previous = None
        for row in block.itertuples(index=False):
            current = row
            if previous is not None:
                rows.append(
                    {
                        "author_hash": author,
                        "from_stage": previous.stage,
                        "to_stage": current.stage,
                        "from_rung": previous.rung,
                        "to_rung": current.rung,
                        "from_utc": float(previous.created_utc),
                        "to_utc": float(current.created_utc),
                        "gap_days": (float(current.created_utc) - float(previous.created_utc)) / 86400.0,
                        "forward": bool(pd.notna(previous.rung) and pd.notna(current.rung) and current.rung > previous.rung),
                        "audience": getattr(current, "audience", ""),
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def forward_rate_around(pairs: pd.DataFrame, cutoff_utc: float, window_days: int = 180) -> dict:
    """Share of pairs that cross the cutoff and move to a higher rung.

    The pair's first post is inside the window before the cutoff and the
    second is inside the window after. This is a description of the labeled
    file, not a hazard model with covariates.
    """
    if pairs.empty:
        return {"n_pairs": 0, "n_forward": 0, "forward_rate": float("nan")}
    window = window_days * 86400.0
    cross = pairs[
        (pairs["from_utc"] < cutoff_utc)
        & (pairs["from_utc"] >= cutoff_utc - window)
        & (pairs["to_utc"] >= cutoff_utc)
        & (pairs["to_utc"] < cutoff_utc + window)
    ]
    n = int(len(cross))
    k = int(cross["forward"].sum()) if n else 0
    return {"n_pairs": n, "n_forward": k, "forward_rate": (k / n) if n else float("nan")}


def transition_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["from_stage", "to_stage", "n"])
    counts = pairs.groupby(["from_stage", "to_stage"], dropna=False).size().reset_index(name="n")
    return counts.sort_values("n", ascending=False)
