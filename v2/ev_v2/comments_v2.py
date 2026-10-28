"""Comments v2: a balanced draw around each cutoff, with the parent post as context.

This follows ``COMMENTS_V2_PLAN.md``. Nothing here calls a model.

- Each comment is assigned to one cutoff only, the nearest one. A tie (the
  day halfway between 1 and 13 January 2023) goes to the earlier cutoff.
- The cutoff day itself is left out of both sides.
- Audience comes from the parent submission when it is in the framing file,
  and otherwise from the subreddit lists.
- Inside each cutoff x audience x side the order is a seeded random
  permutation, never a ranking by distance to the cutoff. The cap is the same
  on the left and the right, and is lowered to the shorter side's count.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ev_v2.after_llm import audience_from_subreddit, parent_key
from ev_v2.rdit import CUTOFFS, _as_day

WINDOW_DAYS = 180
CAP_PER_SIDE = 500
SEED = 20261028
BOT_MARKERS = (
    "i am a bot, and this action was performed automatically",
    "this action was performed automatically",
)


def nearest_cutoff(day: date | None) -> tuple[str, int] | None:
    """(cutoff_id, signed days from it) for the nearest cutoff; ties go to the earlier cutoff."""
    if day is None:
        return None
    best = None
    for spec in CUTOFFS:
        delta = (day - spec["cutoff"]).days
        key = (abs(delta), spec["cutoff"].toordinal())
        if best is None or key < best[0]:
            best = (key, spec["cutoff_id"], delta)
    return best[1], best[2]


def assign_candidates(meta: pd.DataFrame, parent_audience: dict[str, str], window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Every comment that can enter the draw, with its cutoff, side, and audience."""
    work = meta.copy()
    work["id"] = work["id"].astype(str)
    if "text_type" in work.columns:
        work = work[work["text_type"].astype(str).str.lower().eq("comment")]
    work = work.drop_duplicates("id")
    cutoff_ids: list[str] = []
    deltas: list[float] = []
    for value in work["created_utc"]:
        found = nearest_cutoff(_as_day(value))
        if found is None:
            cutoff_ids.append("")
            deltas.append(np.nan)
        else:
            cutoff_ids.append(found[0])
            deltas.append(found[1])
    work["draw_cutoff_id"] = cutoff_ids
    work["delta_days"] = deltas
    audiences: list[str] = []
    sources: list[str] = []
    links = work["link_id"] if "link_id" in work.columns else pd.Series("", index=work.index)
    for link, sub in zip(links, work["subreddit"]):
        aud = parent_audience.get(parent_key(link), "")
        if aud in {"enthusiast", "shopper"}:
            audiences.append(aud)
            sources.append("parent_link")
            continue
        aud = audience_from_subreddit(sub)
        audiences.append(aud)
        sources.append("subreddit" if aud else "")
    work["audience"] = audiences
    work["audience_source"] = sources
    keep = (
        work["draw_cutoff_id"].ne("")
        & work["delta_days"].notna()
        & work["delta_days"].abs().le(window_days)
        & work["delta_days"].ne(0)
        & work["audience"].isin(["enthusiast", "shopper"])
    )
    work = work[keep].copy()
    work["delta_days"] = work["delta_days"].astype(int)
    work["draw_side"] = np.where(work["delta_days"] < 0, "left", "right")
    return work.reset_index(drop=True)


def _cells():
    for index, spec in enumerate(CUTOFFS):
        for a_index, audience in enumerate(("enthusiast", "shopper")):
            for s_index, side in enumerate(("left", "right")):
                yield (index * 4 + a_index * 2 + s_index), spec["cutoff_id"], audience, side


def ordered_sides(candidates: pd.DataFrame, seed: int = SEED) -> dict[tuple[str, str, str], list[str]]:
    """Seeded random order of ids inside each cutoff x audience x side. Not ranked by distance."""
    out: dict[tuple[str, str, str], list[str]] = {}
    for cell_index, cutoff_id, audience, side in _cells():
        block = candidates[
            candidates["draw_cutoff_id"].eq(cutoff_id)
            & candidates["audience"].eq(audience)
            & candidates["draw_side"].eq(side)
        ]
        ids = sorted(block["id"].astype(str))
        rng = np.random.default_rng([seed, cell_index])
        order = rng.permutation(len(ids))
        out[(cutoff_id, audience, side)] = [ids[i] for i in order]
    return out


def is_usable_body(body) -> bool:
    if body is None or (isinstance(body, float) and body != body):
        return False
    text = str(body).strip()
    if not text or text.lower() in {"[removed]", "[deleted]", "nan"}:
        return False
    low = text.lower()
    return not any(marker in low for marker in BOT_MARKERS)


def balanced_draw(
    order: dict[tuple[str, str, str], list[str]],
    usable: set[str] | None = None,
    cap: int = CAP_PER_SIDE,
) -> tuple[list[tuple[str, str, str, str]], pd.DataFrame]:
    """Equal counts on each side of each cutoff x audience.

    ``usable`` limits the draw to ids with a usable body; ``None`` treats every
    id as usable. Returns (id, cutoff_id, audience, side) rows and a count table.
    """
    rows: list[tuple[str, str, str, str]] = []
    counts = []
    for spec in CUTOFFS:
        cutoff_id = spec["cutoff_id"]
        for audience in ("enthusiast", "shopper"):
            sides = {}
            for side in ("left", "right"):
                ids = order.get((cutoff_id, audience, side), [])
                sides[side] = (ids, [i for i in ids if usable is None or i in usable])
            n = min(cap, len(sides["left"][1]), len(sides["right"][1]))
            for side in ("left", "right"):
                for doc_id in sides[side][1][:n]:
                    rows.append((doc_id, cutoff_id, audience, side))
            counts.append(
                {
                    "cutoff_id": cutoff_id,
                    "cutoff": spec["cutoff"].isoformat(),
                    "audience": audience,
                    "cap": cap,
                    "n_candidates_left": len(sides["left"][0]),
                    "n_candidates_right": len(sides["right"][0]),
                    "n_usable_left": len(sides["left"][1]),
                    "n_usable_right": len(sides["right"][1]),
                    "n_drawn_per_side": n,
                    "trimmed_to_short_side": bool(n < cap),
                    "validated": False,
                }
            )
    return rows, pd.DataFrame(counts)


def prefix_ids(order: dict[tuple[str, str, str], list[str]], k: int) -> list[str]:
    """The first ``k`` ids of every side, which is all a draw with cap <= k can use when bodies are present."""
    out: list[str] = []
    for ids in order.values():
        out.extend(ids[:k])
    return out


def parent_fields(title, selftext, excerpt_chars: int = 300) -> tuple[str, str]:
    def clean(value) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none", "[removed]", "[deleted]"} else text

    return clean(title), clean(selftext)[:excerpt_chars]
