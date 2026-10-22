"""Shopper-composition checks. Still model labels, still unvalidated.

The keyword filter keeps Tesla, Rivian, and similar mentions inside the
shopper subs, and the stage model calls most of those posts ownership.
Two slices ask whether the shopper contrasts are carried by that mix:

- r/carbuying alone
- posts whose stage label is pre-purchase (learning, comparing, evaluating, buying)
"""

from __future__ import annotations

import pandas as pd

from ev_v2.agreement import STAGE_FROM_MODEL
from ev_v2.framing import (
    ENTHUSIAST,
    PRIMARY_BARRIER,
    SHOPPER,
    audience_gap,
    barrier_column,
    event_contrasts,
    wilson_interval,
)

PRE_PURCHASE_LABELS = (
    "learning about electric vehicles",
    "comparing electric vehicle options",
    "evaluating cost or fit before buying",
    "buying an electric vehicle",
)

PRE_PURCHASE_CODES = tuple(STAGE_FROM_MODEL[label] for label in PRE_PURCHASE_LABELS)


def pre_purchase_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["stage_label"].isin(PRE_PURCHASE_LABELS)


def shopper_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    shop = frame[frame["audience"] == SHOPPER].copy()
    return {
        "all_shopper": shop,
        "carbuying": shop[shop["subreddit"] == "carbuying"].copy(),
        "pre_purchase": shop[pre_purchase_mask(shop)].copy(),
    }


def slice_shares(frame: pd.DataFrame, barriers: list[str]) -> pd.DataFrame:
    """Shares inside each shopper slice, and the enthusiast file for comparison."""
    rows = []
    enth = frame[frame["audience"] == ENTHUSIAST]
    pieces = {"enthusiast": enth, **shopper_slices(frame)}
    measures = list(barriers) + ["owning an electric vehicle", "support for electric vehicles"]
    for name, block in pieces.items():
        labeled = block.copy()
        labeled["audience"] = SHOPPER if name != "enthusiast" else ENTHUSIAST
        # overall_shares expects both audiences. Build one audience at a time.
        shares = _one_audience_shares(labeled, barriers, labeled["audience"].iloc[0] if len(labeled) else SHOPPER)
        shares["slice"] = name
        rows.append(shares)
    out = pd.concat(rows, ignore_index=True)
    # Gap versus the enthusiast file, for the three shopper slices.
    enth_shares = out[out["slice"] == "enthusiast"]
    gap_rows = []
    for name in ("all_shopper", "carbuying", "pre_purchase"):
        block = out[out["slice"] == name]
        combined = pd.concat([enth_shares.assign(audience=ENTHUSIAST), block.assign(audience=SHOPPER)], ignore_index=True)
        for measure in measures:
            gap = audience_gap(combined, measure)
            gap_rows.append({"slice": name, "measure": measure, "gap_shopper_minus_enthusiast": gap["diff"], "gap_lo": gap["lo"], "gap_hi": gap["hi"]})
    gaps = pd.DataFrame(gap_rows)
    return out.merge(gaps, on=["slice", "measure"], how="left")


def _one_audience_shares(frame: pd.DataFrame, barriers: list[str], audience: str) -> pd.DataFrame:
    """overall_shares drops an audience that is absent, so pass a two-audience frame of copies when needed."""
    if frame.empty:
        return pd.DataFrame(columns=["measure", "audience", "n", "k", "share", "lo", "hi"])
    rows = []
    group = frame
    n = int(len(group))
    for barrier in barriers:
        column = barrier_column(barrier)
        k = int(group[column].fillna(False).astype(bool).sum()) if column in group.columns else 0
        share, lo, hi = wilson_interval(k, n)
        rows.append({"measure": barrier, "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
    if "stage_label" in group.columns:
        k = int((group["stage_label"] == "owning an electric vehicle").sum())
        share, lo, hi = wilson_interval(k, n)
        rows.append({"measure": "owning an electric vehicle", "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
    if "stance_label" in group.columns:
        k = int((group["stance_label"] == "support for electric vehicles").sum())
        share, lo, hi = wilson_interval(k, n)
        rows.append({"measure": "support for electric vehicles", "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def slice_contrasts(frame: pd.DataFrame, barriers: list[str]) -> pd.DataFrame:
    """Event contrasts inside each shopper slice. Enthusiast rows are the full enthusiast file, once."""
    pieces = []
    enth = frame[frame["audience"] == ENTHUSIAST].copy()
    enth_contrasts = event_contrasts(enth.assign(audience=ENTHUSIAST), barriers)
    enth_contrasts = enth_contrasts[enth_contrasts["audience"] == ENTHUSIAST]
    enth_contrasts["slice"] = "enthusiast"
    pieces.append(enth_contrasts)
    for name, block in shopper_slices(frame).items():
        tagged = block.copy()
        tagged["audience"] = SHOPPER
        contrasts = event_contrasts(tagged, barriers)
        contrasts = contrasts[contrasts["audience"] == SHOPPER]
        contrasts["slice"] = name
        pieces.append(contrasts)
    out = pd.concat(pieces, ignore_index=True)
    out["primary"] = out.apply(lambda row: PRIMARY_BARRIER.get(row["event_id"]) == row["barrier"], axis=1)
    return out
