#!/usr/bin/env python3
"""Label the 2026 shopper extract and fold it into the Framing A tables.

The raw CSV is gitignored. This script appends model labels to
``v2/outputs/framing_labels.csv`` and rebuilds the contrast tables.
It does not redraw the annotation sheet. Labels stay unvalidated.

From the repository root::

    python v2/label_2026_shoppers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotate import BARRIER_LABELS, annotate_frame  # noqa: E402
from ev_v2.blocks import assign_week_block  # noqa: E402
from ev_v2.bootstrap import block_bootstrap_contrasts, bootstrap_interrupted_series, window_flags  # noqa: E402
from ev_v2.cleaning import prepare_posts  # noqa: E402
from ev_v2.framing import (  # noqa: E402
    SHOPPER,
    event_contrasts,
    interrupted_series_table,
    monthly_counts,
    overall_shares,
    upsert_framing_findings,
)
from ev_v2.round5_report import upsert_round5_findings  # noqa: E402
from ev_v2.sensitivity import slice_contrasts, slice_shares  # noqa: E402
from run_round5 import ALL_BARRIERS, _shopper_2026_note  # noqa: E402

RAW = ROOT / "data" / "raw" / "mainstream_2026.csv"
LABELS = ROOT / "outputs" / "framing_labels.csv"
TIMES = ROOT / "data" / "raw" / "enthusiast_times.csv"
OUT = ROOT / "outputs"
FINDINGS = OUT / "FINDINGS.md"
CHECKPOINT = ROOT / "data" / "raw" / "nli_2026.jsonl"

SLIM = [
    "id",
    "audience",
    "subreddit",
    "month",
    "stage_label",
    "stance_label",
    "barrier_charging_access_on",
    "barrier_driving_range_on",
    "barrier_vehicle_price_or_total_cost_on",
    "barrier_purchase_incentives_or_tax_credits_on",
    "barrier_battery_reliability_on",
    "barrier_dealer_or_buying_friction_on",
    "barrier_environmental_impact_on",
]


def _with_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    times = pd.read_csv(TIMES) if TIMES.exists() else pd.DataFrame(columns=["id", "created_utc"])
    times["id"] = times["id"].astype(str)
    work = work.merge(times[["id", "created_utc"]], on="id", how="left")
    blocks = []
    for audience, subreddit, created in zip(work["audience"], work["subreddit"], work["created_utc"]):
        if audience != "enthusiast":
            blocks.append("")
            continue
        blocks.append(assign_week_block(subreddit, created))
    work["week_block"] = blocks
    work["subreddit_month"] = work["subreddit"].astype(str) + "|" + work["month"]
    return work


def main() -> int:
    if not RAW.is_file():
        print(f"missing {RAW}", file=sys.stderr)
        return 2
    import torch

    torch.set_num_threads(4)
    posts = pd.read_csv(RAW)
    posts["audience"] = SHOPPER
    kept, counts = prepare_posts(posts)
    print(
        f"kept {counts['kept_rows']} of {counts['input_rows']} "
        f"(bots {counts['removed_bot']}, short {counts['removed_short_or_empty']})",
        flush=True,
    )
    if kept.empty:
        print("nothing to label")
        return 0
    labeled = annotate_frame(kept, checkpoint_path=CHECKPOINT, progress_every=50, batch_size=8).frame
    labeled["id"] = labeled["id"].astype(str)
    meta = kept.copy()
    meta["id"] = meta["id"].astype(str)
    meta["month"] = meta["month"].astype(str).str.slice(0, 7)
    meta["audience"] = SHOPPER
    merged = meta.merge(labeled, on="id", how="inner")
    slim = merged[[col for col in SLIM if col in merged.columns]].copy()
    existing = pd.read_csv(LABELS)
    existing["id"] = existing["id"].astype(str)
    before = len(existing)
    slim = slim[~slim["id"].isin(set(existing["id"]))]
    combined = pd.concat([existing, slim], ignore_index=True)
    combined.to_csv(LABELS, index=False)
    print(f"framing_labels {before} -> {len(combined)} (+{len(slim)})", flush=True)
    blocked = _with_blocks(combined)
    contrasts = event_contrasts(blocked, ALL_BARRIERS)
    contrasts.to_csv(OUT / "framing_event_contrasts.csv", index=False)
    its = interrupted_series_table(blocked, ALL_BARRIERS)
    its.to_csv(OUT / "framing_its.csv", index=False)
    shares = overall_shares(blocked, ALL_BARRIERS)
    shares.to_csv(OUT / "framing_overall_shares.csv", index=False)
    # Monthly shares for the three plotted barriers, same shape run_framing writes.
    pieces = []
    for barrier in ALL_BARRIERS:
        from ev_v2.framing import barrier_column

        counts_m = monthly_counts(blocked, barrier_column(barrier))
        counts_m["barrier"] = barrier
        pieces.append(counts_m)
    pd.concat(pieces, ignore_index=True).to_csv(OUT / "framing_monthly_shares.csv", index=False)
    print("block bootstrap", flush=True)
    block_bootstrap_contrasts(blocked, ALL_BARRIERS).to_csv(OUT / "framing_block_bootstrap.csv", index=False)
    window_flags(blocked).to_csv(OUT / "framing_window_flags.csv", index=False)
    print("ITS bootstrap", flush=True)
    bootstrap_interrupted_series(blocked, ALL_BARRIERS).to_csv(OUT / "framing_its_bootstrap.csv", index=False)
    slice_shares(blocked, ALL_BARRIERS).to_csv(OUT / "framing_sensitivity_shares.csv", index=False)
    slice_contrasts(blocked, ALL_BARRIERS).to_csv(OUT / "framing_sensitivity_contrasts.csv", index=False)
    _shopper_2026_note()
    from ev_v2.round5_report import upsert_round5_card

    upsert_framing_findings(FINDINGS, OUT)
    upsert_round5_findings(FINDINGS, OUT)
    upsert_round5_card(ROOT / "data_card" / "DATA_CARD.md")
    # Credit-deadline rows, so the log shows whether the window moved.
    credit = contrasts[(contrasts["event_id"] == "credit_deadline_2025") & (contrasts["primary"] == True)]  # noqa: E712
    print(credit[["audience", "baseline", "diff", "lo", "hi", "n_event", "n_baseline"]].to_string(index=False))
    credit_its = its[(its["event_id"] == "credit_deadline_2025") & (its["primary"] == True)]  # noqa: E712
    print(credit_its[["audience", "post", "post_lo", "post_hi", "n_months", "n_months_post", "status"]].to_string(index=False))
    print("model", BARRIER_LABELS[0], "labels unvalidated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
