#!/usr/bin/env python3
"""Pull and label the four thin enthusiast baseline months.

Budget, stated before any request: at most 3,360 posts (40 per
subreddit-window, three windows, seven subreddits, four months) and 90
minutes of CPU labeling. Raw text stays under ``v2/data/raw/``. The script
appends slim labels only. It does not redraw the annotation sheet and does
not start the 282k pull.

From ``v2/``::

    python fill_thin_baselines.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from download_reddit import ArcticClient, fetch_posts  # noqa: E402
from ev_v2.annotate import annotate_frame  # noqa: E402
from ev_v2.arctic import CORE_SUBREDDITS  # noqa: E402
from ev_v2.blocks import assign_week_block  # noqa: E402
from ev_v2.bootstrap import block_bootstrap_contrasts, bootstrap_interrupted_series, window_flags  # noqa: E402
from ev_v2.cleaning import prepare_posts  # noqa: E402
from ev_v2.framing import (  # noqa: E402
    ENTHUSIAST,
    barrier_column,
    event_contrasts,
    event_study,
    interrupted_series_table,
    monthly_counts,
    overall_shares,
    upsert_framing_findings,
)
from ev_v2.round5_report import upsert_round5_findings  # noqa: E402
from ev_v2.round6_report import upsert_round6_findings  # noqa: E402
from ev_v2.sensitivity import slice_contrasts, slice_shares  # noqa: E402
from ev_v2.thin_baselines import FILL_LIMIT, LABEL_BUDGET_MINUTES, max_posts, planned_windows  # noqa: E402
from run_round5 import ALL_BARRIERS  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
RAW_POSTS = RAW_DIR / "thin_baseline_posts.csv"
CHECKPOINT = RAW_DIR / "nli_thin.jsonl"
TIMES = RAW_DIR / "enthusiast_times.csv"
LABELS = ROOT / "outputs" / "framing_labels.csv"
OUT = ROOT / "outputs"
FINDINGS = OUT / "FINDINGS.md"
CACHE = RAW_DIR / "thin_baseline_cache.json"

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


def _snapshot() -> None:
    for name in (
        "framing_block_bootstrap.csv",
        "framing_its_bootstrap.csv",
        "framing_event_contrasts.csv",
        "framing_its.csv",
        "framing_window_flags.csv",
        "framing_overall_shares.csv",
    ):
        src = OUT / name
        dest = OUT / name.replace(".csv", "_before_fill.csv")
        if src.exists() and not dest.exists():
            shutil.copyfile(src, dest)


def _with_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    times = pd.read_csv(TIMES) if TIMES.exists() else pd.DataFrame(columns=["id", "created_utc"])
    times["id"] = times["id"].astype(str)
    times = times.drop_duplicates("id")
    work = work.merge(times[["id", "created_utc"]], on="id", how="left")
    blocks = []
    for audience, subreddit, created in zip(work["audience"], work["subreddit"], work["created_utc"]):
        if audience != ENTHUSIAST:
            blocks.append("")
            continue
        blocks.append(assign_week_block(subreddit, created))
    work["week_block"] = blocks
    work["subreddit_month"] = work["subreddit"].astype(str) + "|" + work["month"]
    return work


def _pull(existing_ids: set[str]) -> pd.DataFrame:
    windows = planned_windows()
    print(
        f"budget: at most {max_posts()} posts across {len(windows)} windows, "
        f"labeling stops at {LABEL_BUDGET_MINUTES} minutes",
        flush=True,
    )
    client = ArcticClient(CACHE, min_interval=0.8)
    rows: list[dict] = []
    seen = set(existing_ids)
    for i, (subreddit, month, start, end) in enumerate(windows, start=1):
        got = fetch_posts(client, subreddit, start, end, FILL_LIMIT)
        kept_here = 0
        for row in got:
            doc_id = str(row.get("id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            row["subreddit"] = row.get("subreddit") or subreddit
            row["fill_month"] = month
            rows.append(row)
            kept_here += 1
        print(f"  {i}/{len(windows)} r/{subreddit} {start}..{end} returned {len(got)} new {kept_here}", flush=True)
        if i % 10 == 0:
            client.save()
    client.save()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["id"] = frame["id"].astype(str)
    frame["created_utc"] = pd.to_numeric(frame["created_utc"], errors="coerce")
    frame["created_date"] = pd.to_datetime(frame["created_utc"], unit="s", utc=True)
    frame["month"] = frame["created_date"].dt.strftime("%Y-%m")
    # Keep a post only when its UTC month is the month that window was meant to fill.
    frame = frame[frame["month"] == frame["fill_month"]].copy()
    frame = frame.drop_duplicates("id")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_POSTS, index=False)
    return frame


def _append_times(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    fresh = frame[["id", "created_utc"]].copy()
    fresh["id"] = fresh["id"].astype(str)
    if TIMES.exists():
        old = pd.read_csv(TIMES)
        old["id"] = old["id"].astype(str)
        fresh = fresh[~fresh["id"].isin(set(old["id"]))]
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh
    TIMES.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TIMES, index=False)


def _label(posts: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    import torch

    torch.set_num_threads(4)
    started = time.perf_counter()
    # Label in chunks so a slow machine stops at the stated budget.
    pieces = []
    remaining = posts
    while len(remaining):
        elapsed = (time.perf_counter() - started) / 60.0
        if pieces and elapsed >= LABEL_BUDGET_MINUTES:
            print(f"labeling budget reached at {elapsed:.1f} minutes; stopping", flush=True)
            break
        chunk = remaining.iloc[:200]
        remaining = remaining.iloc[200:]
        part = annotate_frame(chunk, checkpoint_path=CHECKPOINT, progress_every=50, batch_size=8).frame
        pieces.append(part)
    minutes = (time.perf_counter() - started) / 60.0
    if not pieces:
        return pd.DataFrame(), minutes
    return pd.concat(pieces, ignore_index=True), minutes


def _rebuild(combined: pd.DataFrame) -> None:
    blocked = _with_blocks(combined)
    print("event contrasts and ITS", flush=True)
    contrasts = event_contrasts(blocked, ALL_BARRIERS)
    contrasts.to_csv(OUT / "framing_event_contrasts.csv", index=False)
    its = interrupted_series_table(blocked, ALL_BARRIERS)
    its.to_csv(OUT / "framing_its.csv", index=False)
    study = event_study(blocked, ALL_BARRIERS)
    study.to_csv(OUT / "framing_event_study.csv", index=False)
    shares = overall_shares(blocked, ALL_BARRIERS)
    shares.to_csv(OUT / "framing_overall_shares.csv", index=False)
    pieces = []
    for barrier in ALL_BARRIERS:
        counts = monthly_counts(blocked, barrier_column(barrier))
        counts["barrier"] = barrier
        pieces.append(counts)
    monthly = pd.concat(pieces, ignore_index=True)
    monthly.to_csv(OUT / "framing_monthly_shares.csv", index=False)
    print("block bootstrap", flush=True)
    block_bootstrap_contrasts(blocked, ALL_BARRIERS).to_csv(OUT / "framing_block_bootstrap.csv", index=False)
    window_flags(blocked).to_csv(OUT / "framing_window_flags.csv", index=False)
    print("ITS bootstrap", flush=True)
    bootstrap_interrupted_series(blocked, ALL_BARRIERS).to_csv(OUT / "framing_its_bootstrap.csv", index=False)
    slice_shares(blocked, ALL_BARRIERS).to_csv(OUT / "framing_sensitivity_shares.csv", index=False)
    slice_contrasts(blocked, ALL_BARRIERS).to_csv(OUT / "framing_sensitivity_contrasts.csv", index=False)
    blocks = blocked.loc[blocked["audience"] == ENTHUSIAST, ["id", "week_block"]].copy()
    blocks.to_csv(OUT / "enthusiast_week_blocks.csv", index=False)
    from run_framing import plot_event_study, plot_its, plot_monthly

    plot_monthly(monthly, OUT / "framing_barrier_shares_monthly.png")
    plot_event_study(study, OUT / "framing_event_study.png")
    plot_its(its, OUT / "framing_its_level.png")
    upsert_framing_findings(FINDINGS, OUT)
    upsert_round5_findings(FINDINGS, OUT)
    upsert_round6_findings(FINDINGS, OUT)
    from ev_v2.manuscript import write_manuscript

    write_manuscript(OUT, ROOT / "paper")


def main() -> int:
    print(
        f"thin baseline fill: months from the plan, cap {max_posts()} posts, "
        f"{LABEL_BUDGET_MINUTES} minutes. Subreddits: {', '.join(CORE_SUBREDDITS)}.",
        flush=True,
    )
    _snapshot()
    existing = pd.read_csv(LABELS)
    existing["id"] = existing["id"].astype(str)
    before_n = len(existing)
    raw = _pull(set(existing["id"]))
    if raw.empty:
        print("no new posts")
        return 0
    print(f"downloaded {len(raw)} new posts", flush=True)
    kept, counts = prepare_posts(raw)
    print(
        f"kept {counts['kept_rows']} of {counts['input_rows']} "
        f"(bots {counts['removed_bot']}, short {counts['removed_short_or_empty']})",
        flush=True,
    )
    if len(kept) > max_posts():
        kept = kept.iloc[: max_posts()].copy()
    labeled, minutes = _label(kept)
    if labeled.empty:
        print("nothing labeled")
        return 0
    labeled["id"] = labeled["id"].astype(str)
    meta = kept.copy()
    meta["id"] = meta["id"].astype(str)
    meta["month"] = meta["month"].astype(str).str.slice(0, 7)
    meta["audience"] = ENTHUSIAST
    merged = meta.merge(labeled, on="id", how="inner")
    slim = merged[[col for col in SLIM if col in merged.columns]].copy()
    slim = slim[~slim["id"].isin(set(existing["id"]))]
    combined = pd.concat([existing, slim], ignore_index=True)
    combined.to_csv(LABELS, index=False)
    _append_times(merged)
    by_month = slim.groupby("month").size().astype(int).to_dict()
    manifest = {
        "validated": False,
        "n_windows": len(planned_windows()),
        "max_posts": max_posts(),
        "label_budget_minutes": LABEL_BUDGET_MINUTES,
        "n_downloaded": int(len(raw)),
        "n_kept": int(counts["kept_rows"]),
        "n_appended": int(len(slim)),
        "n_labels_before": int(before_n),
        "n_labels_after": int(len(combined)),
        "label_minutes": round(minutes, 2),
        "kept_by_month": {str(k): int(v) for k, v in by_month.items()},
        "removed_bot": int(counts["removed_bot"]),
        "removed_short": int(counts["removed_short_or_empty"]),
    }
    (OUT / "thin_baseline_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    _rebuild(combined)
    print("labels unvalidated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
