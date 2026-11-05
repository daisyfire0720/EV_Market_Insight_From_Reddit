#!/usr/bin/env python3
"""Label the enthusiast sample and the shopper dump, then summarize Framing A.

The label step resumes from ``v2/data/raw/nli_scores.jsonl``. The summary
step does not call the model. Every table and figure is unvalidated model
output.

From the repository root::

    python v2/run_framing.py
    python v2/run_framing.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotate import BARRIER_LABELS, annotate_frame  # noqa: E402
from ev_v2.cleaning import prepare_posts  # noqa: E402
from ev_v2.events import EVENTS  # noqa: E402
from ev_v2.framing import (  # noqa: E402
    FRAMING_EVENT_IDS,
    PRIMARY_BARRIER,
    barrier_column,
    build_corpus,
    event_contrasts,
    event_study,
    interrupted_series_table,
    monthly_counts,
    overall_shares,
    upsert_framing_findings,
)

CORE_PATH = ROOT / "data" / "raw" / "posts_downloaded.csv"
DUMP_PATH = ROOT / "data" / "raw" / "mainstream_from_dumps.csv"
KEPT_PATH = ROOT / "data" / "raw" / "framing_kept.csv"
CHECKPOINT = ROOT / "data" / "raw" / "nli_scores.jsonl"
OUT = ROOT / "outputs"
FINDINGS = OUT / "FINDINGS.md"

PLOT_START = "2018-01"
PLOT_END = "2025-12"
PANEL_BARRIERS = (
    "purchase incentives or tax credits",
    "vehicle price or total cost",
    "charging access",
)
EVENT_MARKS = {
    "ira_2022": "2022-08",
    "tesla_price_cut_2023": "2023-01",
    "nacs_2023": "2023-05",
    "credit_deadline_2025": "2025-09",
}
COLORS = {"enthusiast": "#1f4e79", "shopper": "#b85c38"}


def _load_corpus() -> pd.DataFrame:
    core = pd.read_csv(CORE_PATH)
    dump = pd.read_csv(DUMP_PATH)
    return build_corpus(core, dump)


def label() -> None:
    import torch

    torch.set_num_threads(4)
    corpus = _load_corpus()
    kept, counts = prepare_posts(corpus)
    kept["id"] = kept["id"].astype(str)
    KEPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in ("id", "audience", "subreddit", "month", "title", "text", "author", "score") if col in kept.columns]
    kept[keep_cols].to_csv(KEPT_PATH, index=False)
    print(
        f"kept {counts['kept_rows']} of {counts['input_rows']} "
        f"(bots {counts['removed_bot']}, short {counts['removed_short_or_empty']})",
        flush=True,
    )
    print(kept["audience"].value_counts().to_dict(), flush=True)
    annotate_frame(kept, checkpoint_path=CHECKPOINT, progress_every=200, batch_size=8)
    print("labels checkpointed", flush=True)


def _scores() -> pd.DataFrame:
    rows = []
    with CHECKPOINT.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame["id"] = frame["id"].astype(str)
    return frame.drop_duplicates("id", keep="last")


def _labeled() -> pd.DataFrame:
    kept = pd.read_csv(KEPT_PATH)
    kept["id"] = kept["id"].astype(str)
    kept["month"] = kept["month"].astype(str).str.slice(0, 7)
    scores = _scores()
    merged = kept.merge(scores, on="id", how="inner")
    return merged


def _event_labels() -> dict[str, str]:
    return {event.event_id: event.label for event in EVENTS}


def _month_x(month: str) -> int:
    from ev_v2.framing import month_index

    return month_index(month)


def plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    window = monthly[(monthly["month"] >= PLOT_START) & (monthly["month"] <= PLOT_END)].copy()
    window["x"] = window["month"].map(_month_x)
    tick_months = [month for month in sorted(window["month"].unique()) if month.endswith("-01") or month.endswith("-07")]
    for ax, barrier in zip(axes, PANEL_BARRIERS):
        block = window[window["barrier"] == barrier]
        for audience, color in COLORS.items():
            group = block[block["audience"] == audience].sort_values("x")
            if group.empty:
                continue
            ax.plot(group["x"], group["share"], color=color, label=audience, linewidth=1.2)
            ax.fill_between(group["x"], group["lo"], group["hi"], color=color, alpha=0.15)
        for month in EVENT_MARKS.values():
            ax.axvline(_month_x(month), color="#888888", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Share")
        ax.set_title(barrier)
        top = float(block["hi"].max()) if len(block) else 0.45
        ax.set_ylim(0, min(1.0, max(0.45, top)))
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks([_month_x(month) for month in tick_months])
    axes[-1].set_xticklabels(tick_months, rotation=45, ha="right")
    fig.suptitle("Barrier shares, model labels (unvalidated)", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_event_study(study: pd.DataFrame, path: Path) -> None:
    labels = _event_labels()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, event_id in zip(axes.ravel(), FRAMING_EVENT_IDS):
        barrier = PRIMARY_BARRIER[event_id]
        block = study[(study["event_id"] == event_id) & (study["barrier"] == barrier)]
        for audience, color in COLORS.items():
            group = block[block["audience"] == audience].sort_values("rel_month")
            group = group[group["n"] > 0]
            if group.empty:
                continue
            ax.errorbar(
                group["rel_month"],
                group["diff"],
                yerr=1.96 * group["se"].fillna(0),
                fmt="o",
                color=color,
                label=audience,
                markersize=3.5,
                linewidth=1,
                capsize=2,
            )
        ax.axhline(0, color="#444444", linewidth=0.6)
        ax.axvline(-0.5, color="#888888", linewidth=0.6, linestyle="--")
        ax.set_title(f"{labels.get(event_id, event_id)}\n{barrier}")
        ax.set_xlabel("Months from the event (omitted: −1)")
    axes[0, 0].set_ylabel("Share minus the month before")
    axes[1, 0].set_ylabel("Share minus the month before")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Event-study contrasts, model labels (unvalidated)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_its(its: pd.DataFrame, path: Path) -> None:
    primary = its[its["primary"] == True].copy()  # noqa: E712
    primary = primary[primary["status"] == "ok"]
    labels = _event_labels()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if primary.empty:
        ax.set_title("No identified interrupted series")
    else:
        primary["row"] = primary["event_id"].map(lambda event_id: labels.get(event_id, event_id)) + " — " + primary["audience"]
        primary = primary.iloc[::-1]
        y = range(len(primary))
        ax.errorbar(
            primary["post"],
            list(y),
            xerr=1.96 * primary["post_se"].fillna(0),
            fmt="o",
            color="#1f4e79",
            capsize=3,
        )
        ax.set_yticks(list(y), primary["row"])
        ax.axvline(0, color="#444444", linewidth=0.6)
    ax.set_xlabel("Level shift in the monthly share")
    ax.set_title("Interrupted series, model labels (unvalidated)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summarize() -> None:
    labeled = _labeled()
    print(f"labeled rows {len(labeled)}", labeled["audience"].value_counts().to_dict(), flush=True)
    barriers = list(BARRIER_LABELS)
    monthly_parts = []
    for barrier in barriers:
        counts = monthly_counts(labeled, barrier_column(barrier))
        counts["barrier"] = barrier
        monthly_parts.append(counts)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    contrasts = event_contrasts(labeled, barriers)
    its = interrupted_series_table(labeled, barriers)
    study = event_study(labeled, barriers)
    overall = overall_shares(labeled, barriers)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "framing_monthly_shares.csv", index=False)
    contrasts.to_csv(OUT / "framing_event_contrasts.csv", index=False)
    its.to_csv(OUT / "framing_its.csv", index=False)
    study.to_csv(OUT / "framing_event_study.csv", index=False)
    overall.to_csv(OUT / "framing_overall_shares.csv", index=False)
    slim_cols = ["id", "audience", "subreddit", "month", "stage_label", "stance_label"]
    slim_cols += [barrier_column(label) for label in barriers]
    labeled[slim_cols].to_csv(OUT / "framing_labels.csv", index=False)
    plot_monthly(monthly, OUT / "framing_barrier_shares_monthly.png")
    plot_event_study(study, OUT / "framing_event_study.png")
    plot_its(its, OUT / "framing_its_level.png")
    upsert_framing_findings(FINDINGS, OUT)
    meta = {
        "validated": False,
        "n": int(len(labeled)),
        "by_audience": labeled["audience"].value_counts().astype(int).to_dict(),
        "model": str(labeled["nli_model"].iloc[0]) if "nli_model" in labeled.columns else "",
        "revision": str(labeled["nli_revision"].dropna().iloc[0]) if labeled["nli_revision"].notna().any() else "",
    }
    (OUT / "framing_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--label-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        label()
    if not args.label_only:
        summarize()


if __name__ == "__main__":
    main()
