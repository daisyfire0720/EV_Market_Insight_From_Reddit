#!/usr/bin/env python3
"""End-to-end pilot on the downloaded sample.

Reads the gitignored text sample and the committed monthly counts.
Writes tables and figures under ``v2/outputs/``. Descriptive only:
event windows are comparisons with the surrounding months, not effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotate import BARRIER_LABELS, annotate_frame  # noqa: E402
from ev_v2.cleaning import prepare_posts  # noqa: E402
from ev_v2.events import EVENTS, event_months  # noqa: E402
from ev_v2.funnel import STAGE_SPECS, assign_stage  # noqa: E402
from ev_v2.sentiment import score_texts  # noqa: E402
from ev_v2.topics import fit_lda  # noqa: E402

RAW_POSTS = ROOT / "data" / "raw" / "posts_downloaded.csv"
COUNTS = ROOT / "data_card" / "monthly_post_counts.csv"
OUT = ROOT / "outputs"
ANNOTATION = ROOT / "annotation"

BRAND_OWNER = {"BoltEV", "Rivian", "Ioniq5", "KiaEV6", "teslamotors"}
GENERAL_EV = {"electricvehicles", "evcharging"}


def community_proxy(sub: str, group: str) -> str:
    if sub in BRAND_OWNER:
        return "brand_community"
    if sub in GENERAL_EV:
        return "general_ev_community"
    if group == "mainstream_car":
        return "mainstream_car"
    return "other"


def _prior_months(month: str, k: int = 3) -> list[str]:
    y, m = (int(x) for x in month.split("-"))
    out = []
    for _ in range(k):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        out.append(f"{y:04d}-{m:02d}")
    return list(reversed(out))


def _year_ago(month: str) -> str:
    y, m = month.split("-")
    return f"{int(y) - 1:04d}-{m}"


def event_volume_table(counts: pd.DataFrame) -> pd.DataFrame:
    ok = counts[counts["count_status"].isin(["ok", "ok_empty"])].copy()
    ok["n_posts"] = pd.to_numeric(ok["n_posts"], errors="coerce").fillna(0)
    monthly = ok.groupby("month", as_index=False)["n_posts"].sum().set_index("month")["n_posts"]
    rows = []
    for event in EVENTS:
        months = event_months(event)
        pre = _prior_months(months[0], 3)
        ago = [_year_ago(m) for m in months]
        def _mean(keys: list[str]) -> tuple[float | None, int]:
            present = [k for k in keys if k in monthly.index]
            if not present:
                return None, 0
            return float(monthly.loc[present].mean()), len(present)

        event_mean, event_n = _mean(months)
        pre_mean, pre_n = _mean(pre)
        ago_mean, ago_n = _mean(ago)
        ratio_pre = None if not pre_mean else event_mean / pre_mean
        ratio_ago = None if not ago_mean else event_mean / ago_mean
        rows.append(
            {
                "event_id": event.event_id,
                "label": event.label,
                "event_date": event.event_date.isoformat(),
                "months": ",".join(months),
                "mean_posts_event_months": event_mean,
                "n_event_months_observed": event_n,
                "mean_posts_prior_3_months": pre_mean,
                "n_prior_months_observed": pre_n,
                "ratio_vs_prior_3": ratio_pre,
                "mean_posts_same_months_prior_year": ago_mean,
                "ratio_vs_year_ago": ratio_ago,
                "source": event.source,
            }
        )
    return pd.DataFrame(rows)


def stratified_ids(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = work["created_date"].dt.year.astype(int)
    groups = [g for _, g in work.groupby(["subreddit", "year"], sort=True) if len(g)]
    if not groups or n <= 0:
        return work.iloc[0:0]
    rng = np.random.default_rng(seed)
    # One row per cell first, then fill in proportion to cell size.
    picked = []
    remaining = []
    for g in groups:
        order = rng.permutation(len(g))
        picked.append(g.iloc[[order[0]]])
        if len(order) > 1:
            remaining.append(g.iloc[order[1:]])
    base = pd.concat(picked, ignore_index=True)
    if len(base) >= n:
        return base.sample(n, random_state=seed)
    if remaining:
        pool = pd.concat(remaining, ignore_index=True)
        extra_n = min(n - len(base), len(pool))
        extra = pool.sample(extra_n, random_state=seed)
        base = pd.concat([base, extra], ignore_index=True)
    return base


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def plot_counts(counts: pd.DataFrame, partial_month: str, path: Path) -> None:
    ok = counts[counts["count_status"].isin(["ok", "ok_empty"])].copy()
    ok["n_posts"] = pd.to_numeric(ok["n_posts"], errors="coerce").fillna(0)
    ok = ok[ok["month"] < partial_month]
    pivot = ok.pivot(index="month", columns="subreddit", values="n_posts").fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=f"r/{col}", linewidth=1.3)
    for event in EVENTS:
        ax.axvline(event.month_start, color="0.4", linewidth=0.6, linestyle="--")
    ax.set_title("Monthly submissions in core EV subreddits")
    ax.set_ylabel("Submissions")
    ax.set_xlabel("Month")
    ticks = list(pivot.index[::6])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_event_ratios(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.15, table["ratio_vs_prior_3"], width=0.3, label="Event months / prior 3 months")
    ax.bar(x + 0.15, table["ratio_vs_year_ago"], width=0.3, label="Event months / same months a year earlier")
    ax.axhline(1.0, color="0.3", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["event_id"], rotation=20, ha="right")
    ax.set_ylabel("Ratio of mean monthly submissions")
    ax.set_title("Core-subreddit volume around coded events")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sentiment(monthly: pd.DataFrame, path: Path) -> None:
    use = monthly[monthly["n"] >= 15].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not use.empty:
        ax.plot(use["month"], use["vader_compound_mean"], color="#1f4e79", label="Mean VADER compound")
        ax.fill_between(use["month"], use["vader_compound_mean"] - use["vader_compound_se"], use["vader_compound_mean"] + use["vader_compound_se"], alpha=0.2)
    ax.axhline(0, color="0.4", linewidth=0.7)
    ax.set_title("Sample sentiment by month (months with at least 15 posts)")
    ax.set_ylabel("VADER compound")
    ticks = list(use["month"][::3]) if len(use) else []
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_funnel(compare: pd.DataFrame, path: Path) -> None:
    stages = [s.name for s in STAGE_SPECS] + ["Unclear"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(stages))
    width = 0.35
    for i, mode in enumerate(["substring", "boundary"]):
        share = compare[compare["mode"] == mode].set_index("stage").reindex(stages)["share"].fillna(0)
        ax.bar(x + (i - 0.5) * width, share, width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("Share of sampled posts")
    ax.set_title("Journey-stage keywords: substring vs word boundary")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_topics(prevalence: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(prevalence["topic"].astype(str), prevalence["n_docs"])
    ax.set_xlabel("Documents (highest-probability topic)")
    ax.set_ylabel("Topic")
    ax.set_title("LDA topic sizes on the text sample")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=Path, default=RAW_POSTS)
    parser.add_argument("--counts", type=Path, default=COUNTS)
    parser.add_argument("--nli-max", type=int, default=400)
    parser.add_argument("--nli-model", default="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ANNOTATION.mkdir(parents=True, exist_ok=True)
    _style()

    if not args.posts.exists():
        raise SystemExit(f"missing posts file: {args.posts}")
    if not args.counts.exists():
        raise SystemExit(f"missing counts file: {args.counts}")

    raw = pd.read_csv(args.posts)
    counts = pd.read_csv(args.counts)
    partial_month = str(counts["month"].max())
    kept, clean_counts = prepare_posts(raw)
    kept["created_date"] = pd.to_datetime(kept["created_utc"], unit="s", utc=True)
    kept["month"] = kept["created_date"].dt.strftime("%Y-%m")
    kept["community_proxy"] = [
        community_proxy(s, g) for s, g in zip(kept["subreddit"], kept.get("community_group", pd.Series([""] * len(kept))))
    ]

    vader = score_texts(kept["text"].tolist())
    kept = pd.concat([kept.reset_index(drop=True), vader], axis=1)
    stages_b, hits_b, stages_s, hits_s = [], [], [], []
    for text in kept["text"]:
        sb, hb = assign_stage(text, "boundary")
        ss, hs = assign_stage(text, "substring")
        stages_b.append(sb)
        hits_b.append("|".join(hb))
        stages_s.append(ss)
        hits_s.append("|".join(hs))
    kept["stage_boundary"] = stages_b
    kept["stage_boundary_hits"] = hits_b
    kept["stage_substring"] = stages_s
    kept["stage_substring_hits"] = hits_s

    lda_note = "ok"
    try:
        lda = fit_lda(kept["text"].tolist(), n_topics=8, random_state=args.seed, min_df=4)
    except ValueError as exc:
        lda_note = f"refit with min_df=2 after: {exc}"
        lda = fit_lda(kept["text"].tolist(), n_topics=8, random_state=args.seed, min_df=2)
    kept["topic"] = lda.topic_ids
    kept["topic_probability_max"] = lda.topic_probability_max
    lda.topic_words.to_csv(OUT / "lda_topics.csv", index=False)
    prevalence = kept.groupby("topic").size().rename("n_docs").reset_index()
    prevalence = prevalence.merge(lda.topic_words[["topic", "top_words"]], on="topic")
    prevalence.to_csv(OUT / "lda_prevalence.csv", index=False)

    # Volume from the census, not from the text sample.
    plot_counts(counts, partial_month, OUT / "monthly_submissions.png")
    events = event_volume_table(counts)
    events.to_csv(OUT / "event_volume.csv", index=False)
    plot_event_ratios(events, OUT / "event_volume_ratios.png")

    sent = (
        kept.groupby("month")
        .agg(n=("vader_compound", "size"), vader_compound_mean=("vader_compound", "mean"), vader_compound_sd=("vader_compound", "std"))
        .reset_index()
        .sort_values("month")
    )
    sent["vader_compound_se"] = sent["vader_compound_sd"] / np.sqrt(sent["n"].clip(lower=1))
    sent.to_csv(OUT / "sample_monthly_sentiment.csv", index=False)
    plot_sentiment(sent, OUT / "sample_monthly_sentiment.png")

    funnel_rows = []
    for mode, col in [("boundary", "stage_boundary"), ("substring", "stage_substring")]:
        vc = kept[col].value_counts(dropna=False)
        for stage, n in vc.items():
            funnel_rows.append({"mode": mode, "stage": stage, "n": int(n), "share": float(n) / len(kept)})
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(OUT / "funnel_match_comparison.csv", index=False)
    plot_funnel(funnel, OUT / "funnel_match_comparison.png")

    by_proxy = (
        kept.groupby("community_proxy")
        .agg(
            n=("id", "size"),
            vader_compound_mean=("vader_compound", "mean"),
            share_stage_unclear_boundary=("stage_boundary", lambda s: float((s == "Unclear").mean())),
            share_stage_consideration_substring=("stage_substring", lambda s: float((s == "Consideration").mean())),
        )
        .reset_index()
    )
    by_proxy.to_csv(OUT / "community_proxy_summary.csv", index=False)

    # Event-window sentiment on the text sample. Small n is expected.
    event_sent_rows = []
    for event in EVENTS:
        months = set(event_months(event))
        in_window = kept[kept["month"].isin(months)]
        event_sent_rows.append(
            {
                "event_id": event.event_id,
                "n_posts_in_sample": int(len(in_window)),
                "vader_compound_mean": None if in_window.empty else float(in_window["vader_compound"].mean()),
            }
        )
    pd.DataFrame(event_sent_rows).to_csv(OUT / "event_sample_sentiment.csv", index=False)

    nli_meta = {"ran": False}
    nli_frame = stratified_ids(kept, args.nli_max, args.seed)
    if not args.skip_nli and len(nli_frame):
        print(f"NLI on {len(nli_frame)} posts with {args.nli_model}", flush=True)
        try:
            nli = annotate_frame(nli_frame, model_name=args.nli_model)
            nli.frame.to_csv(OUT / "nli_documents.csv", index=False)
            merged = nli_frame.merge(nli.frame, on="id", how="left")
            stance = merged["stance_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="n")
            stance["share"] = stance["n"] / stance["n"].sum()
            stance.to_csv(OUT / "nli_stance_shares.csv", index=False)
            stage = merged["stage_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="n")
            stage["share"] = stage["n"] / stage["n"].sum()
            stage.to_csv(OUT / "nli_stage_shares.csv", index=False)
            barrier_rows = []
            for label in BARRIER_LABELS:
                col = "barrier_" + label.replace(" ", "_") + "_on"
                barrier_rows.append({"barrier": label, "n_on": int(merged[col].sum()), "share": float(merged[col].mean()), "n": int(len(merged))})
            barriers = pd.DataFrame(barrier_rows)
            barriers.to_csv(OUT / "nli_barrier_shares.csv", index=False)
            by_group = (
                merged.groupby("community_proxy")["stance_label"].value_counts(normalize=True).rename("share").reset_index()
            )
            by_group.to_csv(OUT / "nli_stance_by_community.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(barriers["barrier"], barriers["share"])
            ax.set_xlabel(f"Share of NLI subset (n={len(merged)}) with score ≥ 0.5")
            ax.set_title("Zero-shot barrier labels")
            fig.tight_layout()
            fig.savefig(OUT / "nli_barrier_shares.png", dpi=140)
            plt.close(fig)
            nli_meta = {
                "ran": True,
                "model": nli.model_name,
                "revision": nli.model_revision,
                "n": int(len(merged)),
                "barrier_threshold": 0.5,
            }
        except Exception as exc:  # noqa: BLE001 - recorded, volume results still stand
            nli_meta = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
            print("NLI failed:", nli_meta["error"], flush=True)

    # Annotation draw. Target 800 when the sample is large enough; otherwise all kept rows.
    target = 800 if len(kept) >= 800 else len(kept)
    sheet = stratified_ids(kept, target, seed=args.seed + 1)
    preview_cols = ["id", "subreddit", "community_proxy", "month", "score", "title"]
    sheet_out = sheet[preview_cols].copy()
    sheet_out["text_excerpt"] = sheet["text"].str.replace("\n", " ", regex=False).str.slice(0, 500)
    for col in [
        "barrier_codes",
        "stage",
        "stance",
        "annotator",
        "notes",
    ]:
        sheet_out[col] = ""
    sheet_out.to_csv(ANNOTATION / "annotation_sample.csv", index=False)
    sheet_out.head(25).to_csv(ANNOTATION / "annotation_preview.csv", index=False)

    manifest = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "posts_file": str(args.posts),
        "input_rows": clean_counts["input_rows"],
        "removed_bot": clean_counts["removed_bot"],
        "removed_short_or_empty": clean_counts["removed_short_or_empty"],
        "kept_rows": clean_counts["kept_rows"],
        "partial_count_month_excluded_from_figure": partial_month,
        "lda": {
            "n_topics": 8,
            "random_state": args.seed,
            "note": lda_note,
            "mean_topic_probability_max": float(kept["topic_probability_max"].mean()),
            "min_topic_probability_max": float(kept["topic_probability_max"].min()),
        },
        "vader": "vaderSentiment.SentimentIntensityAnalyzer",
        "nli": nli_meta,
        "annotation_rows": int(len(sheet_out)),
        "funnel_consideration_share_substring": float((kept["stage_substring"] == "Consideration").mean()),
        "funnel_consideration_share_boundary": float((kept["stage_boundary"] == "Consideration").mean()),
        "caveat": "Text results describe the downloaded sample, not the census. Volume ratios describe archive counts and are not causal.",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
