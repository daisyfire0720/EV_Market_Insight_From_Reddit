"""FINDINGS section for the funnel, the discontinuity scaffold, and transitions.

Numbers are read from the tables round 7 writes. Labels are still unvalidated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ev_v2.round5_report import upsert_marked


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def render_round7_markdown(out_dir) -> str:
    out_dir = Path(out_dir)
    lines = [
        "## Round 7 funnel, discontinuity, and transitions (unvalidated)",
        "",
        "Daisy locked the 6-stage funnel, with onboarding kept separate from long-term ownership, and the 8 topic families. The note is `v2/docs/reference_taxonomy_6stage_8family.pdf`. The codebook for the labmate is `v2/annotation/CODEBOOK.md`. The blank sheet is `v2/annotation/annotation_sheet.csv`. The small NLI model is still the only label on the corpus. It has one ownership class, so it cannot yet measure onboarding against long-term ownership. Shares below are that model's labels, or a sampling lexicon used only to build the sheet. They are not human codes and they are not effects.",
        "",
    ]
    meta = out_dir / "annotation_funnel_meta.csv"
    if meta.exists():
        row = pd.read_csv(meta).iloc[0]
        lines.append(
            f"The sheet has {int(row['n'])} posts ({int(row['n_enthusiast'])} enthusiast, {int(row['n_shopper'])} shopper). "
            f"{int(row['n_with_text'])} have an excerpt. {int(row['n_calibration'])} are flagged `calibration=yes` and are excluded from kappa. "
            f"Code cells are empty."
        )
        lines.append("")
    coverage = out_dir / "annotation_funnel_coverage.csv"
    if coverage.exists():
        lines.append("Sampling coverage inside the sheet (not codes). A stage or family the small model cannot see is absent here even if the coder will still use it:")
        lines.append("")
        lines.append("| Audience | Stratum | Kind | Posts |")
        lines.append("|---|---|---|---:|")
        table = pd.read_csv(coverage)
        for _, row in table.iterrows():
            lines.append(f"| {row['audience']} | {row['stratum']} | {row['kind']} | {int(row['n'])} |")
        lines.append("")
    shares = out_dir / "funnel_stage_shares.csv"
    if shares.exists():
        lines.append("### Small-model stage shares")
        lines.append("")
        lines.append("Ownership is one class. Onboarding is not identified.")
        lines.append("")
        lines.append("| Audience | Stage | Share | n |")
        lines.append("|---|---|---:|---:|")
        for _, row in pd.read_csv(shares).iterrows():
            lines.append(f"| {row['audience']} | {row['stage']} | {_fmt(row['share'])} | {int(row['n'])} |")
        lines.append("")
    authors = out_dir / "author_coverage.csv"
    if authors.exists():
        lines.append("### Repeat-author coverage")
        lines.append("")
        lines.append("Deleted authors are dropped. The hash is not a name.")
        lines.append("")
        lines.append("| Audience | Posts with a hash | Authors | Repeat authors | Share of posts from repeat authors | Max posts |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in pd.read_csv(authors).iterrows():
            lines.append(
                f"| {row['audience']} | {int(row['n_posts'])} | {int(row['n_authors'])} | {int(row['n_repeat_authors'])} | "
                f"{_fmt(row['share_posts_from_repeat_authors'])} | {int(row['max_posts_per_author'])} |"
            )
        lines.append("")
        if "median_posts_per_author" in pd.read_csv(authors).columns:
            med = pd.read_csv(authors)
            all_rows = med[med["audience"] == "all"]
            if len(all_rows):
                lines.append(
                    f"The median author has { _fmt(all_rows.iloc[0]['median_posts_per_author'], 1)} post in this file, "
                    f"and the busiest hash has {int(all_rows.iloc[0]['max_posts_per_author'])} posts. "
                    "The repeat-post share is a minority of accounts, not a panel of buyers."
                )
                lines.append("")
    shocks = out_dir / "stage_transition_shocks.csv"
    if shocks.exists():
        lines.append("### Stage transitions across the cutoffs")
        lines.append("")
        lines.append("A pair is two successive posts by the same author, the first in the 180 days before the cutoff and the second in the 180 days after. Forward means the later post is further down the small-model ladder. Ownership is one rung.")
        lines.append("")
        lines.append("| Cutoff | Fit | Pairs | Forward | Rate |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in pd.read_csv(shocks).iterrows():
            lines.append(
                f"| {row['cutoff_id']} | {row['fit']} | {int(row['n_pairs'])} | {int(row['n_forward'])} | {_fmt(row['forward_rate'])} |"
            )
        lines.append("")
    rdit = out_dir / "rdit.csv"
    if rdit.exists():
        lines.append("### Regression discontinuity in time")
        lines.append("")
        lines.append("Local linear jump in the daily share, triangular kernel, bandwidth chosen by even/odd day cross-validation on 30, 60, 90, and 180 days. The donut drops 7 days on each side. Placebos move the cutoff by 180 days. A thin fit has fewer than 30 posts or 5 days on a side.")
        lines.append("")
        lines.append("| Cutoff | Audience | Outcome | Fit | Bandwidth | Jump | Posts left | Posts right | Thin |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
        table = pd.read_csv(rdit)
        main = table[table["fit"].isin(["main", "donut"]) | table["fit"].astype(str).str.startswith("placebo")]
        for _, row in main.iterrows():
            lines.append(
                f"| {row['cutoff_id']} | {row['audience']} | {row['outcome']} | {row['fit']} | {int(row['bandwidth_days'])} | "
                f"{_fmt(row['tau'])} | {int(row['n_left'])} | {int(row['n_right'])} | {row['thin']} |"
            )
        lines.append("")
    lines.append(
        "GPU instructions for the instruction model are `v2/RUN_ON_MY_GPU.md`. "
        "That run has not been done on this machine. The 800-post sheet is what the labmate codes."
    )
    lines.append("")
    return "\n".join(lines)


def upsert_round7_findings(findings_path, out_dir) -> None:
    upsert_marked(
        Path(findings_path),
        "<!-- round7-start -->",
        "<!-- round7-end -->",
        render_round7_markdown(out_dir),
    )
