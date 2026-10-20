"""FINDINGS and data-card sections for the round-5 checks.

Numbers are read from the tables the runner writes. The section says, in
the first line, that the labels are still unvalidated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _yn(value) -> str:
    if isinstance(value, str):
        return value
    return "yes" if bool(value) else "no"


def render_round5_markdown(out_dir) -> str:
    out_dir = Path(out_dir)
    lines = [
        "## Round 5 checks (unvalidated)",
        "",
        "The labels are still the small open model. No person has coded the annotation sheet. Intervals in this section resample the design: sampled weeks for the enthusiast file, and calendar months for shoppers. A window built from fewer months than the design asks for is flagged. None of these differences is an effect of the statute or the price cut. If the round 6 baseline fill is already in the label file, these tables are the filled file. The before-and-after comparison is in the round 6 section.",
        "",
    ]
    summary = out_dir / "annotation_sample_summary.csv"
    if summary.exists():
        comp = pd.read_csv(summary)
        lines.append("### Annotation sample")
        lines.append("")
        lines.extend(_annotation_prose(comp, out_dir))
        lines.append("")
    boot = out_dir / "framing_block_bootstrap.csv"
    if boot.exists():
        lines.append("### Block-bootstrap intervals")
        lines.append("")
        lines.extend(_bootstrap_prose(pd.read_csv(boot)))
        lines.append("")
    its = out_dir / "framing_its_bootstrap.csv"
    if its.exists():
        lines.append("### Interrupted series, month bootstrap")
        lines.append("")
        lines.extend(_its_prose(pd.read_csv(its)))
        lines.append("")
    shares = out_dir / "framing_sensitivity_shares.csv"
    contrasts = out_dir / "framing_sensitivity_contrasts.csv"
    if shares.exists():
        lines.append("### Shopper composition")
        lines.append("")
        lines.extend(_sensitivity_prose(pd.read_csv(shares), pd.read_csv(contrasts) if contrasts.exists() else None))
        lines.append("")
    lines.append(
        "The codebook, the blank sheet, and the model key are in `v2/annotation/`. "
        "`v2/score_agreement.py` scores two coder files when they exist. It was checked on synthetic labels only."
    )
    lines.append("")
    return "\n".join(lines)


def _annotation_prose(comp: pd.DataFrame, out_dir: Path) -> list[str]:
    lines = []
    meta_path = out_dir / "annotation_sample_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path).iloc[0]
        lines.append(
            f"The sheet has {int(meta['n'])} submissions ({int(meta['n_enthusiast'])} enthusiast, {int(meta['n_shopper'])} shopper). "
            f"Text excerpts were filled for {int(meta['n_with_text'])} of them. "
            "The size is the 800 already set in the research design, so two annotators can finish it. "
            "Inside that 800, each audience is asked for 40 posts on each barrier before the rest of the sample is drawn, "
            "then 25 posts in each of the four event windows, then a year-band remainder. "
            "Posts added only to fill the year bands are dropped first if the layers run past 800. "
            "The practice file of 25 is excluded."
        )
        lines.append("")
    lines.append("| Audience | Code | Posts in the audience draw | Positives in the draw |")
    lines.append("|---|---|---:|---:|")
    show = comp[~comp["code"].astype(str).str.startswith("event:")]
    for _, row in show.iterrows():
        lines.append(f"| {row['audience']} | {row['code']} | {int(row['n_posts'])} | {int(row['n_positive'])} |")
    return lines


def _bootstrap_prose(frame: pd.DataFrame) -> list[str]:
    primary = frame[(frame["primary"] == True) & (frame["reported"] == True)].copy()  # noqa: E712
    lines = [
        "The difference is the event-window share minus the baseline share, same point estimate as the independent-post interval. "
        "The new interval is the 2.5 and 97.5 percentiles of 2,000 cluster resamples. "
        "Enthusiast blocks are sampled weeks. Shopper blocks are calendar months when both sides have at least two months; otherwise the reported interval resamples subreddit-months and the one-month window is flagged.",
        "",
        "| Event | Audience | Baseline | Difference | Independent interval | Block interval | Block | Thin window | Few blocks | Event n | Baseline n |",
        "|---|---|---|---:|---|---|---|---|---|---:|---:|",
    ]
    order = ["ira_2022", "tesla_price_cut_2023", "nacs_2023", "credit_deadline_2025"]
    primary["event_id"] = pd.Categorical(primary["event_id"], order, ordered=True)
    for _, row in primary.sort_values(["event_id", "audience", "baseline"]).iterrows():
        lines.append(
            f"| {row['event_id']} | {row['audience']} | {row['baseline']} | {_fmt(row['diff'])} | "
            f"{_fmt(row['indep_lo'])} to {_fmt(row['indep_hi'])} | {_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])} | "
            f"{row['block']} | {_yn(row['thin_window'])} | {_yn(row['few_blocks'])} | {int(row['n_event'])} | {int(row['n_baseline'])} |"
        )
    lines.append("")
    changed = []
    for _, row in primary.iterrows():
        if bool(row["indep_excludes_zero"]) != bool(row["boot_excludes_zero"]):
            side = "the block interval includes 0" if not bool(row["boot_excludes_zero"]) else "the block interval excludes 0"
            changed.append(f"{row['event_id']} / {row['audience']} / {row['baseline']} ({side})")
    if changed:
        lines.append("Where the two intervals disagree about 0: " + "; ".join(changed) + ".")
    else:
        lines.append("On these primary contrasts, the block interval and the independent-post interval agree about whether 0 is inside.")
    thin = primary[primary["thin_window"] == True]  # noqa: E712
    if len(thin):
        bits = [f"{row['event_id']} / {row['audience']} / {row['baseline']}" for _, row in thin.iterrows()]
        lines.append("Thin windows (fewer sampled months than the design): " + "; ".join(bits) + ".")
    wide = primary[pd.to_numeric(primary["boot_width"], errors="coerce") > 0.4]
    if len(wide):
        bits = [
            f"{row['event_id']} / {row['audience']} / {row['baseline']} ({_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])})"
            for _, row in wide.iterrows()
        ]
        lines.append(
            "Some block intervals exclude 0 and are still too wide to read as a precise shift, because a side has only a few clusters: "
            + "; ".join(bits)
            + "."
        )
    return lines


def _its_prose(frame: pd.DataFrame) -> list[str]:
    primary = frame[frame["primary"] == True].copy()  # noqa: E712
    lines = [
        "Each draw resamples the months in the 12-before / 6-after window and refits the level shift. "
        "The HC1 interval is the one from the original weighted regression.",
        "",
        "| Event | Audience | Level shift | HC1 interval | Month-bootstrap interval | HC1 excludes 0 | Bootstrap excludes 0 | Months |",
        "|---|---|---:|---|---|---|---|---:|",
    ]
    for _, row in primary.sort_values(["event_id", "audience"]).iterrows():
        lines.append(
            f"| {row['event_id']} | {row['audience']} | {_fmt(row['post'])} | {_fmt(row['hc1_lo'])} to {_fmt(row['hc1_hi'])} | "
            f"{_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])} | {_yn(row['hc1_excludes_zero'])} | {_yn(row['boot_excludes_zero'])} | {int(row['n_months'])} |"
        )
    return lines


def _sensitivity_prose(shares: pd.DataFrame, contrasts: pd.DataFrame | None) -> list[str]:
    lines = [
        "Pre-purchase means the stage label is learning, comparing, evaluating, or buying. "
        "Owning and “no clear purchase stage” are left out of that slice. r/carbuying is the third shopper subreddit on its own.",
        "",
        "| Slice | Measure | Share | n | Shopper minus enthusiast | 95% interval |",
        "|---|---|---:|---:|---:|---|",
    ]
    keep_measures = {
        "vehicle price or total cost",
        "dealer or buying friction",
        "charging access",
        "purchase incentives or tax credits",
        "owning an electric vehicle",
        "support for electric vehicles",
    }
    show = shares[shares["measure"].isin(keep_measures) & shares["slice"].isin(["all_shopper", "carbuying", "pre_purchase", "enthusiast"])]
    for _, row in show.sort_values(["measure", "slice"]).iterrows():
        gap = row["gap_shopper_minus_enthusiast"] if "gap_shopper_minus_enthusiast" in row.index else float("nan")
        lines.append(
            f"| {row['slice']} | {row['measure']} | {_fmt(row['share'])} | {int(row['n'])} | {_fmt(gap)} | {_fmt(row.get('gap_lo'))} to {_fmt(row.get('gap_hi'))} |"
        )
    if contrasts is not None and len(contrasts):
        lines.append("")
        lines.append("Primary event contrasts inside the slices (independent-post intervals, because several slices are a few dozen posts):")
        lines.append("")
        lines.append("| Slice | Event | Baseline | Difference | 95% interval | Event n | Baseline n |")
        lines.append("|---|---|---|---:|---|---:|---:|")
        primary = contrasts[(contrasts["primary"] == True) & (contrasts["slice"].isin(["carbuying", "pre_purchase", "all_shopper"]))]  # noqa: E712
        for _, row in primary.sort_values(["event_id", "slice", "baseline"]).iterrows():
            lines.append(
                f"| {row['slice']} | {row['event_id']} | {row['baseline']} | {_fmt(row['diff'])} | {_fmt(row['lo'])} to {_fmt(row['hi'])} | {int(row['n_event'])} | {int(row['n_baseline'])} |"
            )
    lines.append("")
    lines.extend(_sensitivity_reading(shares, contrasts))
    return lines


def _cell(shares: pd.DataFrame, slice_name: str, measure: str) -> pd.Series | None:
    hit = shares[(shares["slice"] == slice_name) & (shares["measure"] == measure)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _sensitivity_reading(shares: pd.DataFrame, contrasts: pd.DataFrame | None) -> list[str]:
    """A short reading of the two slices. Numbers come from the table just printed."""
    lines = []
    price = _cell(shares, "carbuying", "vehicle price or total cost")
    dealer = _cell(shares, "carbuying", "dealer or buying friction")
    own = _cell(shares, "carbuying", "owning an electric vehicle")
    support = _cell(shares, "carbuying", "support for electric vehicles")
    if price is not None and dealer is not None and own is not None:
        lines.append(
            f"On r/carbuying alone (n={int(price['n'])}), price is {_fmt(price['share'])} and dealer friction is {_fmt(dealer['share'])}. "
            f"Both sit above the enthusiast file (gaps {_fmt(price['gap_shopper_minus_enthusiast'])} and {_fmt(dealer['gap_shopper_minus_enthusiast'])}, and both intervals exclude 0). "
            f"Ownership is still {_fmt(own['share'])}"
            + (f" and support is {_fmt(support['share'])}." if support is not None else ".")
            + " That ownership share is still above the enthusiast file. The event-window cells in r/carbuying are mostly single-digit counts, so those contrasts are not a check of the shocks."
        )
        lines.append("")
    pre_price = _cell(shares, "pre_purchase", "vehicle price or total cost")
    pre_dealer = _cell(shares, "pre_purchase", "dealer or buying friction")
    pre_support = _cell(shares, "pre_purchase", "support for electric vehicles")
    if pre_price is not None and pre_dealer is not None:
        lines.append(
            f"The pre-purchase slice has n={int(pre_price['n'])} and, by construction, no ownership posts. "
            f"Price is {_fmt(pre_price['share'])} (gap {_fmt(pre_price['gap_shopper_minus_enthusiast'])}, interval {_fmt(pre_price['gap_lo'])} to {_fmt(pre_price['gap_hi'])}). "
            f"Dealer friction is {_fmt(pre_dealer['share'])} (gap {_fmt(pre_dealer['gap_shopper_minus_enthusiast'])}, interval {_fmt(pre_dealer['gap_lo'])} to {_fmt(pre_dealer['gap_hi'])}). "
            + (f"Support is still {_fmt(pre_support['share'])}." if pre_support is not None else "")
        )
        lines.append("")
    if contrasts is not None and len(contrasts):
        hit = contrasts[
            (contrasts["slice"] == "pre_purchase")
            & (contrasts["event_id"] == "tesla_price_cut_2023")
            & (contrasts["baseline"] == "prior_3_months")
            & (contrasts["barrier"] == "vehicle price or total cost")
        ]
        if len(hit):
            row = hit.iloc[0]
            lines.append(
                f"The pre-purchase price contrast for January 2023 versus the prior three months is {_fmt(row['diff'])} "
                f"({_fmt(row['lo'])} to {_fmt(row['hi'])}, event n={int(row['n_event'])}, baseline n={int(row['n_baseline'])})."
            )
    return lines


def upsert_marked(path: Path, start: str, end: str, section: str) -> None:
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    existing = path.read_text() if path.exists() else ""
    if start in existing and end in existing:
        head, rest = existing.split(start, 1)
        _, tail = rest.split(end, 1)
        path.write_text(head.rstrip() + "\n\n" + block + tail.lstrip("\n"))
        return
    path.write_text(existing.rstrip() + "\n\n" + block)


def upsert_round5_findings(findings_path, out_dir) -> None:
    upsert_marked(
        Path(findings_path),
        "<!-- round5-start -->",
        "<!-- round5-end -->",
        render_round5_markdown(out_dir),
    )


def render_feasibility_markdown(card_dir) -> str:
    card_dir = Path(card_dir)
    lines = [
        "## Round 5 feasibility",
        "",
        "These notes are about what this machine could pull. They are not new labeled results unless a later paragraph gives a count from a finished extract.",
        "",
    ]
    dense = card_dir / "dense_pull_probe.json"
    if dense.exists():
        import json

        payload = json.loads(dense.read_text())
        lines.append("### Denser enthusiast text")
        lines.append("")
        lines.append(payload.get("prose", ""))
        lines.append("")
    shop = card_dir / "shopper_2026_feasibility.json"
    if shop.exists():
        import json

        payload = json.loads(shop.read_text())
        lines.append("### January–September 2026 shopper dumps")
        lines.append("")
        lines.append(payload.get("prose", ""))
        lines.append("")
    if len(lines) <= 4:
        lines.append("The feasibility notes have not been written yet.")
        lines.append("")
    return "\n".join(lines)


def upsert_round5_card(card_path) -> None:
    path = Path(card_path)
    card_dir = path.parent
    upsert_marked(path, "<!-- round5-feasibility-start -->", "<!-- round5-feasibility-end -->", render_feasibility_markdown(card_dir))
