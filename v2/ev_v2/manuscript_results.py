"""Section 8 of the draft, read from the committed 14B tables.

Numbers come from ``v2/outputs/after_llm``. Nothing here invents a label.
The prose stays on the side of "consistent with". A jump is not an effect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STAGE_ROWS = (
    "awareness",
    "consideration",
    "evaluation",
    "purchase",
    "onboarding",
    "long_term_ownership",
    "unclear",
)

# These three fail a fake-date check: a placebo 180 days away is as large as the jump.
DEMOTED = (
    ("ira_signing_2022", "enthusiast", "The placebo 180 days earlier is as large as the jump, and the days after the cutoff are thin."),
    ("tesla_price_cut_2023", "enthusiast", "The placebo 180 days later is as large as the jump."),
    ("credit_end_2025", "shopper", "The interval includes zero, and the placebo 180 days later is larger than the jump and points the other way."),
)

HEADLINES = (
    ("ira_credit_rules_2023", "enthusiast"),
    ("tesla_price_cut_2023", "shopper"),
)


def _points(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    number = round(float(value) * 100, 1)
    if number == 0:
        return "0.0"
    return f"{number:+.1f}"


def _pct(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def _ci(row) -> str:
    lo = row["tau_lo"] if "tau_lo" in row.index else float("nan")
    hi = row["tau_hi"] if "tau_hi" in row.index else float("nan")
    if pd.isna(lo) or pd.isna(hi):
        return "interval not computed"
    return f"[{_points(lo)}, {_points(hi)}]"


def _main_row(frame: pd.DataFrame, cutoff_id: str, audience: str, fit: str = "main") -> pd.Series | None:
    if frame.empty:
        return None
    hit = frame[(frame["cutoff_id"] == cutoff_id) & (frame["audience"] == audience) & (frame["fit"] == fit)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _lookup(frame: pd.DataFrame, cutoff_id: str, audience: str, fit: str) -> pd.Series | None:
    return _main_row(frame, cutoff_id, audience, fit)


def instruction_results(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    after = out_dir / "after_llm"
    lines = [
        "## 8. What the 14B labels show",
        "",
        "**These are unvalidated model labels. Ryan has not coded the sheet, so the human kappa cell below is empty. "
        "A jump at a cutoff is consistent with a change in what was posted. It is not an effect of the statute or of the price cut. "
        "The main specification is submissions only.**",
        "",
    ]
    validity_path = after / "llm_validity.csv"
    if not validity_path.exists():
        lines.append("The 14B tables are not in this checkout. This section does not invent them.")
        return lines
    validity = pd.read_csv(validity_path).iloc[0]
    lines.extend(_run_lines(validity))
    lines.extend(_funnel_lines(after))
    lines.extend(_heatmap_lines(after))
    lines.extend(_rdit_lines(out_dir, after))
    lines.extend(_transition_lines(after))
    lines.extend(_kappa_lines(after))
    lines.extend(_comment_lines(after))
    lines.extend(_disclosure_lines(out_dir, after, validity))
    return lines


def _run_lines(validity: pd.Series) -> list[str]:
    n_rows = int(validity["n_rows"])
    n_unknown = int(validity["n_unknown_stage"])
    n_used = int(validity["n_used"])
    return [
        "### 8.1 The labeler",
        "",
        "The main labeler is Qwen2.5-14B-Instruct, served by Ollama as `qwen2.5:14b-instruct` in the Q4_K_M quant, temperature 0, prompt `funnel-v2`. "
        f"The label file has {n_rows:,} rows and {n_used:,} rows with a codebook stage. {n_unknown} rows put a topic-family code in the stage slot and are left out of the tables. "
        "The small zero-shot model in section 7 is the comparison, not the instrument. `qwen2.5:7b-instruct` was the fallback and was not the run. "
        "The finished labeling pass took about 13 hours on one RTX 4070. The planning rate of about 60 hours was the slow case, not the time that was used. "
        "Detail is in `v2/outputs/after_llm/FINDINGS_14B.md`.",
        "",
    ]


def _funnel_lines(after: Path) -> list[str]:
    path = after / "funnel_shares_by_audience.csv"
    lines = [
        "### 8.2 Six-stage shares",
        "",
        "The denominator is submissions with a codebook stage. Onboarding is separate from long-term ownership. The small model cannot make that split.",
        "",
    ]
    if not path.exists():
        lines.append("The share table is not in this checkout.")
        lines.append("")
        return lines
    shares = pd.read_csv(path)
    lines.append("| Stage | Enthusiast share | Shopper share |")
    lines.append("|---|---:|---:|")
    for stage in STAGE_ROWS:
        enth = shares[(shares["audience"] == "enthusiast") & (shares["stage"] == stage)]
        shop = shares[(shares["audience"] == "shopper") & (shares["stage"] == stage)]
        if enth.empty or shop.empty:
            continue
        lines.append(f"| {stage} | {_pct(enth.iloc[0]['share'])} (n={int(enth.iloc[0]['n']):,}) | {_pct(shop.iloc[0]['share'])} (n={int(shop.iloc[0]['n']):,}) |")
    lines.append("")
    enth_on = shares[(shares["audience"] == "enthusiast") & (shares["stage"] == "onboarding")]
    shop_aw = shares[(shares["audience"] == "shopper") & (shares["stage"] == "awareness")]
    if len(enth_on) and len(shop_aw):
        lines.append(
            f"Enthusiast posts sit with people who already have the car: onboarding is {_pct(enth_on.iloc[0]['share'])}. "
            f"Shopper posts sit earlier: awareness is {_pct(shop_aw.iloc[0]['share'])}. "
            "Purchase and long-term ownership are thin in both files, under 3%. Unclear is about a third of enthusiast posts and about two fifths of shopper posts."
        )
        lines.append("")
    score_path = after / "funnel_shares_score_ge_2.csv"
    if score_path.exists():
        scored = pd.read_csv(score_path)
        enth = scored[(scored["audience"] == "enthusiast") & (scored["stage"] == "onboarding")]
        shop = scored[(scored["audience"] == "shopper") & (scored["stage"] == "awareness")]
        if len(enth) and len(shop):
            lines.append(
                f"Dropping posts with score under 2 leaves the same picture: enthusiast onboarding {_pct(enth.iloc[0]['share'])}, shopper awareness {_pct(shop.iloc[0]['share'])}. "
                "That filter is a sensitivity. The table above keeps every submission."
            )
            lines.append("")
    return lines


def _heatmap_lines(after: Path) -> list[str]:
    path = after / "stage_family_heatmap.csv"
    lines = [
        "### 8.3 Stage-by-family heatmap",
        "",
        "Each cell is the share of submissions at that stage that carry the family. A post can carry more than one family. "
        "The grid is `v2/outputs/after_llm/stage_family_heatmap.csv` and the figure is `v2/outputs/after_llm/stage_family_heatmap.png`.",
        "",
    ]
    if not path.exists():
        return lines
    heat = pd.read_csv(path)
    lines.append("| Audience | Stage | Posts | Leading family | Share of that stage |")
    lines.append("|---|---|---:|---|---:|")
    for audience in ("enthusiast", "shopper"):
        block = heat[heat["audience"] == audience]
        for stage in STAGE_ROWS:
            cells = block[block["stage"] == stage]
            if cells.empty:
                continue
            n_stage = int(cells.iloc[0]["n_stage"])
            if n_stage == 0:
                continue
            top = cells.sort_values(["share", "family"], ascending=[False, True]).iloc[0]
            lines.append(
                f"| {audience} | {stage} | {n_stage:,} | {top['family']} | {_pct(top['share'])} |"
            )
    lines.append("")
    lines.append(
        "The leading family is a description of the grid, not a test. "
        "Dealer friction and environment are the aspects the spot-read trusts least. See section 8.6."
    )
    lines.append("")
    return lines


def _rdit_lines(out_dir: Path, after: Path) -> list[str]:
    lines = [
        "### 8.4 Discontinuity, submissions only",
        "",
        "The outcome is the pre-specified aspect: incentives at the credit cutoffs, price at the Tesla cutoff. "
        "The jump is in percentage points of the daily share. The interval is a week-cluster percentile interval from 1,000 draws, at the bandwidth chosen for the 14B labels. "
        "The small-model column is the same design on the section 7 labels, at the bandwidth that model chose. Bandwidths differ, so the grid rows in the two `rdit.csv` files are the fair comparison. "
        "None of these jumps is reported as an effect.",
        "",
    ]
    big = after / "rdit.csv"
    small = out_dir / "rdit.csv"
    if not big.exists():
        lines.append("The 14B discontinuity table is not in this checkout.")
        lines.append("")
        return lines
    left = pd.read_csv(big)
    right = pd.read_csv(small) if small.exists() else pd.DataFrame()
    lines.append("| Cutoff | Audience | 14B jump | 14B interval | Bandwidth | Placebo −180 / +180 days | Small model |")
    lines.append("|---|---|---:|---|---:|---|---|")
    order = []
    for cutoff_id, audience in left[left["fit"] == "main"][["cutoff_id", "audience"]].drop_duplicates().itertuples(index=False):
        order.append((cutoff_id, audience))
    for cutoff_id, audience in order:
        row = _lookup(left, cutoff_id, audience, "main")
        if row is None:
            continue
        minus = _lookup(left, cutoff_id, audience, "placebo_-180d")
        plus = _lookup(left, cutoff_id, audience, "placebo_+180d")
        small_row = _lookup(right, cutoff_id, audience, "main") if not right.empty else None
        placebo = ""
        if minus is not None and plus is not None:
            placebo = f"{_points(minus['tau'])} / {_points(plus['tau'])}"
        small_cell = ""
        if small_row is not None:
            small_cell = f"{_points(small_row['tau'])} ({int(small_row['bandwidth_days'])}d)"
        mark = ""
        if (cutoff_id, audience) in {(item[0], item[1]) for item in DEMOTED}:
            mark = " — demoted"
        lines.append(
            f"| {cutoff_id}{mark} | {audience} | {_points(row['tau'])} | {_ci(row)} | {int(row['bandwidth_days'])}d | {placebo} | {small_cell} |"
        )
    lines.append("")
    lines.extend(_headline_prose(left, right, after))
    lines.extend(_demoted_prose(left))
    return lines


def _headline_prose(left: pd.DataFrame, right: pd.DataFrame, after: Path) -> list[str]:
    lines = [
        "Two jumps are the ones to put beside the small model. Both intervals include zero. The language is \"consistent with,\" not a claim that the rule or the price cut caused the posts.",
        "",
    ]
    specs = (
        (
            "ira_credit_rules_2023",
            "enthusiast",
            "Enthusiast posts about incentives around the 1 January 2023 credit rules",
            (60, 90),
        ),
        (
            "tesla_price_cut_2023",
            "shopper",
            "Shopper posts about price around the 13 January 2023 Tesla cuts",
            (60, 90),
        ),
    )
    for cutoff_id, audience, label, alts in specs:
        row = _lookup(left, cutoff_id, audience, "main")
        small_row = _lookup(right, cutoff_id, audience, "main") if not right.empty else None
        if row is None:
            continue
        extra = []
        for alt in alts:
            grid = _lookup(left, cutoff_id, audience, f"bandwidth_{alt}")
            if grid is not None:
                extra.append(f"{_points(grid['tau'])} at {alt} days")
        small_bit = f" The small model, at its own bandwidth, is {_points(small_row['tau'])}." if small_row is not None else ""
        grid_bit = f" On the 14B grid the jump is {', and '.join(extra)}." if extra else ""
        lines.append(
            f"{label}: {_points(row['tau'])} points at {int(row['bandwidth_days'])} days, interval {_ci(row)}.{small_bit}{grid_bit} "
            "That is consistent with the small-model direction. It is not distinguishable from no jump."
        )
        lines.append("")
    score_path = after / "rdit_score_ge_2.csv"
    info_path = after / "score_sensitivity.csv"
    if score_path.exists():
        scored = pd.read_csv(score_path)
        names = {
            ("ira_credit_rules_2023", "enthusiast"): "The enthusiast credit-rules jump",
            ("tesla_price_cut_2023", "shopper"): "The shopper Tesla price jump",
        }
        bits = []
        for cutoff_id, audience in HEADLINES:
            row = _lookup(scored, cutoff_id, audience, "main")
            if row is None:
                continue
            bits.append(
                f"{names[(cutoff_id, audience)]} falls to {_points(row['tau'])} "
                f"(n={int(row['n_left']):,} before the cutoff and {int(row['n_right']):,} after)"
            )
        if bits:
            kept = ""
            if info_path.exists():
                info = pd.read_csv(info_path).iloc[0]
                kept = (
                    f" The score check keeps {int(info['n_kept']):,} of {int(info['n_in']):,} submissions "
                    f"and drops {int(info['n_dropped']):,} with score under 2"
                    f" ({int(info['n_missing_score'])} missing)."
                )
            lines.append(
                "The upvote filter removes both of those jumps. "
                + ". ".join(bits)
                + "."
                + kept
                + " Neither headline survives score at least 2. Report that as fragility. The main table still keeps every submission."
            )
            lines.append("")
    return lines


def _demoted_prose(left: pd.DataFrame) -> list[str]:
    lines = [
        "Three jumps fail the fake-date check in section 5. A placebo as large as the real date is a reason to doubt the jump. These three are not results.",
        "",
    ]
    for cutoff_id, audience, reason in DEMOTED:
        row = _lookup(left, cutoff_id, audience, "main")
        minus = _lookup(left, cutoff_id, audience, "placebo_-180d")
        plus = _lookup(left, cutoff_id, audience, "placebo_+180d")
        if row is None:
            continue
        placebo = ""
        if minus is not None and plus is not None:
            placebo = f" Placebos: {_points(minus['tau'])} and {_points(plus['tau'])}."
        lines.append(f"- {audience}, {cutoff_id}: {_points(row['tau'])} {_ci(row)}.{placebo} {reason}")
    lines.append("")
    return lines


def _transition_lines(after: Path) -> list[str]:
    lines = [
        "### 8.5 Transitions, descriptive only",
        "",
    ]
    path = after / "stage_transition_shocks.csv"
    if not path.exists():
        lines.append("The transition table is not in this checkout.")
        lines.append("")
        return lines
    shocks = pd.read_csv(path)
    main = shocks[shocks["fit"] == "main"]
    placebo = shocks[shocks["fit"] != "main"]
    bits = []
    for row in main.itertuples(index=False):
        bits.append(f"{row.cutoff_id} {int(row.n_forward)} of {int(row.n_pairs)} ({_pct(row.forward_rate)})")
    placebo_bit = ""
    if len(placebo):
        placebo_bit = f" Placebo rates run from {_pct(placebo['forward_rate'].min())} to {_pct(placebo['forward_rate'].max())}."
    lines.append(
        "Forward moves on the six-rung ladder, for pairs that cross a cutoff inside 180 days: "
        + "; ".join(bits)
        + "."
        + placebo_bit
        + " No cutoff stands out from its fake dates. The counts are too thin for a claim. They describe the file. They do not estimate how fast a person moves from shopping to buying."
    )
    lines.append("")
    authors = after / "author_coverage.csv"
    if authors.exists():
        frame = pd.read_csv(authors)
        overall = frame[frame["audience"] == "all"]
        if len(overall):
            row = overall.iloc[0]
            lines.append(
                f"Author hashes on the submission file: {int(row['n_posts']):,} posts, {int(row['n_authors']):,} authors, "
                f"{int(row['n_repeat_authors']):,} of them more than once ({_pct(row['share_posts_from_repeat_authors'])} of those posts). "
                f"The busiest hash has {int(row['max_posts_per_author'])} posts."
            )
            lines.append("")
    return lines


def _kappa_lines(after: Path) -> list[str]:
    lines = [
        "### 8.6 Agreement",
        "",
        "The first block is the small model against 14B. It is not a validity check. The human row is a placeholder. Ryan is coding `v2/annotation/annotation_sheet.csv`. This project did not contact him, and it does not invent his codes.",
        "",
        "| Comparison | n | Agreement | Cohen's kappa |",
        "|---|---:|---:|---:|",
    ]
    path = after / "small_vs_llm.csv"
    if path.exists():
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            kappa = "" if pd.isna(row.cohen_kappa) else f"{float(row.cohen_kappa):.2f}"
            agree = "" if pd.isna(row.percent_agreement) else f"{float(row.percent_agreement) * 100:.1f}%"
            n = int(row.n) if not pd.isna(row.n) else 0
            lines.append(f"| Small model vs 14B, {row.field} | {n:,} | {agree} | {kappa} |")
    lines.append("| Human kappa, Daisy and Ryan | — | — | placeholder. Ryan is coding. |")
    lines.append("")
    lines.append(
        "Stage kappa is about 0.11 on the posts where both models give a comparable stage. "
        "The small model has one ownership class, and those posts are not scored against onboarding or long-term ownership. "
        "Charging, price, and range are the aspects the two models share. Dealer friction and environment are low. "
        "A spot-read of 15 rows (seed 20261025, ids in `qc_spot_read_ids.csv`) found the model over-assigning aspects, especially those two. Treat them as low-confidence."
    )
    lines.append("")
    return lines


def _comment_lines(after: Path) -> list[str]:
    lines = [
        "### 8.7 Comments are a sensitivity, not a discontinuity",
        "",
    ]
    sens_path = after / "comments_sensitivity.csv"
    share_path = after / "funnel_shares_comments.csv"
    balance_path = after / "comment_window_balance.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path, dtype=str)
        values = dict(zip(sens["item"], sens["value"]))
        lines.append(
            f"The label file has {int(float(values.get('n_comment_rows', 0))):,} comment rows. "
            f"{int(float(values.get('n_kept_comment', 0))):,} have a codebook stage and an audience. "
            f"Audience comes from the parent submission when that submission is in the study file ({int(float(values.get('n_audience_parent_link', 0))):,} comments) "
            f"and otherwise from the subreddit ({int(float(values.get('n_audience_subreddit', 0))):,}). "
            f"Rows with neither are counted (`no_audience` = {int(float(values.get('n_no_audience', 0)))}), not dropped in silence. "
            f"`rdit_usable` is {values.get('rdit_usable', 'False')}. {values.get('note', '')}"
        )
        lines.append("")
    if share_path.exists():
        shares = pd.read_csv(share_path)
        unclear = shares[shares["stage"] == "unclear"]
        bits = [f"{row.audience} {_pct(row.share)} (n={int(row.n):,})" for row in unclear.itertuples(index=False)]
        if bits:
            lines.append(
                "Comment stage is mostly unclear: " + "; ".join(bits) + ". "
                "The model saw the comment and not the parent post, which is the likely reason. "
                "The comment share table is `v2/outputs/after_llm/funnel_shares_comments.csv`."
            )
            lines.append("")
    if balance_path.exists():
        balance = pd.read_csv(balance_path)
        tesla = balance[(balance["cutoff_id"] == "tesla_price_cut_2023") & (balance["audience"] == "enthusiast")]
        if len(tesla):
            row = tesla.iloc[0]
            lines.append(
                f"Inside 180 days of the Tesla cutoff, enthusiast comments with a codebook stage are "
                f"{int(row['n_left']):,} before the day, {int(row['n_cutoff_day']):,} on the day, and {int(row['n_right']):,} after. "
                "An exploratory file that was not committed, described in `FINDINGS_14B.md`, counted 13,374 left and 2,290 right under a different rule. "
                "Either way the sides are not a balanced window. The full count is `v2/outputs/after_llm/comment_window_balance.csv`. "
                "Batch 1 picked comments by closeness to a cutoff. That design is not usable for RDiT unless the windows are rebalanced."
            )
            lines.append("")
    return lines


def _disclosure_lines(out_dir: Path, after: Path, validity: pd.Series) -> list[str]:
    lines = [
        "### 8.8 What was left out of the label file",
        "",
    ]
    excluded = out_dir / "EXCLUDED_IDS.txt"
    if excluded.exists():
        text = excluded.read_text().strip()
        lines.append(f"One duplicate submission was excluded. `v2/outputs/EXCLUDED_IDS.txt` says: {text}")
        lines.append("")
    n_rows = int(validity["n_rows"])
    n_unknown = int(validity["n_unknown_stage"])
    unknown_path = after / "qc_unknown_codes.csv"
    flagged = ""
    performance = ""
    if unknown_path.exists():
        qc = pd.read_csv(unknown_path)
        flagged_rows = qc[qc["code"].astype(str).str.startswith("rows_flagged_")]
        n_flagged = int(flagged_rows["n"].sum()) if len(flagged_rows) else 0
        share = n_flagged / n_rows if n_rows else float("nan")
        flagged = f"Unknown codes outside the codebook: {n_flagged:,} of {n_rows:,} rows ({share * 100:.1f}%). "
        perf = qc[qc["code"] == "performance"]
        if len(perf):
            performance = f"The largest is `aspect:performance` ({int(perf.iloc[0]['n'])} rows). "
    lines.append(
        flagged
        + performance
        + "An unknown family or aspect code is dropped and the rest of that row is kept. "
        + f"An unknown stage is left out of every table ({n_unknown} rows: a family code where a stage should be). "
        "The raw code stays in `unknown_codes` when the label file has that column. "
        "A remap that moves level mixups back onto the right list, and that sets those 63 stages to unclear, changes funnel shares by at most 0.2 points and the main jumps by at most 0.3 points (`v2/outputs/after_llm_remapped/`). "
        "`performance` is not remapped. It is a codebook question, not a level mixup. See `v2/COMMENTS_V2_PLAN.md`."
    )
    lines.append("")
    return lines
