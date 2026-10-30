"""Tables that wait for an instruction-model label file.

Nothing here invents a label. Rows the model did not return, or returned
with ``ok`` false, are counted and left out. A comment is not dropped by an
inner join onto the submission file: audience comes from the parent post or
the subreddit, and every other row is named in the drop log. The main tables
stay submissions. The comment tables are a sensitivity and are not usable
for a discontinuity unless the windows have been rebalanced. Human agreement
runs only when both coder sheets are passed in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ev_v2.agreement import cohen_kappa, score_coders
from ev_v2.arctic import CORE_SUBREDDITS, GENERAL_SUBREDDITS
from ev_v2.rdit import CUTOFFS, _as_day, rdit_table
from ev_v2.taxonomy import ASPECT_CODES, FAMILY_CODES, SMALL_MODEL_STAGE, STAGE_CODES, STAGE_ORDER
from ev_v2.transitions import author_coverage, consecutive_pairs, forward_rate_around, transition_counts

ASPECT_TO_BARRIER = {
    "incentives_policy": "barrier_purchase_incentives_or_tax_credits_on",
    "price_tco": "barrier_vehicle_price_or_total_cost_on",
    "charging_access": "barrier_charging_access_on",
    "range": "barrier_driving_range_on",
    "battery_reliability": "barrier_battery_reliability_on",
    "dealer_friction": "barrier_dealer_or_buying_friction_on",
    "environment": "barrier_environmental_impact_on",
}

COMPARABLE_STAGES = ("awareness", "consideration", "evaluation", "purchase", "unclear")
_CORE = {name.casefold() for name in CORE_SUBREDDITS}
_GENERAL = {name.casefold() for name in GENERAL_SUBREDDITS}
WINDOW_DAYS = 180
DROP_REASONS = (
    "kept_submission",
    "kept_comment",
    "kept_other",
    "not_ok",
    "unknown_stage",
    "duplicate_id",
    "no_submission_meta",
    "no_audience",
)
COMMENT_RDIT_NOTE = (
    "Batch-1 comments were selected by closeness to a cutoff, so the two sides are not a balanced window. "
    "This sensitivity is not usable for RDiT unless the sample is rebalanced. "
    "The main specification remains submissions only."
)


def _split_codes(value) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def validity_report(labels: pd.DataFrame) -> pd.DataFrame:
    stage = labels["stage"].astype(str) if "stage" in labels.columns else pd.Series([""] * len(labels))
    ok = labels["ok"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes"}) if "ok" in labels.columns else pd.Series([True] * len(labels))
    known = stage.isin(STAGE_CODES)
    return pd.DataFrame(
        [
            {
                "n_rows": int(len(labels)),
                "n_ok": int(ok.sum()),
                "n_bad_json": int((~ok).sum()),
                "n_unknown_stage": int((ok & ~known).sum()),
                "n_used": int((ok & known).sum()),
                "validated": False,
            }
        ]
    )


def usable_labels(labels: pd.DataFrame) -> pd.DataFrame:
    work = labels.copy()
    work["id"] = work["id"].astype(str)
    if "ok" in work.columns:
        keep = work["ok"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
        work = work[keep]
    work = work[work["stage"].astype(str).isin(STAGE_CODES)]
    return work.drop_duplicates("id", keep="first")


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _series_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin({"", "nan", "none"})


def text_kind(frame: pd.DataFrame) -> pd.Series:
    """Missing text type is a submission. Any other value is kept, so a comment is not dropped."""
    if "text_type" not in frame.columns:
        return pd.Series("submission", index=frame.index)
    def one(value: object) -> str:
        if _is_blank(value):
            return "submission"
        kind = str(value).strip().lower()
        if kind in {"submission", "nan", "none"}:
            return "submission"
        return kind
    return frame["text_type"].map(one)


def parent_key(value) -> str:
    if _is_blank(value):
        return ""
    text = str(value).strip()
    if text.startswith("t3_"):
        text = text[3:]
    return text


def audience_from_subreddit(value) -> str:
    if _is_blank(value):
        return ""
    key = str(value).strip().casefold()
    if key in _CORE:
        return "enthusiast"
    if key in _GENERAL:
        return "shopper"
    return ""


def fill_comment_fields(labels: pd.DataFrame, comment_meta: pd.DataFrame | None) -> pd.DataFrame:
    """Fill subreddit, parent link, time, author, and score for comments only."""
    work = labels.copy()
    work["id"] = work["id"].astype(str)
    if comment_meta is None or len(comment_meta) == 0 or "id" not in getattr(comment_meta, "columns", []):
        return work
    meta = comment_meta.copy()
    meta["id"] = meta["id"].astype(str)
    meta = meta.drop_duplicates("id").set_index("id")
    comment_rows = text_kind(work).eq("comment")
    for column in ("subreddit", "link_id", "created_utc", "author_hash", "score", "draw_cutoff_id", "draw_side"):
        if column not in meta.columns:
            continue
        mapped = work["id"].map(meta[column])
        if column not in work.columns:
            work[column] = pd.Series(pd.NA, index=work.index, dtype="object")
        blank = comment_rows & _series_blank(work[column])
        work.loc[blank, column] = mapped.loc[blank].to_numpy()
    return work


def _audience_map(meta: pd.DataFrame) -> dict[str, str]:
    if meta is None or len(meta) == 0 or "id" not in meta.columns or "audience" not in meta.columns:
        return {}
    cleaned = meta.copy()
    cleaned["id"] = cleaned["id"].astype(str)
    cleaned = cleaned.drop_duplicates("id")
    return dict(zip(cleaned["id"], cleaned["audience"].astype(str)))


def row_disposition(
    labels: pd.DataFrame,
    meta: pd.DataFrame,
    comment_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mark every input row kept or name the reason it is left out.

    Submissions take audience from the submission file. Comments take it from
    the parent submission when that parent is in the file, and otherwise from
    the subreddit. A comment with neither is counted as ``no_audience``. It is
    not dropped without a row in the log.
    """
    work = fill_comment_fields(labels, comment_meta)
    work["_kind"] = text_kind(work)
    if "ok" in work.columns:
        ok = work["ok"].map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
    else:
        ok = pd.Series(True, index=work.index)
    if "stage" in work.columns:
        stage_known = work["stage"].astype(str).isin(STAGE_CODES)
    else:
        stage_known = pd.Series(False, index=work.index)
    status = pd.Series("", index=work.index, dtype="object")
    status.loc[~ok] = "not_ok"
    status.loc[status.eq("") & ~stage_known] = "unknown_stage"
    status.loc[status.eq("") & work["id"].duplicated(keep="first")] = "duplicate_id"
    audience_map = _audience_map(meta)
    audiences: list[str] = []
    sources: list[str] = []
    final: list[str] = []
    has_link = "link_id" in work.columns
    has_sub = "subreddit" in work.columns
    for idx, row in work.iterrows():
        current = status.loc[idx]
        if current:
            final.append(str(current))
            audiences.append("")
            sources.append("")
            continue
        kind = str(row["_kind"])
        if kind == "submission":
            aud = audience_map.get(str(row["id"]), "")
            if aud in {"enthusiast", "shopper"}:
                final.append("kept_submission")
                audiences.append(aud)
                sources.append("submission_meta")
            else:
                final.append("no_submission_meta")
                audiences.append("")
                sources.append("")
            continue
        parent = parent_key(row["link_id"]) if has_link else ""
        parent_aud = audience_map.get(parent, "") if parent else ""
        if parent_aud in {"enthusiast", "shopper"}:
            final.append("kept_comment" if kind == "comment" else "kept_other")
            audiences.append(parent_aud)
            sources.append("parent_link")
            continue
        sub_aud = audience_from_subreddit(row["subreddit"]) if has_sub else ""
        if sub_aud:
            final.append("kept_comment" if kind == "comment" else "kept_other")
            audiences.append(sub_aud)
            sources.append("subreddit")
            continue
        final.append("no_audience")
        audiences.append("")
        sources.append("")
    work["status"] = final
    work["audience"] = audiences
    work["audience_source"] = sources
    return work


def drop_log_table(disposition: pd.DataFrame) -> pd.DataFrame:
    counts = disposition["status"].value_counts()
    rows = []
    seen = set(DROP_REASONS)
    for reason in DROP_REASONS:
        rows.append(
            {
                "reason": reason,
                "n": int(counts.get(reason, 0)),
                "role": "kept" if str(reason).startswith("kept_") else "dropped",
            }
        )
    for reason, n in counts.items():
        if reason not in seen:
            rows.append({"reason": str(reason), "n": int(n), "role": "dropped"})
    return pd.DataFrame(rows)


def shares_from_prepared(prepared: pd.DataFrame) -> pd.DataFrame:
    columns = ["audience", "stage", "k", "n", "share", "validated"]
    if prepared.empty or "audience" not in prepared.columns or "stage" not in prepared.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for audience, block in prepared.groupby("audience"):
        if audience not in {"enthusiast", "shopper"}:
            continue
        n = len(block)
        for stage, k in block["stage"].value_counts().items():
            rows.append(
                {
                    "audience": audience,
                    "stage": stage,
                    "k": int(k),
                    "n": int(n),
                    "share": k / n if n else float("nan"),
                    "validated": False,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def heatmap_from_prepared(prepared: pd.DataFrame) -> pd.DataFrame:
    columns = ["audience", "stage", "family", "n_stage", "n_family", "share", "validated"]
    if prepared.empty or "audience" not in prepared.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    families = prepared["families"] if "families" in prepared.columns else pd.Series([""] * len(prepared), index=prepared.index)
    staged = prepared.copy()
    staged["_families"] = families
    for audience, block in staged.groupby("audience"):
        if audience not in {"enthusiast", "shopper"}:
            continue
        for stage in list(STAGE_ORDER) + ["unclear"]:
            stage_rows = block[block["stage"] == stage]
            n_stage = int(len(stage_rows))
            for family in FAMILY_CODES:
                k = int(stage_rows["_families"].map(lambda value, code=family: code in _split_codes(value)).sum()) if n_stage else 0
                rows.append(
                    {
                        "audience": audience,
                        "stage": stage,
                        "family": family,
                        "n_stage": n_stage,
                        "n_family": k,
                        "share": (k / n_stage) if n_stage else float("nan"),
                        "validated": False,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def funnel_shares(labels: pd.DataFrame, meta: pd.DataFrame, comment_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Submission shares. Comments are not in this table; the drop log counts them."""
    prepared, _disposition = _with_outcomes(labels, meta, pd.DataFrame(), comment_meta)
    return shares_from_prepared(prepared)


def stage_family_heatmap(labels: pd.DataFrame, meta: pd.DataFrame, comment_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared, _disposition = _with_outcomes(labels, meta, pd.DataFrame(), comment_meta)
    return heatmap_from_prepared(prepared)


def write_heatmap_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audiences = list(table["audience"].unique())
    stages = [code for code in list(STAGE_ORDER) + ["unclear"]]
    families = list(FAMILY_CODES)
    fig, axes = plt.subplots(1, max(1, len(audiences)), figsize=(6 * max(1, len(audiences)), 5), squeeze=False)
    for index, audience in enumerate(audiences):
        ax = axes[0][index]
        grid = np.full((len(stages), len(families)), np.nan)
        block = table[table["audience"] == audience]
        for i, stage in enumerate(stages):
            for j, family in enumerate(families):
                cell = block[(block["stage"] == stage) & (block["family"] == family)]
                if len(cell):
                    grid[i, j] = cell.iloc[0]["share"]
        image = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages, fontsize=8)
        ax.set_title(f"{audience} (model labels, not human codes)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _with_outcomes(
    labels: pd.DataFrame,
    meta: pd.DataFrame,
    authors: pd.DataFrame,
    comment_meta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Submission rows for the main tables, plus a disposition row for every input row."""
    disposition = row_disposition(labels, meta, comment_meta)
    kept = disposition[disposition["status"].eq("kept_submission")].copy()
    if len(authors):
        authors = authors.copy()
        authors["id"] = authors["id"].astype(str)
        cols = [column for column in ("id", "author_hash", "created_utc") if column in authors.columns]
        kept = kept.drop(columns=[column for column in ("author_hash", "created_utc") if column in kept.columns], errors="ignore")
        kept = kept.merge(authors[cols], on="id", how="left")
    if "aspects" not in kept.columns:
        kept["aspects"] = ""
    for aspect, column in ASPECT_TO_BARRIER.items():
        kept[column] = kept["aspects"].map(lambda value, code=aspect: code in _split_codes(value))
    return kept, disposition


def comment_window_balance(comments: pd.DataFrame, window_days: int = WINDOW_DAYS, note: str = "Closest-to-cutoff batch 1, not a balanced window.") -> pd.DataFrame:
    """How many kept comments sit on each side of each cutoff.

    The cutoff day is counted on its own. A side more than twice the other is
    marked unbalanced. The flag does not make the sample usable for RDiT.
    """
    columns = [
        "cutoff_id",
        "cutoff",
        "audience",
        "window_days",
        "n_left",
        "n_right",
        "n_cutoff_day",
        "n_outside_window",
        "n_missing_time",
        "n_comments",
        "unbalanced",
        "note",
    ]
    if comments.empty or "audience" not in comments.columns:
        return pd.DataFrame(columns=columns)
    if "created_utc" in comments.columns:
        days = comments["created_utc"].map(_as_day)
    else:
        days = pd.Series([None] * len(comments), index=comments.index)
    rows = []
    for spec in CUTOFFS:
        cutoff = spec["cutoff"]
        for audience in ("enthusiast", "shopper"):
            mask = comments["audience"] == audience
            block_days = days[mask]
            n_missing = int(sum(item is None for item in block_days))
            n_left = n_right = n_on = n_out = 0
            for item in block_days:
                if item is None:
                    continue
                delta = (item - cutoff).days
                if abs(delta) > window_days:
                    n_out += 1
                elif delta < 0:
                    n_left += 1
                elif delta > 0:
                    n_right += 1
                else:
                    n_on += 1
            bigger = max(n_left, n_right)
            smaller = min(n_left, n_right)
            unbalanced = bool(bigger > 0 and (smaller == 0 or bigger > 2 * smaller))
            rows.append(
                {
                    "cutoff_id": spec["cutoff_id"],
                    "cutoff": cutoff.isoformat(),
                    "audience": audience,
                    "window_days": window_days,
                    "n_left": n_left,
                    "n_right": n_right,
                    "n_cutoff_day": n_on,
                    "n_outside_window": n_out,
                    "n_missing_time": n_missing,
                    "n_comments": int(mask.sum()),
                    "unbalanced": unbalanced,
                    "note": note,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def comments_sensitivity_table(
    disposition: pd.DataFrame,
    balance: pd.DataFrame,
    *,
    comments_rebalanced: bool = False,
) -> pd.DataFrame:
    kept = disposition[disposition["status"].eq("kept_comment")]
    dropped = disposition[text_kind(disposition).eq("comment") & ~disposition["status"].isin(["kept_comment", "kept_other"])]
    parent_n = int((kept["audience_source"] == "parent_link").sum()) if len(kept) else 0
    subreddit_n = int((kept["audience_source"] == "subreddit").sum()) if len(kept) else 0
    usable = bool(comments_rebalanced)
    note = (
        "The caller marked these comment windows as rebalanced. They are still a sensitivity, not the main specification."
        if usable
        else COMMENT_RDIT_NOTE
    )
    rows = [
        {"item": "n_comment_rows", "value": int(text_kind(disposition).eq("comment").sum())},
        {"item": "n_kept_comment", "value": int(len(kept))},
        {"item": "n_dropped_comment", "value": int(len(dropped))},
        {"item": "n_audience_parent_link", "value": parent_n},
        {"item": "n_audience_subreddit", "value": subreddit_n},
        {"item": "n_no_audience", "value": int((disposition["status"] == "no_audience").sum())},
        {"item": "rdit_usable", "value": usable},
        {"item": "main_specification", "value": "submissions"},
        {"item": "note", "value": note},
    ]
    if len(balance) and "unbalanced" in balance.columns:
        rows.append({"item": "n_unbalanced_cutoff_audiences", "value": int(balance["unbalanced"].sum())})
    return pd.DataFrame(rows)


def rdit_comments_own_draw(comments: pd.DataFrame, n_boot: int = 0) -> pd.DataFrame:
    """Main and donut fits per cutoff, using only the comments drawn for that cutoff.

    Each comment in the v2 draw belongs to one cutoff (``draw_cutoff_id``).
    Placebo rows are dropped here because the draw does not reach the placebo dates.
    """
    if comments.empty or "draw_cutoff_id" not in comments.columns:
        return pd.DataFrame()
    pieces = []
    for spec in CUTOFFS:
        block = comments[comments["draw_cutoff_id"].astype(str).eq(spec["cutoff_id"])]
        if block.empty:
            continue
        for audience in ("enthusiast", "shopper"):
            sub = block[block["audience"].eq(audience)]
            if sub.empty:
                continue
            table = rdit_table(sub, audience=audience, n_boot=n_boot)
            table = table[table["cutoff_id"].eq(spec["cutoff_id"]) & table["fit"].isin(["main", "donut"])]
            pieces.append(table.assign(sample="own_draw"))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def small_model_agreement(labels: pd.DataFrame, small: pd.DataFrame) -> pd.DataFrame:
    left = usable_labels(labels)
    right = small.copy()
    right["id"] = right["id"].astype(str)
    both = left.merge(right, on="id", how="inner", suffixes=("_llm", "_small"))
    rows = []
    small_stage = both["stage_label"].map(lambda value: SMALL_MODEL_STAGE.get(str(value), "unclear")) if "stage_label" in both.columns else pd.Series([""] * len(both))
    llm_stage = both["stage"]
    comparable = small_stage.isin(COMPARABLE_STAGES)
    if int(comparable.sum()):
        kappa = cohen_kappa(small_stage[comparable].tolist(), llm_stage[comparable].tolist())
    else:
        kappa = {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    rows.append(
        {
            "field": "stage_comparable",
            "n": kappa["n"],
            "percent_agreement": kappa["agreement"],
            "cohen_kappa": kappa["kappa"],
            "n_small_ownership_unsplit": int((small_stage == "ownership").sum()),
            "note": "Ownership on the small model is not scored against onboarding or long-term ownership.",
            "validated": False,
        }
    )
    for aspect, column in ASPECT_TO_BARRIER.items():
        if column not in both.columns:
            continue
        pred = both[column].fillna(False).astype(bool).astype(int).tolist()
        truth = both["aspects"].map(lambda value, code=aspect: int(code in _split_codes(value))).tolist()
        kappa = cohen_kappa(pred, truth)
        rows.append(
            {
                "field": aspect,
                "n": kappa["n"],
                "percent_agreement": kappa["agreement"],
                "cohen_kappa": kappa["kappa"],
                "n_small_ownership_unsplit": 0,
                "note": "Small-model barrier versus instruction-model aspect. Not a human gold label.",
                "validated": False,
            }
        )
    rows.append(
        {
            "field": "family",
            "n": 0,
            "percent_agreement": float("nan"),
            "cohen_kappa": float("nan"),
            "n_small_ownership_unsplit": 0,
            "note": "The small model has no topic-family field. No family agreement is computed.",
            "validated": False,
        }
    )
    return pd.DataFrame(rows)


def transition_tables(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "author_hash" not in prepared.columns or "created_utc" not in prepared.columns:
        empty = pd.DataFrame()
        return empty, empty, empty
    work = prepared.copy()
    work["stage_label"] = work["stage"]
    coverage = author_coverage(work)
    pairs = consecutive_pairs(work, order_map=STAGE_ORDER)
    from datetime import datetime, timezone

    rows = []
    for spec in CUTOFFS:
        cutoff = datetime(spec["cutoff"].year, spec["cutoff"].month, spec["cutoff"].day, tzinfo=timezone.utc).timestamp()
        for shift, name in ((0, "main"), (-180, "placebo_-180d"), (180, "placebo_+180d")):
            rate = forward_rate_around(pairs, cutoff + shift * 86400, window_days=180)
            rows.append({"cutoff_id": spec["cutoff_id"], "fit": name, "validated": False, **rate})
    return coverage, pd.DataFrame(rows), transition_counts(pairs)


def llm_model_key(labels: pd.DataFrame) -> pd.DataFrame:
    work = usable_labels(labels)
    rows = []
    for row in work.itertuples(index=False):
        families = _split_codes(getattr(row, "families", ""))
        aspects = _split_codes(getattr(row, "aspects", ""))
        item = {"id": str(row.id), "stage": str(row.stage), "stance": str(getattr(row, "stance", ""))}
        any_aspect = False
        for code in ASPECT_CODES:
            if code == "none":
                continue
            on = int(code in aspects)
            item[code] = on
            any_aspect = any_aspect or bool(on)
        item["none"] = int(not any_aspect)
        for code in FAMILY_CODES:
            item[code] = int(code in families)
        rows.append(item)
    return pd.DataFrame(rows)


def attach_scores(prepared: pd.DataFrame, score_path: Path | None) -> pd.DataFrame:
    """Fill blank scores from the score file. A score already on the row is kept."""
    if score_path is None or not Path(score_path).is_file():
        return prepared
    scores = pd.read_csv(score_path)
    if "id" not in scores.columns or "score" not in scores.columns:
        return prepared
    scores = scores[["id", "score"]].copy()
    scores["id"] = scores["id"].astype(str)
    scores = scores.drop_duplicates("id").rename(columns={"score": "_score_file"})
    work = prepared.copy()
    work["id"] = work["id"].astype(str)
    work = work.merge(scores, on="id", how="left")
    if "score" not in work.columns:
        work["score"] = work["_score_file"]
    else:
        blank = _series_blank(work["score"])
        work.loc[blank, "score"] = work.loc[blank, "_score_file"].to_numpy()
    return work.drop(columns=["_score_file"])


def score_subset(prepared: pd.DataFrame, min_score: int = 2) -> tuple[pd.DataFrame, dict]:
    """Drop items with score under min_score. Missing scores are counted and left out of this check."""
    if "score" not in prepared.columns:
        return prepared.iloc[0:0].copy(), {
            "status": "no_scores",
            "min_score": min_score,
            "n_in": int(len(prepared)),
            "n_kept": 0,
            "n_dropped": 0,
            "n_missing_score": int(len(prepared)),
        }
    score = pd.to_numeric(prepared["score"], errors="coerce")
    kept = prepared[score >= min_score].copy()
    return kept, {
        "status": f"score_ge_{min_score}",
        "min_score": min_score,
        "n_in": int(len(prepared)),
        "n_kept": int(len(kept)),
        "n_dropped": int((score < min_score).sum()),
        "n_missing_score": int(score.isna().sum()),
    }


def _load_comment_meta(path: Path | None, labels: pd.DataFrame) -> pd.DataFrame | None:
    if path is None or not Path(path).is_file():
        return None
    if "text_type" not in labels.columns or not text_kind(labels).eq("comment").any():
        return None
    return pd.read_csv(path)


def run_tables(
    llm_path: Path,
    small_path: Path,
    author_path: Path,
    out_dir: Path,
    coder_a: Path | None = None,
    coder_b: Path | None = None,
    n_boot: int = 200,
    score_path: Path | None = None,
    min_score: int = 2,
    comment_meta_path: Path | None = None,
    comments_rebalanced: bool = False,
    comment_n_boot: int = 0,
) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(llm_path)
    report = validity_report(raw)
    small = pd.read_csv(small_path)
    small["id"] = small["id"].astype(str)
    meta_cols = [col for col in ("id", "audience", "subreddit", "month") if col in small.columns]
    authors = pd.read_csv(author_path) if author_path.is_file() else pd.DataFrame(columns=["id", "author_hash", "created_utc"])
    if len(authors):
        authors["id"] = authors["id"].astype(str)
    comment_meta = _load_comment_meta(comment_meta_path, raw)
    prepared, disposition = _with_outcomes(raw, small[meta_cols], authors, comment_meta)
    prepared = attach_scores(prepared, score_path)
    shares = shares_from_prepared(prepared)
    heat = heatmap_from_prepared(prepared)
    pieces = []
    for audience in prepared["audience"].dropna().unique():
        pieces.append(rdit_table(prepared, audience=str(audience), n_boot=n_boot))
    rdit = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    coverage, shocks, matrix = transition_tables(prepared)
    agreement = small_model_agreement(raw, small)
    logged = drop_log_table(disposition)
    comments = disposition[disposition["status"].eq("kept_comment")].copy()
    comment_shares = shares_from_prepared(comments)
    balance_note = (
        "All kept comments in the window. A balanced draw is balanced on its own cutoff (draw_cutoff_id), not on every window."
        if comments_rebalanced
        else "Closest-to-cutoff batch 1, not a balanced window."
    )
    balance = comment_window_balance(comments, note=balance_note)
    sensitivity = comments_sensitivity_table(disposition, balance, comments_rebalanced=comments_rebalanced)
    report.to_csv(out_dir / "llm_validity.csv", index=False)
    shares.to_csv(out_dir / "funnel_shares_by_audience.csv", index=False)
    heat.to_csv(out_dir / "stage_family_heatmap.csv", index=False)
    if len(heat):
        write_heatmap_figure(heat, out_dir / "stage_family_heatmap.png")
    rdit.to_csv(out_dir / "rdit.csv", index=False)
    coverage.to_csv(out_dir / "author_coverage.csv", index=False)
    shocks.to_csv(out_dir / "stage_transition_shocks.csv", index=False)
    matrix.to_csv(out_dir / "stage_transition_counts.csv", index=False)
    agreement.to_csv(out_dir / "small_vs_llm.csv", index=False)
    logged.to_csv(out_dir / "row_drop_log.csv", index=False)
    comment_shares.to_csv(out_dir / "funnel_shares_comments.csv", index=False)
    balance.to_csv(out_dir / "comment_window_balance.csv", index=False)
    sensitivity.to_csv(out_dir / "comments_sensitivity.csv", index=False)
    if comments_rebalanced and len(comments):
        # Point estimates only, and only after the windows have been rebuilt.
        # Batch 1 as pulled is not this branch.
        comment_prepared = comments.copy()
        if "aspects" not in comment_prepared.columns:
            comment_prepared["aspects"] = ""
        for aspect, column in ASPECT_TO_BARRIER.items():
            comment_prepared[column] = comment_prepared["aspects"].map(lambda value, code=aspect: code in _split_codes(value))
        comment_pieces = [
            rdit_table(comment_prepared, audience=str(audience), n_boot=comment_n_boot)
            for audience in comment_prepared["audience"].dropna().unique()
        ]
        if comment_pieces:
            pd.concat(comment_pieces, ignore_index=True).to_csv(out_dir / "rdit_comments_rebalanced.csv", index=False)
        own = rdit_comments_own_draw(comment_prepared, n_boot=comment_n_boot)
        if len(own):
            own.to_csv(out_dir / "rdit_comments_own_draw.csv", index=False)
    filtered, score_info = score_subset(prepared, min_score=min_score)
    pd.DataFrame([score_info]).to_csv(out_dir / "score_sensitivity.csv", index=False)
    if score_info["status"] == f"score_ge_{min_score}" and len(filtered):
        shares_from_prepared(filtered).to_csv(out_dir / "funnel_shares_score_ge_2.csv", index=False)
        sens_pieces = [rdit_table(filtered, audience=str(audience), n_boot=0) for audience in filtered["audience"].dropna().unique()]
        if sens_pieces:
            pd.concat(sens_pieces, ignore_index=True).to_csv(out_dir / "rdit_score_ge_2.csv", index=False)
        _coverage, sens_shocks, _matrix = transition_tables(filtered)
        sens_shocks.to_csv(out_dir / "stage_transition_shocks_score_ge_2.csv", index=False)
    status = {"coder_sheets": "not_provided", "n_joined": 0}
    if coder_a and coder_b and Path(coder_a).is_file() and Path(coder_b).is_file():
        tables = score_coders(pd.read_csv(coder_a), pd.read_csv(coder_b), model_key=llm_model_key(raw), n_boot=min(n_boot, 200))
        for name, frame in tables.items():
            frame.to_csv(out_dir / f"human_{name}.csv", index=False)
        status = {"coder_sheets": "scored", "n_joined": int(tables["n_joined"].iloc[0]["n_joined"])}
    pd.DataFrame([status]).to_csv(out_dir / "annotator_status.csv", index=False)
    return {
        "validity": report,
        "shares": shares,
        "heatmap": heat,
        "rdit": rdit,
        "agreement": agreement,
        "drop_log": logged,
        "comment_shares": comment_shares,
        "comment_balance": balance,
        "comments_sensitivity": sensitivity,
    }
