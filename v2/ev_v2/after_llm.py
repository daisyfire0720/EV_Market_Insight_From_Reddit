"""Tables that wait for an instruction-model label file.

Nothing here invents a label. Rows the model did not return, or returned
with ``ok`` false, are counted and left out. Human agreement runs only when
both coder sheets are passed in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ev_v2.agreement import cohen_kappa, score_coders
from ev_v2.rdit import CUTOFFS, rdit_table
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


def funnel_shares(labels: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = usable_labels(labels).merge(meta[["id", "audience"]], on="id", how="inner")
    rows = []
    for audience, block in merged.groupby("audience"):
        n = len(block)
        for stage, k in block["stage"].value_counts().items():
            rows.append({"audience": audience, "stage": stage, "k": int(k), "n": int(n), "share": k / n if n else float("nan"), "validated": False})
    return pd.DataFrame(rows)


def stage_family_heatmap(labels: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = usable_labels(labels).merge(meta[["id", "audience"]], on="id", how="inner")
    rows = []
    for audience, block in merged.groupby("audience"):
        for stage in list(STAGE_ORDER) + ["unclear"]:
            stage_rows = block[block["stage"] == stage]
            n_stage = int(len(stage_rows))
            for family in FAMILY_CODES:
                k = int(stage_rows["families"].map(lambda value, code=family: code in _split_codes(value)).sum()) if n_stage else 0
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
    return pd.DataFrame(rows)


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


def _with_outcomes(labels: pd.DataFrame, meta: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    work = usable_labels(labels).merge(meta, on="id", how="inner")
    if len(authors):
        work = work.merge(authors[["id", "author_hash", "created_utc"]], on="id", how="left")
    for aspect, column in ASPECT_TO_BARRIER.items():
        work[column] = work["aspects"].map(lambda value, code=aspect: code in _split_codes(value))
    return work


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
    if "score" in prepared.columns or score_path is None or not Path(score_path).is_file():
        return prepared
    scores = pd.read_csv(score_path)
    if "id" not in scores.columns or "score" not in scores.columns:
        return prepared
    scores = scores[["id", "score"]].copy()
    scores["id"] = scores["id"].astype(str)
    return prepared.merge(scores, on="id", how="left")


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
    prepared = attach_scores(_with_outcomes(raw, small[meta_cols], authors), score_path)
    shares = funnel_shares(raw, small[meta_cols])
    heat = stage_family_heatmap(raw, small[meta_cols])
    pieces = []
    for audience in prepared["audience"].dropna().unique():
        pieces.append(rdit_table(prepared, audience=str(audience), n_boot=n_boot))
    rdit = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    coverage, shocks, matrix = transition_tables(prepared)
    agreement = small_model_agreement(raw, small)
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
    filtered, score_info = score_subset(prepared, min_score=min_score)
    pd.DataFrame([score_info]).to_csv(out_dir / "score_sensitivity.csv", index=False)
    if score_info["status"] == f"score_ge_{min_score}" and len(filtered):
        funnel_shares(raw[raw["id"].astype(str).isin(set(filtered["id"]))], small[meta_cols]).to_csv(
            out_dir / "funnel_shares_score_ge_2.csv", index=False
        )
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
    return {"validity": report, "shares": shares, "heatmap": heat, "rdit": rdit, "agreement": agreement}
