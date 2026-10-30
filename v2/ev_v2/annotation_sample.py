"""Stratified annotation draw for the Framing A validation package.

The sheet is 800 submissions, the size already fixed in the research design,
so two people can finish it. The draw is not a plain year-band sample. Rare
barrier positives and the four event windows are filled first. Prevalence
cells fill what remains, and the total is trimmed back to 800 by dropping
prevalence and event rows before any rare-barrier row.

No model label is written onto the coder sheet. Predictions live in a
separate key so a coder cannot see them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_v2.agreement import (
    BARRIER_CODES,
    BARRIER_FROM_MODEL,
    STAGE_FROM_MODEL,
    STANCE_FROM_MODEL,
)
from ev_v2.framing import AUDIENCES, FRAMING_EVENT_IDS, barrier_column

# Rare labels first so incentives and dealer friction are not whatever is left over.
BARRIER_DRAW_ORDER = (
    "incentives_policy",
    "dealer_friction",
    "environment",
    "range",
    "battery_reliability",
    "charging_access",
    "price_tco",
)

CODE_TO_MODEL_COLUMN = {code: barrier_column(label) for label, code in BARRIER_FROM_MODEL.items()}

EVENT_MONTHS = {
    "ira_2022": ["2022-08"],
    "tesla_price_cut_2023": ["2023-01"],
    "nacs_2023": ["2023-05", "2023-06"],
    "credit_deadline_2025": ["2025-09", "2025-10"],
}

YEAR_BANDS = ("2015-2017", "2018-2020", "2021-2023", "2024-latest")

# The coder sheet uses the 6-stage funnel. Onboarding is a stage, not a side flag.
# calibration is filled in by the draw (yes or blank). Every code column stays empty.
SHEET_COLUMNS = [
    "id",
    "subreddit",
    "title",
    "text_excerpt",
    "calibration",
    "purchase_selection",
    "charging_energy",
    "battery_range",
    "cost_incentives",
    "dealer_access",
    "reliability_service",
    "policy_market",
    "transition_narratives",
    "charging_access",
    "range",
    "price_tco",
    "incentives_policy",
    "battery_reliability",
    "dealer_friction",
    "environment",
    "none",
    "stage",
    "stance",
    "annotator",
    "notes",
]

TARGET_N = 800
POSITIVE_FLOOR = 40
EVENT_QUOTA = 25
CELL_QUOTA = 50
SEED = 20261019
CALIBRATION_N = 50
STAGE_FLOOR = 20
FAMILY_FLOOR = 15


def year_band(month: str) -> str:
    year = int(str(month)[:4])
    if year <= 2017:
        return "2015-2017"
    if year <= 2020:
        return "2018-2020"
    if year <= 2023:
        return "2021-2023"
    return "2024-latest"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def prepare_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    work["year_band"] = work["month"].map(year_band)
    for code, column in CODE_TO_MODEL_COLUMN.items():
        work[code] = _as_bool(work[column]) if column in work.columns else False
    return work


def draw_annotation_sample(
    frame: pd.DataFrame,
    *,
    target: int = TARGET_N,
    positive_floor: int = POSITIVE_FLOOR,
    event_quota: int = EVENT_QUOTA,
    cell_quota: int = CELL_QUOTA,
    seed: int = SEED,
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Return one row per drawn id, with the layer that added it.

    ``frame`` must already have audience, month, id, and the seven barrier
    boolean columns from ``prepare_label_frame`` (the short code names).
    """
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    if exclude_ids:
        blocked = {str(item) for item in exclude_ids}
        work = work[~work["id"].isin(blocked)]
    if "year_band" not in work.columns:
        work["year_band"] = work["month"].astype(str).str.slice(0, 7).map(year_band)
    work = work.drop_duplicates("id", keep="first")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    selected: set[str] = set()

    def take(pool: pd.DataFrame, k: int, layer: str, detail: str) -> None:
        if k <= 0 or pool.empty:
            return
        rest = pool[~pool["id"].isin(selected)]
        if rest.empty:
            return
        k = min(int(k), int(len(rest)))
        chosen = rest.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
        for doc_id in chosen["id"].tolist():
            selected.add(str(doc_id))
            rows.append({"id": str(doc_id), "layer": layer, "detail": detail})

    def n_positive(code: str, audience: str) -> int:
        if not selected:
            return 0
        block = work[(work["id"].isin(selected)) & (work["audience"] == audience)]
        return int(block[code].sum()) if code in block.columns else 0

    for audience in AUDIENCES:
        group = work[work["audience"] == audience]
        for code in BARRIER_DRAW_ORDER:
            have = n_positive(code, audience)
            pool = group[group[code].fillna(False)]
            take(pool, positive_floor - have, "barrier", f"{audience}|{code}")
    for audience in AUDIENCES:
        group = work[work["audience"] == audience]
        for event_id in FRAMING_EVENT_IDS:
            months = EVENT_MONTHS[event_id]
            pool = group[group["month"].isin(months)]
            already = int(pool["id"].isin(selected).sum()) if len(pool) else 0
            take(pool, event_quota - already, "event", f"{audience}|{event_id}")
    for audience in AUDIENCES:
        group = work[work["audience"] == audience]
        for band in YEAR_BANDS:
            pool = group[group["year_band"] == band]
            already = int(pool["id"].isin(selected).sum()) if len(pool) else 0
            take(pool, cell_quota - already, "prevalence", f"{audience}|{band}")
    # Fill to the target, alternating audiences so one group does not take the remainder.
    guard = 0
    while len(rows) < target and guard < target * 4:
        guard += 1
        progressed = False
        for audience in AUDIENCES:
            if len(rows) >= target:
                break
            rest = work[(work["audience"] == audience) & (~work["id"].isin(selected))]
            if rest.empty:
                continue
            take(rest, 1, "fill", audience)
            progressed = True
        if not progressed:
            break
    if len(rows) > target:
        kept: list[dict] = []
        spare = [row for row in rows if row["layer"] != "barrier"]
        n_barrier = len(rows) - len(spare)
        room = max(0, target - n_barrier)
        # Drop fill first, then prevalence, then event.
        rank = {"event": 0, "prevalence": 1, "fill": 2}
        spare_sorted = sorted(spare, key=lambda row: rank.get(row["layer"], 0))
        kept.extend(row for row in rows if row["layer"] == "barrier")
        kept.extend(spare_sorted[:room])
        rows = kept
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["id", "layer", "detail"])
    return out.drop_duplicates("id", keep="first").reset_index(drop=True)


def sample_composition(draw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Positives per barrier and audience inside the draw. Used to show the floors landed."""
    merged = draw.merge(labels, on="id", how="left")
    rows = []
    for audience in AUDIENCES:
        block = merged[merged["audience"] == audience]
        for code in list(BARRIER_DRAW_ORDER) + ["none"]:
            if code == "none":
                k = int((~block[list(BARRIER_DRAW_ORDER)].any(axis=1)).sum()) if len(block) else 0
            else:
                k = int(block[code].fillna(False).sum()) if code in block.columns else 0
            rows.append({"audience": audience, "code": code, "n_posts": int(len(block)), "n_positive": k})
        for event_id, months in EVENT_MONTHS.items():
            rows.append(
                {
                    "audience": audience,
                    "code": f"event:{event_id}",
                    "n_posts": int(len(block)),
                    "n_positive": int(block["month"].isin(months).sum()) if len(block) else 0,
                }
            )
    return pd.DataFrame(rows)


def draw_funnel_sample(
    frame: pd.DataFrame,
    *,
    target: int = TARGET_N,
    stage_floor: int = STAGE_FLOOR,
    family_floor: int = FAMILY_FLOOR,
    positive_floor: int = 15,
    event_quota: int = 15,
    seed: int = SEED,
    exclude_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Draw posts so each audience covers the stages and families we can see.

    ``frame`` needs ``sampling_stage`` and boolean columns named by family code.
    Those columns are sampling aids from the small model and a lexicon. They
    are not written on the coder sheet.
    """
    from ev_v2.taxonomy import FAMILY_CODES, STAGE_CODES

    work = frame.copy()
    work["id"] = work["id"].astype(str)
    if exclude_ids:
        work = work[~work["id"].isin({str(item) for item in exclude_ids})]
    work = work.drop_duplicates("id", keep="first")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    selected: set[str] = set()
    stage_order = [code for code in ("evaluation", "awareness", "onboarding", "purchase", "long_term_ownership", "consideration", "ownership_unsplit", "unclear") if code in set(work["sampling_stage"])]
    # STAGE_CODES includes unclear; the rare journey stages are requested first.
    for code in STAGE_CODES:
        if code not in stage_order and code in set(work["sampling_stage"]):
            stage_order.append(code)

    def take(pool: pd.DataFrame, k: int, layer: str, detail: str) -> None:
        if k <= 0 or pool.empty:
            return
        rest = pool[~pool["id"].isin(selected)]
        if rest.empty:
            return
        k = min(int(k), int(len(rest)))
        chosen = rest.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
        for doc_id in chosen["id"].tolist():
            selected.add(str(doc_id))
            rows.append({"id": str(doc_id), "layer": layer, "detail": detail})

    def n_selected(pool: pd.DataFrame) -> int:
        if pool.empty or not selected:
            return 0
        return int(pool["id"].isin(selected).sum())

    for audience in AUDIENCES:
        group = work[work["audience"] == audience]
        for stage in stage_order:
            pool = group[group["sampling_stage"] == stage]
            take(pool, stage_floor - n_selected(pool), "stage", f"{audience}|{stage}")
        for family in FAMILY_CODES:
            if family not in group.columns:
                continue
            pool = group[group[family].fillna(False).astype(bool)]
            take(pool, family_floor - n_selected(pool), "family", f"{audience}|{family}")
        for code in BARRIER_DRAW_ORDER:
            if code not in group.columns:
                continue
            pool = group[group[code].fillna(False).astype(bool)]
            take(pool, positive_floor - n_selected(pool), "barrier", f"{audience}|{code}")
        for event_id, months in EVENT_MONTHS.items():
            pool = group[group["month"].isin(months)]
            n_already = int(pool["id"].isin(selected).sum()) if len(pool) else 0
            take(pool, event_quota - n_already, "event", f"{audience}|{event_id}")
    guard = 0
    while len(rows) < target and guard < target * 4:
        guard += 1
        progressed = False
        for audience in AUDIENCES:
            if len(rows) >= target:
                break
            rest = work[(work["audience"] == audience) & (~work["id"].isin(selected))]
            if rest.empty:
                continue
            take(rest, 1, "fill", audience)
            progressed = True
        if not progressed:
            break
    if len(rows) > target:
        rank = {"stage": 0, "family": 1, "barrier": 2, "event": 3, "fill": 4}
        protected = [row for row in rows if row["layer"] in {"stage", "family", "barrier"}]
        spare = [row for row in rows if row["layer"] not in {"stage", "family", "barrier"}]
        spare_sorted = sorted(spare, key=lambda row: rank.get(row["layer"], 9))
        room = max(0, target - len(protected))
        rows = protected + spare_sorted[:room]
        if len(rows) > target:
            # Still over: drop family rows after the stage rows.
            stage_rows = [row for row in rows if row["layer"] == "stage"]
            rest = [row for row in rows if row["layer"] != "stage"]
            room = max(0, target - len(stage_rows))
            rows = stage_rows + rest[:room]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["id", "layer", "detail"])
    return out.drop_duplicates("id", keep="first").reset_index(drop=True)


def calibration_ids(draw: pd.DataFrame, frame: pd.DataFrame, n: int = CALIBRATION_N, seed: int = SEED) -> set[str]:
    """Flag a subset of the draw, spread across audiences and sampling stages."""
    if draw.empty or n <= 0:
        return set()
    merged = draw.merge(frame[["id", "audience", "sampling_stage"]], on="id", how="left")
    rng = np.random.default_rng(seed + 1)
    picked: list[str] = []
    # One from each audience-stage cell first, then fill at random.
    cells = merged.groupby(["audience", "sampling_stage"], dropna=False)
    for _, block in cells:
        if len(picked) >= n:
            break
        choice = block.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        picked.append(str(choice.iloc[0]["id"]))
    rest = merged[~merged["id"].astype(str).isin(picked)]
    need = n - len(picked)
    if need > 0 and len(rest):
        extra = rest.sample(n=min(need, len(rest)), random_state=int(rng.integers(0, 2**31 - 1)))
        picked.extend(extra["id"].astype(str).tolist())
    return set(picked[:n])


def blank_sheet(
    labels: pd.DataFrame,
    draw: pd.DataFrame,
    text_by_id: dict[str, dict] | None = None,
    calibration: set[str] | None = None,
) -> pd.DataFrame:
    """Coder sheet. Code columns are empty. Text is title plus excerpt when the fetch has it."""
    text_by_id = text_by_id or {}
    meta = labels.drop_duplicates("id").set_index("id")
    rows = []
    for doc_id in draw["id"].tolist():
        info = meta.loc[doc_id] if doc_id in meta.index else None
        fetched = text_by_id.get(str(doc_id), {})
        title = fetched.get("title") or ""
        excerpt = fetched.get("text_excerpt") or ""
        subreddit = ""
        if info is not None:
            subreddit = str(info["subreddit"])
        flagged = "yes" if calibration and str(doc_id) in calibration else ""
        rows.append(
            {
                "id": doc_id,
                "subreddit": subreddit,
                "title": title,
                "text_excerpt": excerpt,
                "calibration": flagged,
                "purchase_selection": "",
                "charging_energy": "",
                "battery_range": "",
                "cost_incentives": "",
                "dealer_access": "",
                "reliability_service": "",
                "policy_market": "",
                "transition_narratives": "",
                "charging_access": "",
                "range": "",
                "price_tco": "",
                "incentives_policy": "",
                "battery_reliability": "",
                "dealer_friction": "",
                "environment": "",
                "none": "",
                "stage": "",
                "stance": "",
                "annotator": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=SHEET_COLUMNS)


def model_key(labels: pd.DataFrame, draw: pd.DataFrame) -> pd.DataFrame:
    """Model codes on the codebook scale, for the scorer only."""
    meta = labels.drop_duplicates("id").set_index("id")
    rows = []
    for doc_id in draw["id"].tolist():
        info = meta.loc[doc_id]
        row = {"id": doc_id}
        any_on = False
        for code in BARRIER_CODES:
            if code == "none":
                continue
            on = bool(info[code]) if code in info.index else False
            row[code] = int(on)
            any_on = any_on or on
        row["none"] = int(not any_on)
        row["stage"] = STAGE_FROM_MODEL.get(str(info.get("stage_label", "")), "")
        row["onboarding"] = ""
        row["stance"] = STANCE_FROM_MODEL.get(str(info.get("stance_label", "")), "")
        rows.append(row)
    columns = ["id", *BARRIER_CODES, "stage", "onboarding", "stance"]
    return pd.DataFrame(rows, columns=columns)
