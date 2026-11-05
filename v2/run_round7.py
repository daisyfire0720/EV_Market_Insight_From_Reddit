#!/usr/bin/env python3
"""Rebuild the funnel annotation sheet and the causal scaffolds.

The sheet's code cells stay empty. Sampling strata are written only under
``v2/outputs/``. Labels stay unvalidated.

From ``v2/``::

    python run_round7.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotation_sample import (  # noqa: E402
    blank_sheet,
    calibration_ids,
    draw_funnel_sample,
    model_key,
    prepare_label_frame,
)
from ev_v2.rdit import CUTOFFS, rdit_table  # noqa: E402
from ev_v2.round7_report import upsert_round7_findings  # noqa: E402
from ev_v2.taxonomy import FAMILY_CODES, sampling_families, sampling_stage  # noqa: E402
from ev_v2.transitions import author_coverage, consecutive_pairs, forward_rate_around, transition_counts  # noqa: E402

LABELS = ROOT / "outputs" / "framing_labels.csv"
CACHE = ROOT / "data" / "raw" / "label_text.jsonl"
AUTHORS = ROOT / "outputs" / "post_author_hashes.csv"
OUT = ROOT / "outputs"
ANNOT = ROOT / "annotation"
_USER = re.compile(r"/?u/[A-Za-z0-9_-]+")


def _redact(text: str) -> str:
    return _USER.sub("u/[redacted]", text or "")


def _load_text() -> dict[str, dict]:
    have: dict[str, dict] = {}
    if not CACHE.exists():
        return have
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        have[str(row["id"])] = row
    return have


def _with_sampling(labels: pd.DataFrame, text: dict[str, dict]) -> pd.DataFrame:
    work = prepare_label_frame(labels)
    stages = []
    family_cols = {code: [] for code in FAMILY_CODES}
    for row in work.itertuples(index=False):
        blob = ""
        cached = text.get(str(row.id))
        if cached:
            blob = f"{cached.get('title') or ''}\n{cached.get('excerpt') or ''}"
        stages.append(sampling_stage(row.stage_label, blob))
        flags = {code: bool(getattr(row, code)) for code in (
            "charging_access",
            "range",
            "price_tco",
            "incentives_policy",
            "battery_reliability",
            "dealer_friction",
            "environment",
        )}
        found = sampling_families(flags, row.stage_label, blob)
        for code in FAMILY_CODES:
            family_cols[code].append(code in found)
    work["sampling_stage"] = stages
    for code, values in family_cols.items():
        work[code] = values
    return work


def _sheet_text(text: dict[str, dict], ids: list[str]) -> dict[str, dict]:
    out = {}
    for doc_id in ids:
        cached = text.get(str(doc_id))
        if not cached:
            continue
        title = _redact(str(cached.get("title") or ""))[:300]
        excerpt = _redact(str(cached.get("excerpt") or title))
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "…"
        out[str(doc_id)] = {"title": title, "text_excerpt": excerpt}
    return out


def _exclude_preview() -> set[str]:
    path = ANNOT / "annotation_preview.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["id"].astype(str))


def _coverage(draw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    merged = draw.merge(frame, on="id", how="left")
    rows = []
    for audience, block in merged.groupby("audience"):
        for stage, n in block["sampling_stage"].value_counts().items():
            rows.append({"audience": audience, "stratum": stage, "kind": "stage", "n": int(n)})
        for code in FAMILY_CODES:
            if code not in block.columns:
                continue
            rows.append({"audience": audience, "stratum": code, "kind": "family", "n": int(block[code].fillna(False).sum())})
    return pd.DataFrame(rows)


def _funnel_shares(labels: pd.DataFrame) -> pd.DataFrame:
    from ev_v2.taxonomy import SMALL_MODEL_STAGE

    work = labels.copy()
    work["stage"] = work["stage_label"].map(lambda value: SMALL_MODEL_STAGE.get(str(value), "unclear"))
    rows = []
    for audience, block in work.groupby("audience"):
        n = len(block)
        for stage, k in block["stage"].value_counts().items():
            rows.append({"audience": audience, "stage": stage, "k": int(k), "n": int(n), "share": k / n, "validated": False})
    return pd.DataFrame(rows)


def _rdit(labels: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    work = labels.merge(authors[["id", "created_utc"]], on="id", how="left")
    work["created_utc"] = pd.to_numeric(work["created_utc"], errors="coerce")
    pieces = []
    for audience in ("enthusiast", "shopper"):
        pieces.append(rdit_table(work, audience=audience))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _transitions(labels: pd.DataFrame, authors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = labels.merge(authors, on="id", how="left")
    coverage = author_coverage(work)
    pairs = consecutive_pairs(work)
    rows = []
    for spec in CUTOFFS:
        cutoff = datetime(spec["cutoff"].year, spec["cutoff"].month, spec["cutoff"].day, tzinfo=timezone.utc).timestamp()
        for shift, name in ((0, "main"), (-180, "placebo_-180d"), (180, "placebo_+180d")):
            rate = forward_rate_around(pairs, cutoff + shift * 86400, window_days=180)
            rows.append({"cutoff_id": spec["cutoff_id"], "fit": name, "validated": False, **rate})
    return coverage, pd.DataFrame(rows), transition_counts(pairs)


def main() -> int:
    labels = pd.read_csv(LABELS)
    labels["id"] = labels["id"].astype(str)
    text = _load_text()
    frame = _with_sampling(labels, text)
    draw = draw_funnel_sample(frame, exclude_ids=_exclude_preview())
    flagged = calibration_ids(draw, frame)
    sheet_text = _sheet_text(text, draw["id"].tolist())
    sheet = blank_sheet(frame, draw, sheet_text, flagged)
    key = model_key(frame, draw)
    key["model_name"] = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    key["validated"] = False
    # ``detail`` names the sampling stratum. It stays in outputs, not in the folder the coder opens.
    draw.to_csv(OUT / "annotation_funnel_draw.csv", index=False)
    manifest = draw.drop(columns=["detail"]).merge(frame[["id", "audience", "subreddit", "month", "year_band"]], on="id", how="left")
    manifest["calibration"] = manifest["id"].isin(flagged)
    ANNOT.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(ANNOT / "annotation_sheet.csv", index=False)
    key.to_csv(ANNOT / "annotation_model_key.csv", index=False)
    manifest.to_csv(ANNOT / "sample_manifest.csv", index=False)
    coverage = _coverage(draw, frame)
    coverage.to_csv(OUT / "annotation_funnel_coverage.csv", index=False)
    meta = pd.DataFrame(
        [
            {
                "n": int(len(sheet)),
                "n_enthusiast": int((manifest["audience"] == "enthusiast").sum()),
                "n_shopper": int((manifest["audience"] == "shopper").sum()),
                "n_with_text": int((sheet["text_excerpt"].fillna("").astype(str).str.len() > 0).sum()),
                "n_calibration": int((sheet["calibration"] == "yes").sum()),
                "validated": False,
            }
        ]
    )
    meta.to_csv(OUT / "annotation_funnel_meta.csv", index=False)
    _funnel_shares(labels).to_csv(OUT / "funnel_stage_shares.csv", index=False)
    authors = pd.read_csv(AUTHORS) if AUTHORS.exists() else pd.DataFrame(columns=["id", "author_hash", "created_utc"])
    authors["id"] = authors["id"].astype(str)
    print("rdit", flush=True)
    _rdit(labels, authors).to_csv(OUT / "rdit.csv", index=False)
    print("transitions", flush=True)
    coverage_authors, shocks, matrix = _transitions(labels, authors)
    coverage_authors.to_csv(OUT / "author_coverage.csv", index=False)
    shocks.to_csv(OUT / "stage_transition_shocks.csv", index=False)
    matrix.to_csv(OUT / "stage_transition_counts.csv", index=False)
    upsert_round7_findings(OUT / "FINDINGS.md", OUT)
    print(meta.to_string(index=False), flush=True)
    print(coverage_authors.to_string(index=False), flush=True)
    print("sheet codes empty; labels unvalidated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
