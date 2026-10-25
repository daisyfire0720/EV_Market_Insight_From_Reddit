"""Quality checks, remap sensitivity inputs, and comment metadata for the 14B labels.

Reads outputs/llm_labels.csv and the gitignored text files under data/raw.
Writes small QC tables to outputs/after_llm/qc_*.csv and, under data/raw
(gitignored), a remapped label file and metadata that lets run_after_llm.py
see comments. Nothing here changes a model label in place.

    PYTHONUTF8=1 python qc_14b.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.arctic import CORE_SUBREDDITS, GENERAL_SUBREDDITS  # noqa: E402
from ev_v2.taxonomy import ASPECT_CODES, FAMILY_CODES, STAGE_CODES  # noqa: E402

LABELS = ROOT / "outputs" / "llm_labels.csv"
SMALL = ROOT / "outputs" / "framing_labels.csv"
AUTHORS = ROOT / "outputs" / "post_author_hashes.csv"
SCORES = ROOT / "outputs" / "post_scores.csv"
SUB_TEXT = ROOT / "data" / "raw" / "label_posts.csv"
COM_TEXT = ROOT / "data" / "raw" / "comment_label_batch1_posts.csv"
QC_DIR = ROOT / "outputs" / "after_llm"
RAW = ROOT / "data" / "raw"

# Level mixups: a code from the other list, moved to the list where it exists.
# Near-synonyms are mapped only when the codebook meaning is plain.
ASPECT_TO_FAMILY = {code: code for code in FAMILY_CODES}
ASPECT_TO_FAMILY.update({"reliability": "reliability_service", "maintenance": "reliability_service", "policy": "policy_market", "regulations": "policy_market"})
ASPECT_SYNONYM = {"resale_value": "price_tco"}
FAMILY_TO_ASPECT = {code: code for code in ASPECT_CODES if code != "none"}
FAMILY_SYNONYM = {}
# A family code returned as the stage carries no journey position. The
# remap moves the code to families and sets the stage to unclear.
STAGE_FAMILY_TO_UNCLEAR = set(FAMILY_CODES)


def split(value: str) -> list[str]:
    return [part for part in str(value or "").split("|") if part]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def loose(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def load() -> pd.DataFrame:
    labels = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    small = pd.read_csv(SMALL, dtype=str, keep_default_na=False)[["id", "audience", "subreddit"]]
    subs = pd.read_csv(SUB_TEXT, dtype=str, keep_default_na=False)
    subs["text"] = (subs["title"] + "\n" + subs["selftext"]).str.strip()
    coms = pd.read_csv(COM_TEXT, dtype=str, keep_default_na=False)
    coms["text"] = coms["body"]
    coms["audience"] = coms["subreddit"].map(lambda s: "enthusiast" if s in CORE_SUBREDDITS else ("shopper" if s in GENERAL_SUBREDDITS else ""))
    text = pd.concat([subs[["id", "text"]], coms[["id", "text"]]]).drop_duplicates("id")
    aud = pd.concat([small[["id", "audience", "subreddit"]], coms[["id", "audience", "subreddit"]]]).drop_duplicates("id")
    out = labels.merge(text, on="id", how="left").merge(aud, on="id", how="left")
    out["audience"] = out["audience"].fillna("")
    return out


def distributions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, audience), block in frame.groupby(["text_type", "audience"]):
        n = len(block)
        for field, multi in (("stage", False), ("stance", False), ("families", True), ("aspects", True)):
            if multi:
                counts = pd.Series([code for value in block[field] for code in split(value)]).value_counts()
                empty = int((block[field] == "").sum())
                counts = pd.concat([counts, pd.Series({"(none)": empty})])
            else:
                counts = block[field].value_counts()
            for code, k in counts.items():
                rows.append({"text_type": kind, "audience": audience or "(no audience)", "field": field, "code": code, "k": int(k), "n": n, "share": k / n})
    return pd.DataFrame(rows)


def evidence_check(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["has_evidence"] = work["evidence"].str.strip() != ""
    work["exact"] = [bool(e.strip()) and norm(e) in norm(t) for e, t in zip(work["evidence"], work["text"].fillna(""))]
    work["loose"] = [bool(loose(e)) and loose(e) in loose(t) for e, t in zip(work["evidence"], work["text"].fillna(""))]
    rows = []
    for kind, block in list(work.groupby("text_type")) + [("all", work)]:
        has = block[block["has_evidence"]]
        rows.append(
            {
                "text_type": kind,
                "n": len(block),
                "n_with_evidence": len(has),
                "share_empty": 1 - len(has) / len(block),
                "verbatim_exact": has["exact"].mean(),
                "verbatim_ignoring_punctuation": has["loose"].mean(),
                "note": "Share of non-empty evidence strings found in the post text. Exact ignores case and whitespace only.",
            }
        )
    return pd.DataFrame(rows)


def unknown_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = []
    for row in frame.itertuples(index=False):
        for item in split(row.unknown_codes):
            level, _, code = item.partition(":")
            codes.append({"text_type": row.text_type, "level": level, "code": code})
    table = pd.DataFrame(codes)
    summary = table.groupby(["level", "code"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    flagged = frame[frame["unknown_codes"] != ""]
    head = pd.DataFrame(
        [
            {"level": "(rows)", "code": f"rows_flagged_{kind}", "n": int((flagged["text_type"] == kind).sum()), "of": int((frame["text_type"] == kind).sum())}
            for kind in ("submission", "comment")
        ]
    )
    stage_rows = frame[~frame["stage"].isin(STAGE_CODES)][["id", "text_type", "audience", "stage", "families", "aspects", "stance", "evidence", "unknown_codes"]]
    return pd.concat([head, summary], ignore_index=True), stage_rows


def remap(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    log = []
    for idx, row in out.iterrows():
        unknown = split(row["unknown_codes"])
        if not unknown:
            continue
        families = split(row["families"])
        aspects = split(row["aspects"])
        stage = row["stage"]
        left = []
        for item in unknown:
            level, _, code = item.partition(":")
            action = "kept_dropped"
            if level == "aspect" and code in ASPECT_TO_FAMILY:
                families.append(ASPECT_TO_FAMILY[code]); action = f"family:{ASPECT_TO_FAMILY[code]}"
            elif level == "aspect" and code in ASPECT_SYNONYM:
                aspects.append(ASPECT_SYNONYM[code]); action = f"aspect:{ASPECT_SYNONYM[code]}"
            elif level == "family" and code in FAMILY_TO_ASPECT:
                aspects.append(FAMILY_TO_ASPECT[code]); action = f"aspect:{FAMILY_TO_ASPECT[code]}"
            elif level == "stage" and code in STAGE_FAMILY_TO_UNCLEAR:
                families.append(code); stage = "unclear"; action = f"family:{code}+stage:unclear"
            else:
                left.append(item)
            log.append({"from": item, "to": action})
        out.at[idx, "families"] = "|".join(dict.fromkeys(f for f in families if f in FAMILY_CODES))
        out.at[idx, "aspects"] = "|".join(dict.fromkeys(a for a in aspects if a in ASPECT_CODES))
        out.at[idx, "stage"] = stage
        out.at[idx, "unknown_codes"] = "|".join(left)
    summary = pd.DataFrame(log).groupby(["from", "to"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    return out, summary


def comment_meta() -> None:
    small = pd.read_csv(SMALL, dtype=str, keep_default_na=False)
    coms = pd.read_csv(COM_TEXT, dtype=str, keep_default_na=False)
    coms["audience"] = coms["subreddit"].map(lambda s: "enthusiast" if s in CORE_SUBREDDITS else ("shopper" if s in GENERAL_SUBREDDITS else ""))
    coms["month"] = pd.to_datetime(pd.to_numeric(coms["created_utc"]), unit="s", utc=True).dt.strftime("%Y-%m")
    meta = pd.concat([small, coms[["id", "audience", "subreddit", "month"]]], ignore_index=True).drop_duplicates("id")
    meta.to_csv(RAW / "meta_with_comments.csv", index=False)
    authors = pd.read_csv(AUTHORS, dtype=str, keep_default_na=False)
    pd.concat([authors[["id", "author_hash", "created_utc"]], coms[["id", "author_hash", "created_utc"]]], ignore_index=True).drop_duplicates("id").to_csv(
        RAW / "authors_with_comments.csv", index=False
    )
    scores = pd.read_csv(SCORES, dtype=str, keep_default_na=False)
    pd.concat([scores[["id", "score"]], coms[["id", "score"]]], ignore_index=True).drop_duplicates("id").to_csv(RAW / "scores_with_comments.csv", index=False)


def main() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    frame = load()
    distributions(frame).to_csv(QC_DIR / "qc_distributions.csv", index=False)
    evidence_check(frame).to_csv(QC_DIR / "qc_evidence.csv", index=False)
    summary, stage_rows = unknown_tables(frame)
    summary.to_csv(QC_DIR / "qc_unknown_codes.csv", index=False)
    stage_rows.to_csv(QC_DIR / "qc_unknown_stage_rows.csv", index=False)
    spot = frame.sample(15, random_state=20261025)[["id", "text_type", "audience", "stage", "families", "aspects", "stance", "evidence", "text"]].copy()
    spot["text"] = spot["text"].str.slice(0, 600)
    # Text stays out of the committed folder; ids and labels only.
    spot.drop(columns=["text"]).to_csv(QC_DIR / "qc_spot_read_ids.csv", index=False)
    spot.to_csv(RAW / "qc_spot_read_with_text.csv", index=False)
    remapped, log = remap(frame)
    log.to_csv(QC_DIR / "qc_remap_log.csv", index=False)
    cols = list(pd.read_csv(LABELS, nrows=0).columns)
    remapped[cols].to_csv(RAW / "llm_labels_remapped.csv", index=False)
    comment_meta()
    print("done", len(frame))


if __name__ == "__main__":
    main()
