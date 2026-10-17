"""Zero-shot NLI labels for barriers, purchase-journey stage, and stance.

No paid API is used. The default model is a small MNLI checkpoint that runs
on CPU. Labels follow the draft codebook in ``research_design.md``. Scores
are stored so a threshold can be changed without another forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ev_v2.dealer import decide_dealer

# Multi-label barriers. A post can match more than one.
BARRIER_LABELS = (
    "charging access",
    "driving range",
    "vehicle price or total cost",
    "purchase incentives or tax credits",
    "battery reliability",
    "dealer or buying friction",
    "environmental impact",
)

# Single-label. The last class is the explicit residual.
STAGE_LABELS = (
    "learning about electric vehicles",
    "comparing electric vehicle options",
    "evaluating cost or fit before buying",
    "buying an electric vehicle",
    "owning an electric vehicle",
    "no clear purchase stage",
)

STANCE_LABELS = (
    "support for electric vehicles",
    "opposition to electric vehicles",
    "mixed or uncertain view of electric vehicles",
    "no stance on electric vehicles",
)

# Templates chosen on invented codebook sentences, before the corpus run.
# "This text is about {}." separated barrier labels on those sentences for
# MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33. A bare "{}." and
# "This Reddit post is about {}." did not, on typeform/distilbert-base-uncased-mnli.
BARRIER_HYPOTHESIS = "This text is about {}."
STAGE_HYPOTHESIS = "This text is about {}."
STANCE_HYPOTHESIS = "This text expresses {}."
DEFAULT_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"


@dataclass
class NliOutput:
    frame: pd.DataFrame
    model_name: str
    model_revision: str | None


def _load(model_name: str):
    from transformers import pipeline

    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
        truncation=True,
    )
    revision = None
    model = getattr(clf, "model", None)
    config = getattr(model, "config", None)
    if config is not None:
        revision = getattr(config, "_commit_hash", None) or getattr(config, "name_or_path", None)
    return clf, revision


def _one(clf, text: str, labels: tuple[str, ...], hypothesis: str, multi_label: bool) -> dict[str, float]:
    # The pipeline truncates, but very long strings still waste time.
    snippet = text if len(text) <= 1200 else text[:1200]
    result = clf(
        snippet,
        candidate_labels=list(labels),
        hypothesis_template=hypothesis,
        multi_label=multi_label,
    )
    return {label: float(score) for label, score in zip(result["labels"], result["scores"])}


def _apply_dealer(row: dict, text: str, threshold: float) -> dict:
    score = float(row.get("barrier_dealer_or_buying_friction") or 0.0)
    decision = decide_dealer(text or "", score, threshold)
    row["barrier_dealer_or_buying_friction_on"] = decision.on
    row["dealer_rule"] = decision.reason
    return row


def _load_checkpoint(path: Path | None) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done[str(rec["id"])] = rec
    return done


def annotate_frame(
    frame: pd.DataFrame,
    text_col: str = "text",
    model_name: str = DEFAULT_MODEL,
    barrier_threshold: float = 0.5,
    checkpoint_path: Path | None = None,
    progress_every: int = 50,
) -> NliOutput:
    if frame.empty:
        return NliOutput(pd.DataFrame(), model_name, None)
    texts = frame[text_col].tolist()
    ids = frame["id"].tolist() if "id" in frame.columns else list(range(len(frame)))
    text_by_id = {str(doc_id): text for doc_id, text in zip(ids, texts)}
    done = _load_checkpoint(checkpoint_path)
    pending = [(doc_id, text) for doc_id, text in zip(ids, texts) if str(doc_id) not in done]
    revision = None
    if pending:
        clf, revision = _load(model_name)
        handle = None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            handle = checkpoint_path.open("a", encoding="utf-8")
        try:
            for i, (doc_id, text) in enumerate(pending, start=1):
                row = _score_one(clf, doc_id, text, model_name, barrier_threshold)
                done[str(doc_id)] = row
                if handle is not None:
                    handle.write(json.dumps(_jsonable(row)) + "\n")
                    if i % 20 == 0:
                        handle.flush()
                if progress_every and (i % progress_every == 0 or i == len(pending)):
                    print(f"  NLI {i}/{len(pending)}", flush=True)
        finally:
            if handle is not None:
                handle.close()
    rows = []
    for doc_id in ids:
        row = dict(done[str(doc_id)])
        _apply_dealer(row, text_by_id.get(str(doc_id), ""), barrier_threshold)
        rows.append(row)
    if revision is None and rows:
        revision = rows[0].get("nli_revision")
    return NliOutput(pd.DataFrame(rows), model_name, revision)


def _jsonable(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def _score_one(clf, doc_id, text: str, model_name: str, barrier_threshold: float) -> dict:
    barriers = _one(clf, text, BARRIER_LABELS, BARRIER_HYPOTHESIS, multi_label=True)
    stage = _one(clf, text, STAGE_LABELS, STAGE_HYPOTHESIS, multi_label=False)
    stance = _one(clf, text, STANCE_LABELS, STANCE_HYPOTHESIS, multi_label=False)
    row = {"id": doc_id, "nli_model": model_name}
    for label, score in barriers.items():
        key = "barrier_" + label.replace(" ", "_")
        row[key] = score
        row[key + "_on"] = bool(score >= barrier_threshold)
    row["stage_label"] = max(stage, key=stage.get)
    row["stage_score"] = stage[row["stage_label"]]
    for label, score in stage.items():
        row["stage_" + label.replace(" ", "_")] = score
    row["stance_label"] = max(stance, key=stance.get)
    row["stance_score"] = stance[row["stance_label"]]
    for label, score in stance.items():
        row["stance_" + label.replace(" ", "_")] = score
    decision = decide_dealer(text, row["barrier_dealer_or_buying_friction"], barrier_threshold)
    row["barrier_dealer_or_buying_friction_on"] = decision.on
    row["dealer_rule"] = decision.reason
    model = getattr(clf, "model", None)
    config = getattr(model, "config", None)
    if config is not None:
        row["nli_revision"] = getattr(config, "_commit_hash", None)
    return row
