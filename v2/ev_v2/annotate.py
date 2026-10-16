"""Zero-shot NLI labels for barriers, purchase-journey stage, and stance.

No paid API is used. The default model is a small MNLI checkpoint that runs
on CPU. Labels follow the draft codebook in ``research_design.md``. Scores
are stored so a threshold can be changed without another forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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


def annotate_frame(
    frame: pd.DataFrame,
    text_col: str = "text",
    model_name: str = DEFAULT_MODEL,
    barrier_threshold: float = 0.5,
) -> NliOutput:
    if frame.empty:
        return NliOutput(pd.DataFrame(), model_name, None)
    clf, revision = _load(model_name)
    rows = []
    texts = frame[text_col].tolist()
    ids = frame["id"].tolist() if "id" in frame.columns else list(range(len(frame)))
    for doc_id, text in zip(ids, texts):
        barriers = _one(clf, text, BARRIER_LABELS, BARRIER_HYPOTHESIS, multi_label=True)
        stage = _one(clf, text, STAGE_LABELS, STAGE_HYPOTHESIS, multi_label=False)
        stance = _one(clf, text, STANCE_LABELS, STANCE_HYPOTHESIS, multi_label=False)
        row = {"id": doc_id, "nli_model": model_name}
        for label, score in barriers.items():
            key = "barrier_" + label.replace(" ", "_")
            row[key] = score
            row[key + "_on"] = score >= barrier_threshold
        row["stage_label"] = max(stage, key=stage.get)
        row["stage_score"] = stage[row["stage_label"]]
        for label, score in stage.items():
            row["stage_" + label.replace(" ", "_")] = score
        row["stance_label"] = max(stance, key=stance.get)
        row["stance_score"] = stance[row["stance_label"]]
        for label, score in stance.items():
            row["stance_" + label.replace(" ", "_")] = score
        rows.append(row)
    return NliOutput(pd.DataFrame(rows), model_name, revision)
