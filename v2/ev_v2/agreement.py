"""Inter-coder agreement, model-versus-human scores, and prevalence correction.

Human labels are whatever two coder sheets contain. This module does not
invent them. Items the two coders disagree on are left out of the gold set
and counted. They are not adjudicated here.

Prevalence correction is the Rogan-Gladen estimator. Sensitivity and
specificity come from the agreed items. The apparent rate is the model's
positive rate on the corpus the caller passes in.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

BARRIER_CODES = (
    "charging_access",
    "range",
    "price_tco",
    "incentives_policy",
    "battery_reliability",
    "dealer_friction",
    "environment",
    "none",
)

STAGE_CODES = (
    "awareness",
    "consideration",
    "evaluation",
    "purchase",
    "onboarding",
    "long_term_ownership",
    "ownership",
    "unclear",
)

STANCE_CODES = ("support", "oppose", "mixed", "no_stance")

ONBOARDING_CODES = ("yes", "no", "na")

STAGE_FROM_MODEL = {
    "learning about electric vehicles": "awareness",
    "comparing electric vehicle options": "consideration",
    "evaluating cost or fit before buying": "evaluation",
    "buying an electric vehicle": "purchase",
    "owning an electric vehicle": "ownership",
    "no clear purchase stage": "unclear",
}

STANCE_FROM_MODEL = {
    "support for electric vehicles": "support",
    "opposition to electric vehicles": "oppose",
    "mixed or uncertain view of electric vehicles": "mixed",
    "no stance on electric vehicles": "no_stance",
}

BARRIER_FROM_MODEL = {
    "charging access": "charging_access",
    "driving range": "range",
    "vehicle price or total cost": "price_tco",
    "purchase incentives or tax credits": "incentives_policy",
    "battery reliability": "battery_reliability",
    "dealer or buying friction": "dealer_friction",
    "environmental impact": "environment",
}


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def parse_binary(value) -> int | None:
    if _missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "off"}:
        return 0
    return None


def parse_nominal(value, allowed: tuple[str, ...]) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip().lower().replace(" ", "_")
    if text in allowed:
        return text
    return None


def cohen_kappa(left: list, right: list) -> dict:
    """Cohen's kappa for paired labels. Missing pairs must already be dropped."""
    n = len(left)
    if n == 0 or n != len(right):
        return {"n": n, "agreement": float("nan"), "kappa": float("nan"), "expected": float("nan")}
    hits = sum(a == b for a, b in zip(left, right))
    agreement = hits / n
    categories = set(left) | set(right)
    expected = 0.0
    for category in categories:
        expected += (left.count(category) / n) * (right.count(category) / n)
    if abs(1.0 - expected) < 1e-12:
        kappa = 1.0 if agreement == 1.0 else float("nan")
    else:
        kappa = (agreement - expected) / (1.0 - expected)
    return {"n": n, "agreement": agreement, "kappa": kappa, "expected": expected}


def krippendorff_alpha_nominal(pairs: list[tuple]) -> dict:
    """Nominal Krippendorff's alpha for two coders. A blank is missing, not a code."""
    coincidence: Counter[tuple] = Counter()
    n_units = 0
    n_skipped = 0
    for left, right in pairs:
        values = [value for value in (left, right) if value is not None]
        m = len(values)
        if m < 2:
            n_skipped += 1
            continue
        n_units += 1
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidence[(values[i], values[j])] += 1.0 / (m - 1)
    n = float(sum(coincidence.values()))
    if n <= 1.0:
        return {"n": n_units, "n_skipped": n_skipped, "alpha": float("nan")}
    categories = sorted({item for pair in coincidence for item in pair})
    marginals = [sum(coincidence[(c, k)] for k in categories) for c in categories]
    off = sum(weight for (c, k), weight in coincidence.items() if c != k)
    observed = off / n
    expected_agreement = sum(count * (count - 1.0) for count in marginals) / (n * (n - 1.0))
    expected = 1.0 - expected_agreement
    if expected <= 1e-12:
        alpha = float("nan")
    else:
        alpha = 1.0 - observed / expected
    return {"n": n_units, "n_skipped": n_skipped, "alpha": alpha}


def binary_scores(gold: list[int], pred: list[int]) -> dict:
    """Precision, recall, and F1 for the positive class."""
    tp = fp = fn = tn = 0
    for truth, guess in zip(gold, pred):
        if guess == 1 and truth == 1:
            tp += 1
        elif guess == 1 and truth == 0:
            fp += 1
        elif guess == 0 and truth == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "n": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "specificity": specificity,
    }


def rogan_gladen(apparent: float, sensitivity: float, specificity: float) -> dict:
    """Correct a model positive rate. Status is ``not_identified`` when the denominator is ~0."""
    if any(np.isnan(value) for value in (apparent, sensitivity, specificity)):
        return {"pi": float("nan"), "pi_unclipped": float("nan"), "denominator": float("nan"), "status": "not_identified"}
    denominator = sensitivity + specificity - 1.0
    if abs(denominator) < 1e-8:
        return {"pi": float("nan"), "pi_unclipped": float("nan"), "denominator": denominator, "status": "not_identified"}
    pi = (apparent - (1.0 - specificity)) / denominator
    return {
        "pi": float(np.clip(pi, 0.0, 1.0)),
        "pi_unclipped": float(pi),
        "denominator": float(denominator),
        "status": "ok",
    }


def _paired(frame: pd.DataFrame, column: str, parser) -> tuple[list, list, int]:
    left_raw = frame[f"{column}__a"]
    right_raw = frame[f"{column}__b"]
    left = []
    right = []
    blank_mismatch = 0
    for a, b in zip(left_raw, right_raw):
        pa, pb = parser(a), parser(b)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            blank_mismatch += 1
            continue
        left.append(pa)
        right.append(pb)
    return left, right, blank_mismatch


def _gold_mask(left: list, right: list) -> list[bool]:
    return [a == b for a, b in zip(left, right)]


def score_coders(
    coder_a: pd.DataFrame,
    coder_b: pd.DataFrame,
    model_key: pd.DataFrame | None = None,
    corpus_prevalence: dict[str, float] | None = None,
    n_boot: int = 1000,
    seed: int = 20261019,
) -> dict[str, pd.DataFrame]:
    """Agreement of two coder sheets, and model scores where a key is present.

    ``corpus_prevalence`` maps a barrier code to the model's positive rate on
    the corpus. When it is omitted, the Rogan-Gladen apparent rate is the
    model rate on the agreed items.
    """
    left = coder_a.copy()
    right = coder_b.copy()
    left["id"] = left["id"].astype(str)
    right["id"] = right["id"].astype(str)
    calibration_ids: set[str] = set()
    for sheet in (left, right):
        if "calibration" not in sheet.columns:
            continue
        flagged = sheet["calibration"].astype(str).str.strip().str.lower().isin({"yes", "1", "true", "y"})
        calibration_ids.update(sheet.loc[flagged, "id"].astype(str))
    both_all = left.merge(right, on="id", suffixes=("__a", "__b"), how="inner")
    both = both_all[~both_all["id"].isin(calibration_ids)].copy()
    calibration_both = both_all[both_all["id"].isin(calibration_ids)].copy()
    agreement_rows = []
    model_rows = []
    prevalence_rows = []
    rng = np.random.default_rng(seed)

    from ev_v2.taxonomy import FAMILY_CODES

    fields: list[tuple[str, object]] = [(code, parse_binary) for code in list(BARRIER_CODES) + list(FAMILY_CODES)]
    fields.append(("stage", lambda value: parse_nominal(value, STAGE_CODES)))
    fields.append(("onboarding", lambda value: parse_nominal(value, ONBOARDING_CODES)))
    fields.append(("stance", lambda value: parse_nominal(value, STANCE_CODES)))

    model = None
    if model_key is not None and len(model_key):
        model = model_key.copy()
        model["id"] = model["id"].astype(str)
        model = model.set_index("id")

    for column, parser in fields:
        if f"{column}__a" not in both.columns or f"{column}__b" not in both.columns:
            continue
        a_vals, b_vals, blank_mismatch = _paired(both, column, parser)
        kappa = cohen_kappa(a_vals, b_vals)
        alpha = krippendorff_alpha_nominal(list(zip(a_vals, b_vals)))
        # Alpha above used complete pairs only. Also report alpha with blanks as missing,
        # which is the same list once blanks are dropped. A second pass keeps the raw
        # blanks so a coder who skipped an item is missing rather than dropped from n.
        raw_pairs = []
        for a, b in zip(both[f"{column}__a"], both[f"{column}__b"]):
            raw_pairs.append((parser(a), parser(b)))
        alpha_missing = krippendorff_alpha_nominal(raw_pairs)
        n_disagree = sum(x != y for x, y in zip(a_vals, b_vals))
        boot = _bootstrap_kappa(a_vals, b_vals, rng, n_boot) if a_vals else (float("nan"), float("nan"))
        agreement_rows.append(
            {
                "field": column,
                "n_both_coded": kappa["n"],
                "n_disagree": n_disagree,
                "n_blank_mismatch": blank_mismatch,
                "percent_agreement": kappa["agreement"],
                "cohen_kappa": kappa["kappa"],
                "kappa_lo": boot[0],
                "kappa_hi": boot[1],
                "krippendorff_alpha": alpha_missing["alpha"],
                "alpha_n": alpha_missing["n"],
                "alpha_skipped": alpha_missing["n_skipped"],
            }
        )
        if model is None or column == "onboarding":
            continue
        if column not in model.columns and column not in BARRIER_CODES:
            continue
        _append_model_rows(
            column,
            both,
            parser,
            a_vals,
            b_vals,
            model,
            model_rows,
            prevalence_rows,
            corpus_prevalence or {},
            rng,
            n_boot,
        )

    calibration_rows = _agreement_rows(calibration_both, fields, rng, n_boot) if len(calibration_both) else []
    return {
        "agreement": pd.DataFrame(agreement_rows),
        "calibration_agreement": pd.DataFrame(calibration_rows),
        "model_vs_human": pd.DataFrame(model_rows),
        "prevalence": pd.DataFrame(prevalence_rows),
        "n_joined": pd.DataFrame(
            [{"n_joined": int(len(both)), "n_calibration_excluded": int(len(calibration_ids)), "n_calibration_joined": int(len(calibration_both))}]
        ),
    }


def _agreement_rows(both: pd.DataFrame, fields: list, rng: np.random.Generator, n_boot: int) -> list[dict]:
    rows = []
    if both.empty:
        return rows
    for column, parser in fields:
        if f"{column}__a" not in both.columns or f"{column}__b" not in both.columns:
            continue
        a_vals, b_vals, blank_mismatch = _paired(both, column, parser)
        kappa = cohen_kappa(a_vals, b_vals)
        raw_pairs = [(parser(a), parser(b)) for a, b in zip(both[f"{column}__a"], both[f"{column}__b"])]
        alpha_missing = krippendorff_alpha_nominal(raw_pairs)
        n_disagree = sum(x != y for x, y in zip(a_vals, b_vals))
        boot = _bootstrap_kappa(a_vals, b_vals, rng, n_boot) if a_vals else (float("nan"), float("nan"))
        rows.append(
            {
                "field": column,
                "n_both_coded": kappa["n"],
                "n_disagree": n_disagree,
                "n_blank_mismatch": blank_mismatch,
                "percent_agreement": kappa["agreement"],
                "cohen_kappa": kappa["kappa"],
                "kappa_lo": boot[0],
                "kappa_hi": boot[1],
                "krippendorff_alpha": alpha_missing["alpha"],
                "alpha_n": alpha_missing["n"],
                "alpha_skipped": alpha_missing["n_skipped"],
            }
        )
    return rows


def _bootstrap_kappa(left: list, right: list, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n = len(left)
    if n < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    draws = np.empty(n_boot)
    index = np.arange(n)
    for i in range(n_boot):
        take = rng.choice(index, size=n, replace=True)
        draws[i] = cohen_kappa([left[j] for j in take], [right[j] for j in take])["kappa"]
    clean = draws[np.isfinite(draws)]
    if len(clean) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def _append_model_rows(
    column: str,
    both: pd.DataFrame,
    parser,
    a_vals: list,
    b_vals: list,
    model: pd.DataFrame,
    model_rows: list,
    prevalence_rows: list,
    corpus_prevalence: dict[str, float],
    rng: np.random.Generator,
    n_boot: int,
) -> None:
    ids = both["id"].tolist()
    agreed_ids = []
    gold = []
    for doc_id, a, b in zip(ids, [parser(v) for v in both[f"{column}__a"]], [parser(v) for v in both[f"{column}__b"]]):
        if a is None or b is None or a != b:
            continue
        if doc_id not in model.index or column not in model.columns:
            continue
        predicted = parser(model.at[doc_id, column])
        if predicted is None:
            continue
        agreed_ids.append(doc_id)
        gold.append((a, predicted))
    if not gold:
        return
    n_disagree = sum(x != y for x, y in zip(a_vals, b_vals))
    if column in BARRIER_CODES:
        truth = [item[0] for item in gold]
        pred = [item[1] for item in gold]
        scores = binary_scores(truth, pred)
        model_rows.append({"field": column, "class_label": "1", "n_gold": scores["n"], "n_coder_disagree_excluded": n_disagree, **{k: scores[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}})
        apparent_gold = float(np.mean(pred)) if pred else float("nan")
        apparent = corpus_prevalence.get(column, apparent_gold)
        corrected = rogan_gladen(apparent, scores["sensitivity"], scores["specificity"])
        lo, hi = _bootstrap_prevalence(truth, pred, apparent, rng, n_boot)
        prevalence_rows.append(
            {
                "field": column,
                "n_gold": scores["n"],
                "apparent_source": "corpus" if column in corpus_prevalence else "agreed_items",
                "apparent": apparent,
                "sensitivity": scores["sensitivity"],
                "specificity": scores["specificity"],
                "pi": corrected["pi"],
                "pi_unclipped": corrected["pi_unclipped"],
                "pi_lo": lo,
                "pi_hi": hi,
                "status": corrected["status"],
            }
        )
        return
    labels = sorted({item[0] for item in gold} | {item[1] for item in gold})
    f1s = []
    for label in labels:
        truth = [1 if item[0] == label else 0 for item in gold]
        pred = [1 if item[1] == label else 0 for item in gold]
        scores = binary_scores(truth, pred)
        f1s.append(scores["f1"])
        model_rows.append(
            {
                "field": column,
                "class_label": label,
                "n_gold": scores["n"],
                "n_coder_disagree_excluded": n_disagree,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "tp": scores["tp"],
                "fp": scores["fp"],
                "fn": scores["fn"],
                "tn": scores["tn"],
            }
        )
    finite = [value for value in f1s if not (isinstance(value, float) and np.isnan(value))]
    model_rows.append(
        {
            "field": column,
            "class_label": "macro",
            "n_gold": len(gold),
            "n_coder_disagree_excluded": n_disagree,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float(np.mean(finite)) if finite else float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    )


def _bootstrap_prevalence(truth: list[int], pred: list[int], apparent: float, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n = len(truth)
    if n < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    draws = np.empty(n_boot)
    index = np.arange(n)
    for i in range(n_boot):
        take = rng.choice(index, size=n, replace=True)
        scores = binary_scores([truth[j] for j in take], [pred[j] for j in take])
        draws[i] = rogan_gladen(apparent, scores["sensitivity"], scores["specificity"])["pi"]
    clean = draws[np.isfinite(draws)]
    if len(clean) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def collapse_stage(stage: str | None, onboarding: str | None, split_ownership: bool) -> str | None:
    """One ownership stage, or Ryan's split, from the same two columns.

    ``split_ownership`` False ignores onboarding. True maps ownership plus
    onboarding=yes to ``onboarding`` and ownership plus onboarding=no to
    ``long_term_ownership``. A missing onboarding flag stays ``ownership``.
    """
    if stage is None:
        return None
    if not split_ownership or stage != "ownership":
        return stage
    if onboarding == "yes":
        return "onboarding"
    if onboarding == "no":
        return "long_term_ownership"
    return "ownership"
