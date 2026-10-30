"""Framing A arithmetic. No model and no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_v2.annotate import _score_batch
from ev_v2.framing import (
    build_corpus,
    event_contrasts,
    event_study,
    fit_interrupted_series,
    proportion_difference,
    wilson_interval,
)
from ev_v2.arctic import CORE_SUBREDDITS


def test_wilson_interval_covers_a_known_share():
    share, lo, hi = wilson_interval(50, 100)
    assert abs(share - 0.5) < 1e-9
    assert lo < 0.5 < hi
    assert hi < 0.7
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]


def test_proportion_difference_of_a_twenty_point_gap():
    diff = proportion_difference(50, 100, 30, 100)
    assert abs(diff["diff"] - 0.2) < 1e-9
    assert diff["lo"] > 0
    assert diff["hi"] < 0.4


def test_interrupted_series_recovers_a_level_shift():
    rows = []
    # 12 months before 2022-08, the event month, and 6 after. Share jumps by 0.2.
    start = 2021 * 12 + 7  # 2021-08
    for step in range(19):
        index = start + step
        year, mon = divmod(index, 12)
        month = f"{year:04d}-{mon + 1:02d}"
        share = 0.3 if month >= "2022-08" else 0.1
        rows.append({"month": month, "n": 200, "k": int(round(share * 200))})
    fit = fit_interrupted_series(pd.DataFrame(rows), "2022-08")
    assert fit["status"] == "ok"
    assert abs(fit["post"] - 0.2) < 0.02
    assert abs(fit["time"]) < 0.01
    assert fit["post_lo"] > 0.1


def test_missing_month_does_not_renumber_time():
    # Drop 2022-07. The post shift should still be about 0.2 because time is calendar time.
    rows = []
    start = 2021 * 12 + 7
    for step in range(19):
        index = start + step
        year, mon = divmod(index, 12)
        month = f"{year:04d}-{mon + 1:02d}"
        if month == "2022-07":
            continue
        share = 0.3 if month >= "2022-08" else 0.1
        rows.append({"month": month, "n": 200, "k": int(round(share * 200))})
    fit = fit_interrupted_series(pd.DataFrame(rows), "2022-08")
    assert fit["status"] == "ok"
    assert abs(fit["post"] - 0.2) < 0.03


def test_event_study_uses_the_month_before_as_the_reference():
    rows = []
    for month, on in (("2022-07", False), ("2022-08", True), ("2022-09", True)):
        for i in range(40):
            rows.append(
                {
                    "id": f"{month}-{i}",
                    "audience": "shopper",
                    "month": month,
                    "barrier_purchase_incentives_or_tax_credits_on": on,
                }
            )
    frame = pd.DataFrame(rows)
    study = event_study(frame, ["purchase incentives or tax credits"], pre=1, post=1)
    august = study[(study["event_id"] == "ira_2022") & (study["audience"] == "shopper") & (study["rel_month"] == 0)]
    assert len(august) == 1
    assert abs(float(august.iloc[0]["diff"]) - 1.0) < 1e-9
    assert float(august.iloc[0]["lo"]) > 0.5


def test_corpus_keeps_enthusiast_and_shopper_without_duplicating_ids():
    core = pd.DataFrame(
        [
            {"id": "e1", "subreddit": CORE_SUBREDDITS[0], "month": "2020-01", "title": "t", "selftext": "body", "author": "a"},
            {"id": "shared", "subreddit": "cars", "month": "2020-02", "title": "old", "selftext": "x", "author": "a"},
        ]
    )
    dump = pd.DataFrame(
        [
            {"id": "shared", "subreddit": "cars", "month": "2020-02", "title": "from dump", "selftext": "y", "author": "b"},
            {"id": "s2", "subreddit": "Autos", "month": "2021-01", "title": "shop", "selftext": "z", "author": "c"},
        ]
    )
    corpus = build_corpus(core, dump)
    assert set(corpus["id"]) == {"e1", "shared", "s2"}
    assert corpus.loc[corpus["id"] == "e1", "audience"].iloc[0] == "enthusiast"
    assert corpus.loc[corpus["id"] == "s2", "audience"].iloc[0] == "shopper"
    # The dump copy wins only when the id was not already in the enthusiast file.
    # "shared" is a shopper id in the old API sample, so the enthusiast filter drops it
    # and the dump row is the one that remains.
    assert corpus.loc[corpus["id"] == "shared", "title"].iloc[0] == "from dump"


class _FakeClassifier:
    def __call__(self, snippets, candidate_labels, hypothesis_template, multi_label):
        results = []
        for snippet in snippets:
            n = len(candidate_labels)
            scores = [0.1] * n
            scores[0] = 0.9
            results.append({"labels": list(candidate_labels), "scores": scores})
        return results[0] if len(results) == 1 else results


def test_batch_scoring_keeps_a_markup_as_dealer_friction():
    rows = _score_batch(
        _FakeClassifier(),
        [("p1", "The dealer markup was two thousand over sticker.")],
        "fake-model",
        0.5,
    )
    assert rows[0]["barrier_dealer_or_buying_friction_on"] is True
    assert rows[0]["dealer_rule"] == "lexicon"
    assert rows[0]["id"] == "p1"


def test_event_contrast_prior_three_months():
    rows = []
    # May–Jul low, August high. IRA window is August 2022.
    for month, on, n in (("2022-05", False, 30), ("2022-06", False, 30), ("2022-07", False, 30), ("2022-08", True, 30)):
        for i in range(n):
            rows.append(
                {
                    "id": f"{month}-{i}",
                    "audience": "enthusiast",
                    "month": month,
                    "barrier_purchase_incentives_or_tax_credits_on": on,
                    "barrier_vehicle_price_or_total_cost_on": False,
                    "barrier_charging_access_on": False,
                    "barrier_driving_range_on": False,
                    "barrier_battery_reliability_on": False,
                    "barrier_dealer_or_buying_friction_on": False,
                    "barrier_environmental_impact_on": False,
                }
            )
    # The other audience needs rows or the function still returns a row with n=0.
    frame = pd.DataFrame(rows)
    contrasts = event_contrasts(frame, ["purchase incentives or tax credits"])
    hit = contrasts[
        (contrasts["event_id"] == "ira_2022")
        & (contrasts["audience"] == "enthusiast")
        & (contrasts["baseline"] == "prior_3_months")
    ]
    assert abs(float(hit.iloc[0]["diff"]) - 1.0) < 1e-9
    assert int(hit.iloc[0]["n_baseline"]) == 90
