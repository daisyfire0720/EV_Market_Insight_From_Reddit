"""Funnel sheet, local labeler, discontinuity scaffold, and author transitions."""

from datetime import date, timedelta

import pandas as pd

from ev_v2.agreement import score_coders
from ev_v2.annotation_sample import (
    CALIBRATION_N,
    SHEET_COLUMNS,
    blank_sheet,
    calibration_ids,
    draw_funnel_sample,
    prepare_label_frame,
)
from ev_v2.rdit import _jump, choose_bandwidth, rdit_table
from ev_v2.taxonomy import provisional_ownership_split, sampling_families, sampling_stage
from ev_v2.transitions import author_coverage, consecutive_pairs, forward_rate_around
from llm_label import MockBackend, label_posts, parse_label


def test_ownership_lexicon_is_one_sided_and_is_not_a_keyword_stage():
    assert provisional_ownership_split("I just bought a Bolt and need to set up the app") == "onboarding"
    assert provisional_ownership_split("After two winters the battery health is worse") == "long_term_ownership"
    assert provisional_ownership_split("I just bought it and the warranty and battery health are the question") is None
    assert sampling_stage("owning an electric vehicle", "hello") == "ownership_unsplit"
    assert sampling_stage("learning about electric vehicles") == "awareness"
    flags = {"charging_access": True, "price_tco": False, "incentives_policy": False, "range": False, "battery_reliability": False, "dealer_friction": False, "environment": False}
    assert "charging_energy" in sampling_families(flags, "owning an electric vehicle", "")


def _funnel_frame() -> pd.DataFrame:
    rows = []
    stages = [
        "learning about electric vehicles",
        "comparing electric vehicle options",
        "evaluating cost or fit before buying",
        "buying an electric vehicle",
        "owning an electric vehicle",
    ]
    for audience in ("enthusiast", "shopper"):
        for index in range(80):
            stage = stages[index % len(stages)]
            text = "I just bought a Bolt and need to set up the app" if index % 5 == 4 else "After two winters the battery health is worse"
            if index % 5 != 4:
                text = text if stage == "owning an electric vehicle" else "which EV versus the other"
            rows.append(
                {
                    "id": f"{audience[0]}{index}",
                    "audience": audience,
                    "subreddit": "electricvehicles" if audience == "enthusiast" else "cars",
                    "month": "2023-01" if index % 2 == 0 else "2022-08",
                    "stage_label": stage,
                    "stance_label": "support for electric vehicles",
                    "sampling_stage": sampling_stage(stage, text),
                    "barrier_charging_access_on": index % 4 == 0,
                    "barrier_driving_range_on": False,
                    "barrier_vehicle_price_or_total_cost_on": False,
                    "barrier_purchase_incentives_or_tax_credits_on": index % 7 == 0,
                    "barrier_battery_reliability_on": False,
                    "barrier_dealer_or_buying_friction_on": index % 6 == 0,
                    "barrier_environmental_impact_on": False,
                    "purchase_selection": stage.startswith("comparing"),
                    "charging_energy": index % 4 == 0,
                    "battery_range": False,
                    "cost_incentives": index % 7 == 0,
                    "dealer_access": index % 6 == 0,
                    "reliability_service": "warranty" in text,
                    "policy_market": False,
                    "transition_narratives": False,
                }
            )
    return prepare_label_frame(pd.DataFrame(rows))


def test_funnel_draw_covers_stages_and_the_sheet_stays_blank():
    frame = _funnel_frame()
    draw = draw_funnel_sample(frame, target=80, stage_floor=4, family_floor=3, positive_floor=2, event_quota=2, seed=3)
    assert len(draw) == 80
    assert draw["id"].is_unique
    merged = draw.merge(frame, on="id")
    for audience in ("enthusiast", "shopper"):
        stages = set(merged.loc[merged["audience"] == audience, "sampling_stage"])
        assert "awareness" in stages
        assert "onboarding" in stages or "long_term_ownership" in stages
    flagged = calibration_ids(draw, frame, n=10, seed=3)
    assert len(flagged) == 10
    sheet = blank_sheet(frame, draw, {"e0": {"title": "Hello", "text_excerpt": "Hello"}}, flagged)
    assert list(sheet.columns) == SHEET_COLUMNS
    assert set(sheet["stage"]).issubset({""})
    assert set(sheet["purchase_selection"]).issubset({""})
    assert int((sheet["calibration"] == "yes").sum()) == 10
    assert "sampling_stage" not in sheet.columns


def test_calibration_rows_are_left_out_of_the_main_kappa():
    coder_a = pd.DataFrame(
        {
            "id": ["1", "2"],
            "stage": ["awareness", "onboarding"],
            "calibration": ["", "yes"],
            "stance": ["support", "oppose"],
        }
    )
    coder_b = pd.DataFrame(
        {
            "id": ["1", "2"],
            "stage": ["awareness", "purchase"],
            "calibration": ["", "yes"],
            "stance": ["support", "support"],
        }
    )
    tables = score_coders(coder_a, coder_b, n_boot=30)
    stage = tables["agreement"].set_index("field").loc["stage"]
    assert stage["n_both_coded"] == 1
    assert tables["n_joined"].iloc[0]["n_calibration_excluded"] == 1
    assert len(tables["calibration_agreement"]) >= 1


def test_local_linear_jump_recovers_a_step_and_the_mock_labeler_resumes(tmp_path):
    cutoff = date(2022, 8, 16)
    rows = []
    for offset in range(-40, 41):
        if offset == 0:
            continue
        day = cutoff + timedelta(days=offset)
        on = offset > 0
        for copy in range(4):
            rows.append(
                {
                    "id": f"{offset}_{copy}",
                    "audience": "enthusiast",
                    "created_day": day,
                    "barrier_purchase_incentives_or_tax_credits_on": on,
                }
            )
    frame = pd.DataFrame(rows)
    daily_day = frame["created_day"].map(lambda item: (item - cutoff).days).to_numpy()
    y = frame["barrier_purchase_incentives_or_tax_credits_on"].astype(int).to_numpy()
    # Collapse is inside rdit_table. The direct jump uses daily rows.
    table = rdit_table(frame, audience="enthusiast")
    main = table[(table["cutoff_id"] == "ira_signing_2022") & (table["fit"] == "main")].iloc[0]
    assert main["tau"] > 0.5
    assert main["bandwidth_days"] in {30, 60, 90, 180}
    assert not bool(main["thin"])
    jump = _jump(daily_day, y, y * 0 + 1, bandwidth=30, donut=0)
    assert jump["tau"] > 0.5
    posts = pd.DataFrame([{"id": "p1", "title": "Apartment charging", "selftext": "Can I make this work?"}])
    cache = tmp_path / "cache.jsonl"
    first = label_posts(posts, MockBackend(), cache, batch_size=2)
    second = label_posts(posts, MockBackend(), cache, batch_size=2)
    assert first.iloc[0]["model_name"] == "mock"
    assert first.iloc[0]["model_version"] == "mock-0"
    assert first.iloc[0]["stage"] == "evaluation"
    assert second.iloc[0]["stage"] == "evaluation"
    parsed = parse_label("note {\"stage\": \"purchase\", \"families\": [\"dealer_access\"], \"aspects\": [], \"stance\": \"mixed\", \"evidence\": \"one two\"}")
    assert parsed["ok"] and parsed["stage"] == "purchase"
    # choose_bandwidth is deterministic on this step.
    from ev_v2.rdit import _daily

    assert choose_bandwidth(_daily(frame, "barrier_purchase_incentives_or_tax_credits_on", cutoff)) in {30, 60, 90, 180}


def test_repeat_authors_drive_the_transition_count():
    frame = pd.DataFrame(
        [
            {"id": "a1", "author_hash": "h1", "created_utc": 1_000, "stage_label": "learning about electric vehicles", "audience": "enthusiast"},
            {"id": "a2", "author_hash": "h1", "created_utc": 1_000 + 10 * 86400, "stage_label": "buying an electric vehicle", "audience": "enthusiast"},
            {"id": "b1", "author_hash": "", "created_utc": 2_000, "stage_label": "owning an electric vehicle", "audience": "shopper"},
            {"id": "c1", "author_hash": "h2", "created_utc": 3_000, "stage_label": "comparing electric vehicle options", "audience": "shopper"},
        ]
    )
    coverage = author_coverage(frame)
    all_rows = coverage.set_index("audience").loc["all"]
    assert all_rows["n_authors"] == 2
    assert all_rows["n_repeat_authors"] == 1
    pairs = consecutive_pairs(frame)
    assert len(pairs) == 1
    assert bool(pairs.iloc[0]["forward"]) is True
    rate = forward_rate_around(pairs, cutoff_utc=5_000, window_days=30)
    assert rate["n_pairs"] == 1
    assert rate["n_forward"] == 1
