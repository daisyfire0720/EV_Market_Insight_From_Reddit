"""Round-5 checks. Synthetic labels only. No network and no model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard

from ev_v2.agreement import (
    cohen_kappa,
    collapse_stage,
    krippendorff_alpha_nominal,
    rogan_gladen,
    score_coders,
)
from ev_v2.annotation_sample import (
    SHEET_COLUMNS,
    blank_sheet,
    draw_annotation_sample,
    model_key,
    prepare_label_frame,
)
from ev_v2.blocks import SAMPLE_EVENT_WINDOWS, _quarter_window, assign_week_block
from ev_v2.bootstrap import block_bootstrap_contrasts, window_flags
from ev_v2.dumps import extract_paths
from ev_v2.sensitivity import pre_purchase_mask, slice_contrasts, slice_shares
from score_agreement import main as score_main


def test_cohen_kappa_matches_a_hand_count():
    # 3 agreements out of 4. Margins give expected agreement 0.5, so kappa is 0.5.
    left = ["1", "1", "1", "2"]
    right = ["1", "1", "2", "2"]
    stats = cohen_kappa(left, right)
    assert stats["agreement"] == 0.75
    assert abs(stats["kappa"] - 0.5) < 1e-9


def test_krippendorff_alpha_matches_the_coincidence_matrix():
    pairs = [("1", "1"), ("1", "1"), ("1", "2"), ("2", "2")]
    alpha = krippendorff_alpha_nominal(pairs)["alpha"]
    # Do = 0.25, De = 30/56, alpha = 1 - 0.25 / (30/56).
    assert abs(alpha - (1 - 0.25 / (30 / 56))) < 1e-9
    perfect = krippendorff_alpha_nominal([("1", "1"), ("2", "2")])["alpha"]
    assert perfect == 1.0
    skipped = krippendorff_alpha_nominal([("1", None), ("1", "1"), ("2", "2")])
    assert skipped["n_skipped"] == 1
    assert skipped["alpha"] == 1.0


def test_rogan_gladen_and_the_unidentified_case():
    fixed = rogan_gladen(0.5, 0.9, 0.8)
    assert fixed["status"] == "ok"
    assert abs(fixed["pi_unclipped"] - (0.3 / 0.7)) < 1e-9
    broken = rogan_gladen(0.4, 0.4, 0.6)
    assert broken["status"] == "not_identified"
    assert broken["pi"] != broken["pi"]


def test_score_coders_drops_disagreements_and_scores_the_model():
    coder_a = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "incentives_policy": [1, 1, 0, 1],
            "stage": ["ownership", "purchase", "unclear", "ownership"],
            "onboarding": ["yes", "na", "na", "no"],
            "stance": ["support", "support", "no_stance", "oppose"],
        }
    )
    coder_b = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "incentives_policy": [1, 0, 0, 1],
            "stage": ["ownership", "purchase", "unclear", "awareness"],
            "onboarding": ["yes", "na", "na", "no"],
            "stance": ["support", "support", "no_stance", "oppose"],
        }
    )
    key = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "incentives_policy": [1, 1, 0, 1],
            "stage": ["ownership", "purchase", "unclear", "ownership"],
            "stance": ["support", "support", "no_stance", "oppose"],
        }
    )
    tables = score_coders(coder_a, coder_b, model_key=key, corpus_prevalence={"incentives_policy": 0.2}, n_boot=200)
    incentives = tables["agreement"].set_index("field").loc["incentives_policy"]
    assert incentives["n_both_coded"] == 4
    assert incentives["n_disagree"] == 1
    model = tables["model_vs_human"]
    row = model[(model["field"] == "incentives_policy")].iloc[0]
    # Gold is the three agreed items: positives on 1 and 4, negative on 3. Model matches them.
    assert row["n_gold"] == 3
    assert row["f1"] == 1
    prev = tables["prevalence"].set_index("field").loc["incentives_policy"]
    assert prev["apparent_source"] == "corpus"
    assert prev["status"] == "ok"
    assert collapse_stage("ownership", "yes", split_ownership=False) == "ownership"
    assert collapse_stage("ownership", "yes", split_ownership=True) == "onboarding"
    assert collapse_stage("ownership", "no", split_ownership=True) == "long_term_ownership"


def test_score_script_on_synthetic_files(tmp_path: Path):
    coder_a = tmp_path / "a.csv"
    coder_b = tmp_path / "b.csv"
    pd.DataFrame({"id": ["1"], "incentives_policy": [1], "stage": ["ownership"], "onboarding": ["yes"], "stance": ["support"]}).to_csv(coder_a, index=False)
    pd.DataFrame({"id": ["1"], "incentives_policy": [1], "stage": ["ownership"], "onboarding": ["yes"], "stance": ["support"]}).to_csv(coder_b, index=False)
    out = tmp_path / "out"
    assert score_main(["--coder-a", str(coder_a), "--coder-b", str(coder_b), "--out", str(out), "--n-boot", "50"]) == 0
    assert (out / "agreement.csv").exists()
    missing = score_main(["--coder-a", str(tmp_path / "nope.csv"), "--coder-b", str(coder_b), "--out", str(out)])
    assert missing == 2


def _labels() -> pd.DataFrame:
    rows = []
    months = ["2016-03", "2019-05", "2022-08", "2023-01", "2023-05", "2024-04", "2025-09"]
    i = 0
    for audience in ("enthusiast", "shopper"):
        for month in months:
            for k in range(8):
                i += 1
                sub = "electricvehicles" if audience == "enthusiast" else ("carbuying" if k == 0 else "cars")
                stage = "buying an electric vehicle" if k < 2 else "owning an electric vehicle"
                rows.append(
                    {
                        "id": f"{audience[0]}{i}",
                        "audience": audience,
                        "subreddit": sub,
                        "month": month,
                        "stage_label": stage,
                        "stance_label": "support for electric vehicles",
                        "barrier_charging_access_on": k == 4,
                        "barrier_driving_range_on": False,
                        "barrier_vehicle_price_or_total_cost_on": k == 5,
                        "barrier_purchase_incentives_or_tax_credits_on": k < 2,
                        "barrier_battery_reliability_on": False,
                        "barrier_dealer_or_buying_friction_on": k == 3,
                        "barrier_environmental_impact_on": False,
                    }
                )
    return prepare_label_frame(pd.DataFrame(rows))


def test_draw_hits_the_target_and_the_sheet_hides_the_model():
    frame = _labels()
    draw = draw_annotation_sample(frame, target=40, positive_floor=4, event_quota=2, cell_quota=2, seed=1)
    assert len(draw) == 40
    assert draw["id"].is_unique
    merged = draw.merge(frame, on="id")
    for audience in ("enthusiast", "shopper"):
        block = merged[merged["audience"] == audience]
        assert int(block["incentives_policy"].sum()) >= 4
        assert int(block["dealer_friction"].sum()) >= 4
    sheet = blank_sheet(frame, draw, {"e1": {"title": "Hello", "text_excerpt": "Hello there"}})
    assert list(sheet.columns) == SHEET_COLUMNS
    assert "stage_label" not in sheet.columns
    assert set(sheet["stage"]).issubset({""})
    assert set(sheet["incentives_policy"]).issubset({""})
    key = model_key(frame, draw)
    assert set(key["stage"]).issubset({"awareness", "consideration", "evaluation", "purchase", "ownership", "unclear"})
    assert key["incentives_policy"].isin([0, 1]).all()


def test_week_block_prefers_the_event_window_and_matches_the_downloader():
    stamp = int(datetime(2022, 8, 16, 15, tzinfo=timezone.utc).timestamp())
    block = assign_week_block("electricvehicles", stamp)
    assert "ira_2022" in block
    assert block.startswith("electricvehicles|ira_2022|2022-08-14")
    from download_reddit import _quarter_window as downloader_window

    for subreddit, year, quarter in (("Rivian", 2021, 3), ("BoltEV", 2019, 1), ("evcharging", 2024, 4)):
        assert _quarter_window(subreddit, year, quarter, SAMPLE_EVENT_WINDOWS[0][1]) == downloader_window(
            subreddit, year, quarter, SAMPLE_EVENT_WINDOWS[0][1]
        )
    # The comparison date above is not the point. Compare on the sample day.
    from datetime import date

    as_of = date(2026, 10, 16)
    assert _quarter_window("teslamotors", 2022, 3, as_of) == downloader_window("teslamotors", 2022, 3, as_of)


def test_block_bootstrap_widens_when_posts_share_a_week_and_flags_a_thin_window():
    rows = []
    # Four enthusiast weeks in the event month, all incentives on. Four in the prior months, all off.
    for week, month, on in (
        ("w1", "2022-08", True),
        ("w2", "2022-08", True),
        ("w3", "2022-08", True),
        ("w4", "2022-08", True),
        ("b1", "2022-07", False),
        ("b2", "2022-06", False),
        ("b3", "2022-05", False),
        ("b4", "2022-05", False),
    ):
        for i in range(20):
            rows.append(
                {
                    "id": f"{week}-{i}",
                    "audience": "enthusiast",
                    "subreddit": "electricvehicles",
                    "month": month,
                    "week_block": f"electricvehicles|{week}",
                    "subreddit_month": f"electricvehicles|{month}",
                    "barrier_purchase_incentives_or_tax_credits_on": on,
                }
            )
    # Shopper: one event month, three baseline months, so the calendar-month block is flagged.
    for month, on, n in (("2022-08", True, 30), ("2022-07", False, 10), ("2022-06", False, 10), ("2022-05", False, 10)):
        for i in range(n):
            rows.append(
                {
                    "id": f"s-{month}-{i}",
                    "audience": "shopper",
                    "subreddit": "cars" if i % 2 == 0 else "Autos",
                    "month": month,
                    "week_block": "",
                    "subreddit_month": f"{'cars' if i % 2 == 0 else 'Autos'}|{month}",
                    "barrier_purchase_incentives_or_tax_credits_on": on,
                }
            )
    frame = pd.DataFrame(rows)
    boot = block_bootstrap_contrasts(frame, ["purchase incentives or tax credits"], n_boot=300, seed=2)
    enth = boot[(boot["audience"] == "enthusiast") & (boot["baseline"] == "prior_3_months") & (boot["reported"] == True)].iloc[0]  # noqa: E712
    assert abs(enth["diff"] - 1) < 1e-9
    assert enth["boot_lo"] == 1
    assert enth["block"] == "week_block"
    assert enth["thin_window"] == False  # noqa: E712
    shop = boot[(boot["audience"] == "shopper") & (boot["baseline"] == "prior_3_months") & (boot["block"] == "month_block")].iloc[0]
    assert bool(shop["few_blocks"]) is True
    reported = boot[(boot["audience"] == "shopper") & (boot["baseline"] == "prior_3_months") & (boot["reported"] == True)].iloc[0]  # noqa: E712
    assert reported["block"] == "subreddit_month"
    flags = window_flags(frame[frame["audience"] == "enthusiast"])
    ira = flags[(flags["event_id"] == "ira_2022") & (flags["window"] == "event") & (flags["audience"] == "enthusiast")].iloc[0]
    assert ira["observed_months"] == 1
    assert ira["thin_window"] == False  # noqa: E712
    # Drop June so the prior-three window is thin.
    thin = frame[~((frame["audience"] == "enthusiast") & (frame["month"] == "2022-06"))]
    flags = window_flags(thin)
    prior = flags[(flags["event_id"] == "ira_2022") & (flags["window"] == "prior_3_months") & (flags["audience"] == "enthusiast")].iloc[0]
    assert prior["observed_months"] == 2
    assert prior["missing_months"] == "2022-06"
    assert bool(prior["thin_window"]) is True


def test_shopper_slices_separate_carbuying_and_pre_purchase():
    rows = []
    for i, (sub, stage) in enumerate(
        (
            ("carbuying", "buying an electric vehicle"),
            ("carbuying", "owning an electric vehicle"),
            ("cars", "owning an electric vehicle"),
            ("cars", "learning about electric vehicles"),
        )
    ):
        rows.append(
            {
                "id": str(i),
                "audience": "shopper",
                "subreddit": sub,
                "month": "2023-01",
                "stage_label": stage,
                "stance_label": "support for electric vehicles",
                "barrier_charging_access_on": False,
                "barrier_driving_range_on": False,
                "barrier_vehicle_price_or_total_cost_on": sub == "carbuying",
                "barrier_purchase_incentives_or_tax_credits_on": False,
                "barrier_battery_reliability_on": False,
                "barrier_dealer_or_buying_friction_on": False,
                "barrier_environmental_impact_on": False,
            }
        )
    rows.append(
        {
            "id": "e",
            "audience": "enthusiast",
            "subreddit": "electricvehicles",
            "month": "2023-01",
            "stage_label": "owning an electric vehicle",
            "stance_label": "opposition to electric vehicles",
            "barrier_charging_access_on": False,
            "barrier_driving_range_on": False,
            "barrier_vehicle_price_or_total_cost_on": False,
            "barrier_purchase_incentives_or_tax_credits_on": False,
            "barrier_battery_reliability_on": False,
            "barrier_dealer_or_buying_friction_on": False,
            "barrier_environmental_impact_on": False,
        }
    )
    frame = pd.DataFrame(rows)
    assert int(pre_purchase_mask(frame).sum()) == 2
    shares = slice_shares(frame, ["vehicle price or total cost"])
    car = shares[(shares["slice"] == "carbuying") & (shares["measure"] == "owning an electric vehicle")].iloc[0]
    assert car["n"] == 2
    assert car["k"] == 1
    pre = shares[(shares["slice"] == "pre_purchase") & (shares["measure"] == "vehicle price or total cost")].iloc[0]
    assert pre["n"] == 2
    assert pre["k"] == 1
    contrasts = slice_contrasts(frame, ["vehicle price or total cost"])
    assert set(contrasts["slice"]) >= {"carbuying", "pre_purchase", "all_shopper"}
    assert (contrasts["audience"] == "enthusiast").any()


def test_fast_subreddit_scan_keeps_only_shopper_lines(tmp_path: Path):
    def post(doc_id, subreddit, title):
        return {
            "id": doc_id,
            "subreddit": subreddit,
            "author": "someone",
            "title": title,
            "selftext": "",
            "created_utc": int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()),
            "score": 1,
            "num_comments": 0,
            "url": "",
            "link_flair_text": "",
        }

    payload = "".join(json.dumps(obj) + "\n" for obj in (post("a", "cars", "A Tesla lease"), post("b", "news", "A Tesla rocket"), post("c", "Autos", "electric vehicle")))
    path = tmp_path / "RS_2024-06.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode()))
    out = tmp_path / "out.csv"
    stats = extract_paths([path], out, fast_subreddit_scan=True)
    kept = pd.read_csv(out)
    assert set(kept["id"]) == {"a", "c"}
    assert stats.rows_kept == 2
    assert stats.skipped_subreddit >= 1


def test_event_window_dates_match_the_downloader_source():
    source = Path(__file__).resolve().parents[1].joinpath("download_reddit.py").read_text()
    for event_id, start, end in SAMPLE_EVENT_WINDOWS:
        assert event_id in source
        assert f"date({start.year}, {start.month}, {start.day})" in source
        assert f"date({end.year}, {end.month}, {end.day})" in source
