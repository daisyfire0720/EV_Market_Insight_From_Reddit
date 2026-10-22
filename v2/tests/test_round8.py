"""The downstream command on mock instruction-model output, not a real GPU run."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from ev_v2.after_llm import run_tables
from llm_label import MockBackend, label_posts
from run_after_llm import main as after_main


def _posts() -> pd.DataFrame:
    start = datetime(2022, 7, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(24):
        stamp = start + timedelta(days=index * 3)
        audience = "enthusiast" if index % 2 == 0 else "shopper"
        rows.append(
            {
                "id": f"p{index}",
                "audience": audience,
                "subreddit": "electricvehicles" if audience == "enthusiast" else "cars",
                "month": stamp.strftime("%Y-%m"),
                "title": "Apartment charging",
                "selftext": "Can I make an EV work if I street park?",
                "stage_label": "evaluating cost or fit before buying" if index % 3 else "learning about electric vehicles",
                "author_hash": "h1" if index < 4 else f"h{index}",
                "created_utc": stamp.timestamp(),
                "barrier_charging_access_on": True,
                "barrier_purchase_incentives_or_tax_credits_on": index > 12,
                "barrier_vehicle_price_or_total_cost_on": False,
                "barrier_driving_range_on": False,
                "barrier_battery_reliability_on": False,
                "barrier_dealer_or_buying_friction_on": False,
                "barrier_environmental_impact_on": False,
            }
        )
    return pd.DataFrame(rows)


def test_mock_labels_rebuild_the_tables_and_do_not_invent_coders(tmp_path):
    posts = _posts()
    labeled = label_posts(posts, MockBackend(), tmp_path / "cache.jsonl")
    llm_path = tmp_path / "llm.csv"
    small_path = tmp_path / "small.csv"
    author_path = tmp_path / "authors.csv"
    labeled.to_csv(llm_path, index=False)
    posts.to_csv(small_path, index=False)
    posts[["id", "author_hash", "created_utc"]].to_csv(author_path, index=False)
    out = tmp_path / "tables"
    tables = run_tables(llm_path, small_path, author_path, out, n_boot=0)
    assert int(tables["validity"].iloc[0]["n_used"]) == 24
    assert set(tables["shares"]["stage"]) == {"evaluation"}
    assert (out / "stage_family_heatmap.csv").exists()
    assert (out / "stage_family_heatmap.png").exists()
    assert (out / "rdit.csv").exists()
    assert (out / "stage_transition_shocks.csv").exists()
    assert "family" in set(tables["agreement"]["field"])
    status = pd.read_csv(out / "annotator_status.csv")
    assert status.iloc[0]["coder_sheets"] == "not_provided"
    heat = pd.read_csv(out / "stage_family_heatmap.csv")
    cell = heat[(heat["stage"] == "evaluation") & (heat["family"] == "charging_energy") & (heat["audience"] == "enthusiast")]
    assert float(cell.iloc[0]["share"]) == 1.0
    missing = after_main(["--llm", str(tmp_path / "missing.csv"), "--out", str(out)])
    assert missing == 2


def test_coder_sheets_are_scored_only_when_both_exist(tmp_path):
    posts = _posts().head(6)
    labeled = label_posts(posts, MockBackend(), tmp_path / "cache.jsonl")
    llm_path = tmp_path / "llm.csv"
    labeled.to_csv(llm_path, index=False)
    posts.to_csv(tmp_path / "small.csv", index=False)
    posts[["id", "author_hash", "created_utc"]].to_csv(tmp_path / "authors.csv", index=False)
    coder_a = tmp_path / "a.csv"
    coder_b = tmp_path / "b.csv"
    sheet = pd.DataFrame(
        {
            "id": posts["id"],
            "stage": ["evaluation"] * len(posts),
            "charging_access": [1] * len(posts),
            "stance": ["mixed"] * len(posts),
        }
    )
    sheet.to_csv(coder_a, index=False)
    sheet.to_csv(coder_b, index=False)
    out = tmp_path / "with_coders"
    code = after_main(
        [
            "--llm",
            str(llm_path),
            "--small",
            str(tmp_path / "small.csv"),
            "--authors",
            str(tmp_path / "authors.csv"),
            "--coder-a",
            str(coder_a),
            "--coder-b",
            str(coder_b),
            "--out",
            str(out),
            "--n-boot",
            "0",
        ]
    )
    assert code == 0
    agreement = pd.read_csv(out / "human_agreement.csv")
    stage = agreement.set_index("field").loc["stage"]
    assert int(stage["n_both_coded"]) == len(posts)
    assert pd.read_csv(out / "annotator_status.csv").iloc[0]["coder_sheets"] == "scored"
    rdit = pd.read_csv(out / "rdit.csv")
    assert set(rdit["fit"]).issuperset({"main", "donut", "bandwidth_30", "bandwidth_180"})
