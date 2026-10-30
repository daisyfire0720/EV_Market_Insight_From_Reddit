"""Comment rows are counted. The main tables stay submissions.

These tests use mock rows only. They do not read the 14B label file and
they do not invent a model label for a real post.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ev_v2.after_llm import audience_from_subreddit, run_tables


def _stamp(day: int) -> float:
    return datetime(2023, 1, day, tzinfo=timezone.utc).timestamp()


def test_subreddit_audience_matches_the_study_lists():
    assert audience_from_subreddit("BoltEV") == "enthusiast"
    assert audience_from_subreddit("CARS") == "shopper"
    assert audience_from_subreddit("notacommunity") == ""
    assert audience_from_subreddit("") == ""


def test_comments_are_logged_and_stay_out_of_the_main_tables(tmp_path):
    meta = pd.DataFrame(
        [
            {
                "id": "s1",
                "audience": "enthusiast",
                "subreddit": "electricvehicles",
                "month": "2023-01",
                "author_hash": "h1",
                "created_utc": _stamp(1),
                "stage_label": "learning about electric vehicles",
            },
            {
                "id": "s2",
                "audience": "shopper",
                "subreddit": "cars",
                "month": "2023-01",
                "author_hash": "h2",
                "created_utc": _stamp(20),
                "stage_label": "comparing electric vehicle options",
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {"id": "s1", "text_type": "submission", "stage": "awareness", "families": "charging_energy", "aspects": "charging_access", "stance": "no_stance", "ok": True},
            {"id": "s2", "text_type": "submission", "stage": "consideration", "families": "cost_incentives", "aspects": "price_tco", "stance": "no_stance", "ok": True},
            {"id": "s_missing", "text_type": "submission", "stage": "awareness", "families": "", "aspects": "", "stance": "no_stance", "ok": True},
            # Parent link wins even though the comment's own subreddit would be shopper.
            {"id": "c_parent", "text_type": "comment", "stage": "onboarding", "families": "charging_energy", "aspects": "charging_access", "stance": "no_stance", "ok": True, "subreddit": "cars", "link_id": "t3_s1", "created_utc": _stamp(2)},
            {"id": "c_sub", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "", "stance": "no_stance", "ok": True, "subreddit": "Autos", "link_id": "", "created_utc": _stamp(28)},
            {"id": "c_sub", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "", "stance": "no_stance", "ok": True, "subreddit": "Autos", "link_id": "", "created_utc": _stamp(28)},
            {"id": "c_none", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "", "stance": "no_stance", "ok": True, "subreddit": "notacommunity", "link_id": "t3_missing", "created_utc": _stamp(3)},
            {"id": "c_bad_stage", "text_type": "comment", "stage": "dealer_access", "families": "", "aspects": "", "stance": "no_stance", "ok": True, "subreddit": "cars", "link_id": "", "created_utc": _stamp(4)},
            {"id": "c_bad_json", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "", "stance": "no_stance", "ok": False, "subreddit": "cars", "link_id": "", "created_utc": _stamp(5)},
        ]
    )
    llm_path = tmp_path / "llm.csv"
    small_path = tmp_path / "small.csv"
    author_path = tmp_path / "authors.csv"
    labels.to_csv(llm_path, index=False)
    meta.to_csv(small_path, index=False)
    meta[["id", "author_hash", "created_utc"]].to_csv(author_path, index=False)
    out = tmp_path / "tables"
    tables = run_tables(llm_path, small_path, author_path, out, n_boot=0)
    main_n = int(tables["shares"].groupby("audience")["n"].first().sum())
    assert main_n == 2
    assert set(tables["shares"]["audience"]) == {"enthusiast", "shopper"}
    comment_n = int(tables["comment_shares"].groupby("audience")["n"].first().sum())
    assert comment_n == 2
    by_audience = tables["comment_shares"].groupby("audience")["stage"].apply(set)
    assert by_audience.loc["enthusiast"] == {"onboarding"}
    assert by_audience.loc["shopper"] == {"unclear"}
    log = tables["drop_log"].set_index("reason")["n"]
    assert int(log.loc["kept_submission"]) == 2
    assert int(log.loc["kept_comment"]) == 2
    assert int(log.loc["no_submission_meta"]) == 1
    assert int(log.loc["no_audience"]) == 1
    assert int(log.loc["duplicate_id"]) == 1
    assert int(log.loc["unknown_stage"]) == 1
    assert int(log.loc["not_ok"]) == 1
    assert int(log.sum()) == len(labels)
    sensitivity = tables["comments_sensitivity"].set_index("item")["value"].astype(str)
    assert sensitivity.loc["rdit_usable"] == "False"
    assert sensitivity.loc["main_specification"] == "submissions"
    assert "not usable for RDiT" in sensitivity.loc["note"]
    assert "rebalanced" in sensitivity.loc["note"]
    assert int(sensitivity.loc["n_audience_parent_link"]) == 1
    assert int(sensitivity.loc["n_audience_subreddit"]) == 1
    assert (out / "row_drop_log.csv").exists()
    assert (out / "funnel_shares_comments.csv").exists()
    assert (out / "comment_window_balance.csv").exists()
    assert not (out / "rdit_comments_rebalanced.csv").exists()
    balance = pd.read_csv(out / "comment_window_balance.csv")
    tesla = balance[(balance["cutoff_id"] == "tesla_price_cut_2023") & (balance["audience"] == "enthusiast")].iloc[0]
    assert int(tesla["n_left"]) == 1
    assert int(tesla["n_right"]) == 0


def test_rebalanced_flag_is_what_makes_comment_rdit_eligible(tmp_path):
    meta = pd.DataFrame(
        [{"id": "s1", "audience": "enthusiast", "subreddit": "electricvehicles", "author_hash": "h", "created_utc": _stamp(10), "stage_label": "learning about electric vehicles"}]
    )
    labels = pd.DataFrame(
        [
            {"id": "s1", "text_type": "submission", "stage": "awareness", "families": "", "aspects": "incentives_policy", "ok": True},
            {"id": "c1", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "incentives_policy", "ok": True, "subreddit": "electricvehicles", "link_id": "", "created_utc": _stamp(1)},
            {"id": "c2", "text_type": "comment", "stage": "unclear", "families": "", "aspects": "", "ok": True, "subreddit": "electricvehicles", "link_id": "", "created_utc": _stamp(20)},
        ]
    )
    paths = [tmp_path / name for name in ("llm.csv", "small.csv", "authors.csv")]
    labels.to_csv(paths[0], index=False)
    meta.to_csv(paths[1], index=False)
    meta[["id", "author_hash", "created_utc"]].to_csv(paths[2], index=False)
    out = tmp_path / "tables"
    tables = run_tables(paths[0], paths[1], paths[2], out, n_boot=0, comments_rebalanced=True)
    usable = tables["comments_sensitivity"].set_index("item").loc["rdit_usable", "value"]
    assert str(usable) == "True"
    assert (out / "rdit_comments_rebalanced.csv").exists()
    assert int(tables["shares"].groupby("audience")["n"].first().sum()) == 1


def test_manuscript_is_14b_first_and_the_comment_plan_is_not_a_run():
    from ev_v2.manuscript import render_manuscript

    root = Path(__file__).resolve().parents[1]
    text = render_manuscript(root / "outputs")
    assert "consistent with" in text
    assert "13rktx5" in text
    assert "13rkt8q" in text
    assert "placeholder. Ryan is coding." in text
    assert "not usable for RDiT" in text
    assert "+3.5" in text
    assert "+10.5" in text
    assert "qwen2.5:14b-instruct" in text
    assert "qwen2.5:7b-instruct" in text
    assert "score under 2" in text
    assert "unvalidated" in text
    assert "does not reproduce" in text
    assert "parent title" in text
    assert "donut" in text
    status = (root / "paper" / "STATUS.md").read_text()
    assert "coder_ryan.csv" in status
    assert "performance" in status
    assert "[CITATION NEEDED]" in status
    plan = (root / "COMMENTS_V2_PLAN.md").read_text()
    assert "0.91" in plan
    assert "6.6 hours" in plan
    assert "56.8 hours" in plan
    assert "performance" in plan
    assert "does not pull" in plan
