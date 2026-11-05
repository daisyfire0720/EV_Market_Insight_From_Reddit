"""Ryan, the 14B default, comment coverage, and the score sensitivity.

These tests do not call a model and do not invent labels.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ev_v2.after_llm import run_tables
from ev_v2.comment_coverage import authors_on_both_sides, coverage_table
from fetch_comments import _clusters, _in_clusters
from fetch_label_text import _hash_author, slim_comment
from llm_label import (
    DEFAULT_HF_MODEL,
    DEFAULT_OLLAMA_MODEL,
    FALLBACK_HF_MODEL,
    FALLBACK_OLLAMA_MODEL,
    MockBackend,
    codebook_prompt,
    label_posts,
    resolve_model,
)


def _ira() -> float:
    return datetime(2022, 8, 16, tzinfo=timezone.utc).timestamp()


def test_resolve_model_defaults_to_14b_and_falls_back_to_7b():
    assert resolve_model("ollama", "", False) == DEFAULT_OLLAMA_MODEL == "qwen2.5:14b-instruct"
    assert resolve_model("vllm", "", False) == "qwen2.5:14b-instruct"
    assert resolve_model("transformers", "", False) == DEFAULT_HF_MODEL == "Qwen/Qwen2.5-14B-Instruct"
    assert resolve_model("ollama", "custom-tag", False) == "custom-tag"
    assert resolve_model("ollama", "custom-tag", True) == FALLBACK_OLLAMA_MODEL == "qwen2.5:7b-instruct"
    assert resolve_model("transformers", "", True) == FALLBACK_HF_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert resolve_model("mock", "", False) == "mock"


def test_slim_comment_hashes_the_author_and_marks_the_text_type():
    row = slim_comment(
        {
            "id": "c1",
            "author": "someuser",
            "body": "The lease was ready on Saturday.",
            "created_utc": 1,
            "score": 3,
            "subreddit": "cars",
        }
    )
    assert row["text_type"] == "comment"
    assert row["id"] == "c1"
    assert "author" not in row
    assert row["author_hash"] == _hash_author("someuser")
    assert "someuser" not in "".join(str(value) for value in row.values())


def test_comment_row_is_labeled_as_a_comment(tmp_path):
    posts = pd.DataFrame(
        [
            {"id": "c1", "title": "", "body": "I picked the car up yesterday.", "text_type": "comment"},
            {"id": "s1", "title": "First week", "selftext": "The app is confusing.", "text_type": "submission"},
        ]
    )
    labeled = label_posts(posts, MockBackend(), tmp_path / "cache.jsonl")
    kinds = labeled.set_index("id")["text_type"]
    assert kinds.loc["c1"] == "comment"
    assert kinds.loc["s1"] == "submission"
    assert "Reddit comment" in codebook_prompt("", "[comment]\nI picked the car up yesterday.")
    assert "Reddit submission" in codebook_prompt("First week", "The app is confusing.")


def test_comment_pull_uses_the_cutoff_windows_and_skips_the_gap():
    clusters = _clusters()
    assert len(clusters) == 2
    gap = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    inside = datetime(2023, 1, 13, tzinfo=timezone.utc).timestamp()
    assert _in_clusters(inside, clusters)
    assert not _in_clusters(gap, clusters)


def test_comments_add_authors_on_both_sides_of_a_cutoff():
    cutoff = _ira()
    day = 86400.0
    events = pd.DataFrame(
        [
            {"author_hash": "a", "created_utc": cutoff - 10 * day, "text_type": "submission"},
            {"author_hash": "a", "created_utc": cutoff + 10 * day, "text_type": "submission"},
            {"author_hash": "b", "created_utc": cutoff - 10 * day, "text_type": "submission"},
            {"author_hash": "b", "created_utc": cutoff + 10 * day, "text_type": "comment"},
            {"author_hash": "c", "created_utc": cutoff - 10 * day, "text_type": "comment"},
            {"author_hash": "c", "created_utc": cutoff + 10 * day, "text_type": "comment"},
            {"author_hash": "d", "created_utc": cutoff - 200 * day, "text_type": "comment"},
            {"author_hash": "d", "created_utc": cutoff + 10 * day, "text_type": "comment"},
            {"author_hash": "", "created_utc": cutoff - 10 * day, "text_type": "comment"},
            {"author_hash": "", "created_utc": cutoff + 10 * day, "text_type": "comment"},
        ]
    )
    submissions = authors_on_both_sides(events, cutoff, text_types=("submission",))
    both = authors_on_both_sides(events, cutoff, text_types=None)
    assert submissions == {"a"}
    assert both == {"a", "b", "c"}
    table = coverage_table(events).set_index("cutoff_id")
    ira = table.loc["ira_signing_2022"]
    assert int(ira["n_authors_submissions_only"]) == 1
    assert int(ira["n_authors_submissions_and_comments"]) == 3
    assert int(ira["n_gained"]) == 2
    other = table.loc["tesla_price_cut_2023"]
    assert int(other["n_gained"]) == 0


def test_score_under_2_is_a_sensitivity_and_the_main_table_keeps_it(tmp_path):
    start = datetime(2022, 7, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(8):
        stamp = start + pd.Timedelta(days=index * 3)
        rows.append(
            {
                "id": f"p{index}",
                "audience": "enthusiast" if index % 2 == 0 else "shopper",
                "subreddit": "electricvehicles",
                "month": stamp.strftime("%Y-%m"),
                "title": "Apartment charging",
                "selftext": "Can I make an EV work if I street park?",
                "stage_label": "evaluating cost or fit before buying",
                "author_hash": "h1" if index < 4 else f"h{index}",
                "created_utc": stamp.timestamp(),
                "barrier_charging_access_on": True,
                "barrier_purchase_incentives_or_tax_credits_on": False,
                "barrier_vehicle_price_or_total_cost_on": False,
                "barrier_driving_range_on": False,
                "barrier_battery_reliability_on": False,
                "barrier_dealer_or_buying_friction_on": False,
                "barrier_environmental_impact_on": False,
            }
        )
    posts = pd.DataFrame(rows)
    labeled = label_posts(posts, MockBackend(), tmp_path / "cache.jsonl")
    scores = pd.DataFrame(
        {
            "id": [f"p{index}" for index in range(8)],
            "score": [1, 5, 5, None, 0, 2, 9, 4],
        }
    )
    llm_path = tmp_path / "llm.csv"
    small_path = tmp_path / "small.csv"
    author_path = tmp_path / "authors.csv"
    score_path = tmp_path / "scores.csv"
    labeled.to_csv(llm_path, index=False)
    posts.to_csv(small_path, index=False)
    posts[["id", "author_hash", "created_utc"]].to_csv(author_path, index=False)
    scores.to_csv(score_path, index=False)
    out = tmp_path / "tables"
    tables = run_tables(llm_path, small_path, author_path, out, n_boot=0, score_path=score_path, min_score=2)
    main_n = int(tables["shares"].groupby("audience")["n"].first().sum())
    assert main_n == 8
    info = pd.read_csv(out / "score_sensitivity.csv").iloc[0]
    assert info["status"] == "score_ge_2"
    assert int(info["n_dropped"]) == 2
    assert int(info["n_missing_score"]) == 1
    assert int(info["n_kept"]) == 5
    kept = pd.read_csv(out / "funnel_shares_score_ge_2.csv")
    assert int(kept.groupby("audience")["n"].first().sum()) == 5
    assert (out / "rdit_score_ge_2.csv").exists()
    assert (out / "stage_transition_shocks_score_ge_2.csv").exists()


def test_ryan_is_named_in_the_codebook_and_the_readme():
    root = Path(__file__).resolve().parents[1]
    codebook = (root / "annotation" / "CODEBOOK.md").read_text()
    readme = (root / "annotation" / "README_FOR_RYAN.md").read_text()
    assert "Ryan" in codebook and "Ryan-yu-wei" in codebook
    assert "coder_ryan.csv" in codebook
    assert "annotation_sheet.csv" in readme
    assert "calibration" in readme
    assert "annotation_model_key.csv" in readme
    assert "coder_ryan.csv" in readme
    assert "main" in readme


def test_manuscript_names_ryan_the_14b_model_and_the_score_check():
    from ev_v2.manuscript import render_manuscript

    text = render_manuscript(Path(__file__).resolve().parents[1] / "outputs")
    assert "Ryan" in text
    assert "qwen2.5:14b-instruct" in text
    assert "qwen2.5:7b-instruct" in text
    assert "score under 2" in text
    assert "annotation_sheet.csv" in text
    assert "references.bib" in text
    assert "unvalidated" in text
