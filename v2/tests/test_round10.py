"""Comment label batches stay inside the 30-hour planning budget."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from ev_v2.comment_batches import (
    N_SUBMISSIONS,
    PLAN_HOURS_14B,
    PLAN_HOURS_7B,
    assign_batches,
    comment_slots,
    hours_at,
    materialize_label_input,
)
import fetch_comments
from fetch_comments import needs_subreddit_split


def test_slots_use_the_slow_end_of_each_planning_band():
    slots_14b = comment_slots(PLAN_HOURS_14B)
    slots_7b = comment_slots(PLAN_HOURS_7B)
    assert slots_14b == 1630
    assert slots_7b == 26088
    assert hours_at(N_SUBMISSIONS + slots_14b, PLAN_HOURS_14B) <= 30
    assert hours_at(N_SUBMISSIONS + slots_7b, PLAN_HOURS_7B) <= 30
    assert hours_at(N_SUBMISSIONS + slots_14b + 1, PLAN_HOURS_14B) > 30


def test_only_a_timeout_is_split_across_subreddits():
    assert needs_subreddit_split("timeout", 0)
    assert not needs_subreddit_split({"data": []}, 100)
    assert not needs_subreddit_split({"data": []}, 3)


def test_a_timeout_is_retried_on_the_same_query(monkeypatch):
    calls = []

    def fake(params):
        calls.append(params.get("subreddit"))
        if len(calls) < 3:
            return "timeout"
        return {"data": []}

    monkeypatch.setattr(fetch_comments, "_fetch_comment_page", fake)
    monkeypatch.setattr(fetch_comments.time, "sleep", lambda _seconds: None)
    clusters = fetch_comments._clusters()
    start, end = clusters[0]
    rows, failed = fetch_comments._search_side("unused", start, end, "asc", set(), clusters, ["rivian"])
    assert rows == []
    assert failed is False
    assert calls == [None, None, None]


def test_a_full_page_outside_the_study_asks_only_the_home_subreddit(monkeypatch):
    calls = []
    created = fetch_comments._clusters()[0][0] + 10

    def fake(params):
        calls.append(params.get("subreddit"))
        if params.get("subreddit") == "rivian":
            return {
                "data": [
                    {
                        "id": "home1",
                        "author": "someone",
                        "created_utc": created,
                        "subreddit": "Rivian",
                        "score": 2,
                        "link_id": "t3_abc",
                        "body": "Still in the window.",
                    }
                ]
            }
        return {
            "data": [
                {
                    "id": f"off{index}",
                    "author": "someone",
                    "created_utc": created,
                    "subreddit": "news",
                    "score": 1,
                    "link_id": "t3_z",
                    "body": "elsewhere",
                }
                for index in range(100)
            ]
        }

    monkeypatch.setattr(fetch_comments, "_fetch_comment_page", fake)
    monkeypatch.setattr(fetch_comments.time, "sleep", lambda _seconds: None)
    clusters = fetch_comments._clusters()
    start, end = clusters[0]
    rows, failed = fetch_comments._search_side("unused", start, end, "desc", set(), clusters, ["rivian", "cars"])
    assert failed is False
    assert [row["id"] for row in rows] == ["home1"]
    assert calls[0] is None
    assert calls[1:] == ["rivian", "cars"]


def _events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    cutoff = datetime(2022, 8, 16, tzinfo=timezone.utc)
    stamps = {
        "before": cutoff - timedelta(days=10),
        "after": cutoff + timedelta(days=10),
        "far": cutoff + timedelta(days=170),
    }
    return pd.DataFrame(
        [
            {"author_hash": digest, "created_utc": stamps[when].timestamp(), "text_type": kind}
            for digest, when, kind in rows
        ]
    )


def test_cutoff_crossing_authors_fill_batch_one_before_everyone_else():
    events = _events(
        [
            ("cross", "before", "submission"),
            ("cross", "after", "comment"),
            ("one", "before", "submission"),
        ]
    )
    comments = pd.DataFrame(
        [
            {"id": "far", "author_hash": "one", "created_utc": events.loc[2, "created_utc"], "subreddit": "cars", "link_id": "", "score": 1, "text_type": "comment"},
            {"id": "near", "author_hash": "cross", "created_utc": events.loc[1, "created_utc"], "subreddit": "cars", "link_id": "", "score": 3, "text_type": "comment"},
        ]
    )
    # Pad with non-crossing comments so the 14B cap is the limit.
    extra = []
    base = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    for index in range(1700):
        extra.append(
            {
                "id": f"e{index:04d}",
                "author_hash": "one",
                "created_utc": base + index,
                "subreddit": "cars",
                "link_id": "",
                "score": 1,
                "text_type": "comment",
            }
        )
    comments = pd.concat([comments, pd.DataFrame(extra)], ignore_index=True)
    batch1, batch2, info = assign_batches(comments, events)
    assert info["model_plan"] == "14b"
    assert len(batch1) == 1630
    assert len(batch2) == len(comments) - 1630
    assert batch1.iloc[0]["id"] == "near"
    assert "body" not in batch1.columns
    assert hours_at(N_SUBMISSIONS + len(batch1), PLAN_HOURS_14B) <= 30


def test_a_large_priority_set_uses_the_7b_budget(tmp_path):
    cutoff = datetime(2023, 1, 13, tzinfo=timezone.utc)
    rows = []
    events = []
    for index in range(2000):
        digest = f"a{index:04d}"
        before = (cutoff - timedelta(days=3)).timestamp()
        after = (cutoff + timedelta(days=3)).timestamp()
        events.append({"author_hash": digest, "created_utc": before, "text_type": "submission"})
        events.append({"author_hash": digest, "created_utc": after, "text_type": "comment"})
        rows.append(
            {
                "id": f"c{index:04d}",
                "author_hash": digest,
                "created_utc": after,
                "subreddit": "teslamotors",
                "link_id": "p",
                "score": 2,
                "text_type": "comment",
            }
        )
    batch1, batch2, info = assign_batches(pd.DataFrame(rows), pd.DataFrame(events))
    assert info["model_plan"] == "7b"
    assert info["n_priority"] == 2000
    assert len(batch1) == 2000
    assert len(batch2) == 0
    assert hours_at(N_SUBMISSIONS + len(batch1), PLAN_HOURS_7B) <= 30
    cache = tmp_path / "comments.jsonl"
    cache.write_text('{"id": "c0000", "body": "The price cut landed."}\n')
    batch_path = tmp_path / "batch.csv"
    batch1.to_csv(batch_path, index=False)
    out = tmp_path / "local.csv"
    assert materialize_label_input(batch_path, cache, out) == 2000
    local = pd.read_csv(out)
    assert local.loc[local["id"] == "c0000", "body"].iloc[0] == "The price cut landed."
    assert "The price cut landed." not in batch_path.read_text()
