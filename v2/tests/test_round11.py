"""Resume, dedup, and merge checks for the 14B label run. Mock rows only."""

import json

import pandas as pd
import pytest

from llm_label import (
    MockBackend,
    label_posts,
    main,
    merge_label_files,
    prompt_hash,
    smoke_limit_minutes,
)


def _posts():
    return pd.DataFrame(
        [
            {"id": "p1", "title": "Apartment charging", "selftext": "Can I make this work?"},
            {"id": "p2", "title": "Price", "selftext": "The credit changed the quote."},
            {"id": "p3", "title": "Pickup", "body": "I took the car home Friday.", "text_type": "comment"},
            {"id": "p4", "title": "Range", "excerpt": "The guess-o-meter dropped."},
        ]
    )


def _ids(path):
    return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]


class _StopAfter:
    name = "mock"
    version = "mock-0"

    def __init__(self, limit: int):
        self.limit = limit
        self.calls = 0

    def complete(self, prompts):
        self.calls += 1
        if self.calls > self.limit:
            raise ConnectionError("ollama down")
        return MockBackend().complete(prompts)


def test_smoke_limit_for_the_60_hour_14b_run():
    minutes = smoke_limit_minutes()
    assert minutes == pytest.approx(60 * 100 / (22827 + 26088) * 60)
    assert round(minutes, 1) == 7.4


def test_a_crash_resumes_without_duplicating_finished_ids(tmp_path):
    posts = _posts()
    cache = tmp_path / "labels.jsonl"
    with pytest.raises(ConnectionError):
        label_posts(posts, _StopAfter(1), cache, batch_size=1)
    assert _ids(cache) == ["p1"]
    finished = label_posts(posts, MockBackend(), cache, batch_size=1)
    assert _ids(cache) == ["p1", "p2", "p3", "p4"]
    assert finished["id"].tolist() == ["p1", "p2", "p3", "p4"]
    assert finished["id"].is_unique
    assert set(finished["text_type"]) == {"submission", "comment"}


def test_a_second_pass_does_not_append_cached_ids(tmp_path):
    posts = _posts().iloc[:2]
    cache = tmp_path / "labels.jsonl"
    label_posts(posts, MockBackend(), cache, batch_size=2)
    before = cache.read_text()
    label_posts(posts, MockBackend(), cache, batch_size=2)
    assert cache.read_text() == before
    assert _ids(cache) == ["p1", "p2"]


def test_a_torn_last_line_is_cut_and_not_glued_to_the_next_row(tmp_path):
    cache = tmp_path / "labels.jsonl"
    good = {
        "id": "p1",
        "text_type": "submission",
        "model_name": "mock",
        "model_version": "mock-0",
        "prompt_hash": prompt_hash(),
        "prompt_version": "funnel-v2",
        "stage": "evaluation",
        "families": "charging_energy",
        "aspects": "charging_access",
        "stance": "mixed",
        "evidence": "mock evidence",
        "ok": True,
        "validated": False,
    }
    cache.write_bytes(json.dumps(good).encode() + b"\n" + b'{"id": "p2", "stage":')
    label_posts(_posts().iloc[:2], MockBackend(), cache, batch_size=1)
    lines = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    assert [row["id"] for row in lines] == ["p1", "p2"]
    assert all(row["ok"] is True for row in lines)
    assert "p2" not in cache.read_text().splitlines()[0]


def test_a_complete_line_without_a_newline_is_kept_separate(tmp_path):
    cache = tmp_path / "labels.jsonl"
    label_posts(_posts().iloc[:1], MockBackend(), cache, batch_size=1)
    text = cache.read_text().rstrip("\n")
    cache.write_text(text)
    label_posts(_posts().iloc[:2], MockBackend(), cache, batch_size=1)
    lines = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    assert [row["id"] for row in lines] == ["p1", "p2"]


def test_max_new_ends_a_session_and_the_same_command_continues(tmp_path):
    posts = _posts()
    cache = tmp_path / "labels.jsonl"
    first = label_posts(posts, MockBackend(), cache, batch_size=1, max_new=2)
    assert first["id"].tolist() == ["p1", "p2"]
    second = label_posts(posts, MockBackend(), cache, batch_size=1, max_new=2)
    assert second["id"].tolist() == ["p1", "p2", "p3", "p4"]
    assert _ids(cache) == ["p1", "p2", "p3", "p4"]


def _expect(tmp_path, ids):
    path = tmp_path / "expect.csv"
    pd.DataFrame({"id": ids}).to_csv(path, index=False)
    return path


def _cache_line(doc_id, stage="evaluation"):
    return json.dumps(
        {
            "id": doc_id,
            "text_type": "submission",
            "model_name": "mock",
            "model_version": "mock-0",
            "prompt_hash": "abc",
            "prompt_version": "funnel-v2",
            "stage": stage,
            "families": "charging_energy",
            "aspects": "charging_access",
            "stance": "mixed",
            "evidence": "mock evidence",
            "ok": True,
            "validated": False,
        }
    )


def test_merge_collapses_identical_duplicates_and_rejects_a_conflict(tmp_path):
    expect = _expect(tmp_path, ["p1", "p2"])
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    line = _cache_line("p1")
    first.write_text(line + "\n" + line + "\n")
    second.write_text(_cache_line("p2") + "\n")
    out = tmp_path / "merged.csv"
    report = merge_label_files([expect], [first, second], out)
    assert report["ok"] is True
    merged = pd.read_csv(out)
    assert merged["id"].tolist() == ["p1", "p2"]
    assert merged["id"].is_unique

    second.write_text(_cache_line("p2", stage="purchase") + "\n" + _cache_line("p2", stage="awareness") + "\n")
    blocked = tmp_path / "blocked.csv"
    bad = merge_label_files([expect], [first, second], blocked)
    assert bad["ok"] is False
    assert bad["n_conflicts"] == 1
    assert not blocked.exists()


def test_merge_requires_every_expected_id_and_valid_json(tmp_path):
    expect = _expect(tmp_path, ["p1", "p2"])
    cache = tmp_path / "a.jsonl"
    cache.write_text(_cache_line("p1") + "\n" + "{not json}\n")
    out = tmp_path / "merged.csv"
    report = merge_label_files([expect], [cache], out)
    assert report["ok"] is False
    assert report["n_missing"] == 1
    assert report["n_invalid"] == 1
    assert not out.exists()
    code = main(
        [
            "--merge-validate",
            "--expect",
            str(expect),
            "--caches",
            str(cache),
            "--out",
            str(out),
        ]
    )
    assert code == 1


def test_merge_command_writes_the_csv_when_the_cache_matches(tmp_path):
    expect = _expect(tmp_path, ["p1"])
    cache = tmp_path / "a.jsonl"
    cache.write_text(_cache_line("p1") + "\n")
    out = tmp_path / "merged.csv"
    code = main(
        [
            "--merge-validate",
            "--expect",
            str(expect),
            "--caches",
            str(cache),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert pd.read_csv(out)["id"].tolist() == ["p1"]
