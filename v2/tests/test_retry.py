from datetime import datetime, timedelta, timezone

import pytest

from ev_v2.mainstream import month_spans
from ev_v2.retry import (
    Checkpoint,
    FetchError,
    classify_and_backoff,
    resolve_window,
    split_span,
)


def _utc(year, month, day, hour=0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def test_text_queries_are_separate_keywords():
    from ev_v2.mainstream import MAINSTREAM_QUERY, text_queries

    queries = text_queries()
    assert "tesla" in queries
    assert "electric vehicle" in queries
    assert all(" OR " not in query for query in queries)
    assert " OR " in MAINSTREAM_QUERY


def test_mainstream_months_cover_2015_through_september_2026():
    from datetime import date

    spans = month_spans(date(2015, 1, 1), date(2026, 10, 1))
    assert len(spans) == 141
    assert spans[0][0].strftime("%Y-%m") == "2015-01"
    assert spans[-1][0].strftime("%Y-%m") == "2026-09"
    assert spans[-1][1].strftime("%Y-%m-%d") == "2026-10-01"


def test_fast_422_waits_out_the_rate_limit_reset():
    kind, seconds = classify_and_backoff(
        0,
        status=422,
        elapsed_s=0.4,
        reset_header="24",
        error="Timeout. Maybe slow down a bit",
    )
    assert kind == "rate_limit"
    assert seconds == pytest.approx(24.5)


def test_slow_422_is_a_query_timeout_with_a_short_backoff():
    kind, seconds = classify_and_backoff(
        0,
        status=422,
        elapsed_s=5.7,
        reset_header="40",
        error="Timeout. Maybe slow down a bit",
    )
    assert kind == "timeout"
    assert seconds == pytest.approx(1.5)
    assert seconds <= 8


def test_429_reset_is_capped():
    kind, seconds = classify_and_backoff(
        0, status=429, elapsed_s=0.2, reset_header="200", error=None
    )
    assert kind == "rate_limit"
    assert seconds == 90


def test_split_span_stops_at_the_minimum():
    start = _utc(2016, 2, 1)
    end = _utc(2016, 3, 1)
    pieces = split_span(start, end, timedelta(days=1))
    assert pieces is not None
    left, right = pieces
    assert left[0] == start and right[1] == end
    assert left[1] == right[0]
    day = (_utc(2016, 2, 2), _utc(2016, 2, 3))
    assert split_span(day[0], day[1], timedelta(days=1)) is None
    half_day = split_span(_utc(2016, 2, 2), _utc(2016, 2, 3), timedelta(hours=6))
    assert half_day is not None


def test_checkpoint_roundtrip_skips_completed_windows(tmp_path):
    path = tmp_path / "ckpt.json"
    ckpt = Checkpoint(path)
    calls = {"n": 0, "ok": 0}

    def fetch(start, end):
        calls["n"] += 1
        span = (end - start).total_seconds()
        if span > 10 * 86400:
            raise FetchError("timeout", kind="timeout")
        calls["ok"] += 1
        return 3

    start, end = _utc(2016, 2, 1), _utc(2016, 3, 2)
    out = resolve_window(
        kind="agg",
        subreddit="cars",
        query="tesla",
        start=start,
        end=end,
        fetch=fetch,
        checkpoint=ckpt,
        min_span=timedelta(days=1),
    )
    assert out.status == "ok"
    assert out.n == calls["ok"] * 3
    assert calls["n"] > calls["ok"] > 1
    first_calls = calls["n"]

    reloaded = Checkpoint(path)
    again = resolve_window(
        kind="agg",
        subreddit="cars",
        query="tesla",
        start=start,
        end=end,
        fetch=fetch,
        checkpoint=reloaded,
        min_span=timedelta(days=1),
    )
    assert again.n == out.n
    assert calls["n"] == first_calls


def test_rate_limit_is_deferred_and_not_checkpointed(tmp_path):
    path = tmp_path / "ckpt.json"
    ckpt = Checkpoint(path)
    calls = {"n": 0}

    def fetch(start, end):
        calls["n"] += 1
        raise FetchError("slow down", kind="rate_limit", status=422)

    start, end = _utc(2024, 6, 1), _utc(2024, 7, 1)
    out = resolve_window(
        kind="agg",
        subreddit="cars",
        query="tesla",
        start=start,
        end=end,
        fetch=fetch,
        checkpoint=ckpt,
        min_span=timedelta(days=1),
    )
    assert out.status == "deferred"
    assert calls["n"] == 1
    assert ckpt.failed_records() == []
    assert ckpt.get("agg|cars|2024-06-01T00:00:00Z|2024-07-01T00:00:00Z|tesla") is None

    again = resolve_window(
        kind="agg",
        subreddit="cars",
        query="tesla",
        start=start,
        end=end,
        fetch=fetch,
        checkpoint=ckpt,
        min_span=timedelta(days=1),
    )
    assert again.status == "deferred"
    assert calls["n"] == 2
