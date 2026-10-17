"""Dump extractor tests. They build a tiny zstd file and do not touch the network."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import zstandard

from ev_v2.dumps import (
    KNOWN_BYTES,
    PIECE_LENGTH,
    POST_COLUMNS,
    SUBMISSION_FILES,
    TORRENT_INFOHASH,
    compare_counts,
    extract_paths,
)


def _ts(day: str) -> int:
    year, month, dom = (int(part) for part in day.split("-"))
    return int(datetime(year, month, dom, tzinfo=timezone.utc).timestamp())


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _write_zst(path: Path, objects: list[dict], frames: int = 1) -> None:
    payload = b"".join(_line(obj) for obj in objects)
    if frames == 1:
        path.write_bytes(zstandard.ZstdCompressor().compress(payload))
        return
    # Two independent frames, which is how some dump files are packed.
    mid = max(1, len(objects) // 2)
    compressor = zstandard.ZstdCompressor()
    first = compressor.compress(b"".join(_line(obj) for obj in objects[:mid]))
    second = compressor.compress(b"".join(_line(obj) for obj in objects[mid:]))
    path.write_bytes(first + second)


def _post(**kwargs) -> dict:
    base = {
        "id": "abc",
        "subreddit": "cars",
        "author": "someone",
        "title": "Looking at a Tesla",
        "selftext": "Is the payment realistic?",
        "created_utc": _ts("2016-04-02"),
        "score": 3,
        "num_comments": 1,
        "url": "https://example.invalid/p",
        "link_flair_text": "Question",
    }
    base.update(kwargs)
    return base


def test_word_boundary_subreddit_and_date_filters(tmp_path: Path):
    posts = [
        _post(id="keep_tesla", title="Looking at a Tesla"),
        _post(id="every", title="Not every car is worth it", selftext="Just shopping."),
        _post(id="vehicles", title="Are electric vehicles cheaper?", selftext=""),
        _post(id="ioniq_token", title="The ioniq5 forum is elsewhere", selftext=""),
        _post(id="phrase", subreddit="Autos", title="An electric vehicle in the city", selftext="", created_utc=_ts("2024-06-01")),
        _post(id="early", title="Tesla in 2014", created_utc=_ts("2014-12-31")),
        _post(id="too_late", title="Tesla after the window", created_utc=_ts("2026-10-01")),
        _post(id="last_day", title="Rivian on the last included day", created_utc=_ts("2026-09-30")),
        _post(id="other_sub", subreddit="teslamotors", title="Tesla delivery day"),
        _post(id="comment_shaped", title=None, body="tesla comment", selftext=None),
        _post(id="carb", subreddit="carbuying", title="Dealer wants a markup on the Ioniq", selftext=""),
    ]
    # title=None should be omitted so the object looks like a comment.
    posts = [post for post in posts if post["id"] != "comment_shaped"]
    posts.append({"id": "c1", "subreddit": "cars", "body": "a tesla comment", "created_utc": _ts("2020-01-01")})
    path = tmp_path / "mixed.zst"
    _write_zst(path, posts, frames=2)
    out = tmp_path / "out.csv"
    stats = extract_paths([path], out)
    frame = pd.read_csv(out)
    assert list(frame.columns) == POST_COLUMNS
    assert set(frame["id"]) == {"keep_tesla", "phrase", "last_day", "carb"}
    assert stats.rows_kept == 4
    assert stats.bad_json == 0
    assert int((frame["community_group"] == "mainstream_car").sum()) == 4
    assert set(frame["sample_stratum"]) == {"dump"}
    carb = frame.loc[frame["id"] == "carb"].iloc[0]
    assert carb["query"] == "ioniq"
    phrase = frame.loc[frame["id"] == "phrase"].iloc[0]
    assert phrase["subreddit"] == "Autos"
    assert phrase["query"] == "electric vehicle"
    assert "tesla" in frame.loc[frame["id"] == "keep_tesla", "query"].iloc[0]


def test_duplicate_ids_collapse_across_files(tmp_path: Path):
    first = tmp_path / "a.zst"
    second = tmp_path / "b.zst"
    _write_zst(first, [_post(id="same", title="Tesla one")])
    _write_zst(second, [_post(id="same", title="Tesla two", subreddit="Autos")])
    out = tmp_path / "out.csv"
    stats = extract_paths([first, second], out)
    frame = pd.read_csv(out)
    assert list(frame["id"]) == ["same"]
    assert stats.skipped_duplicate == 1
    assert frame.iloc[0]["subreddit"] == "cars"


def test_bad_json_is_counted_and_skipped(tmp_path: Path):
    payload = _line(_post(id="ok", title="Tesla ok")) + b"not-json\n" + _line(_post(id="ok2", title="electric car"))
    path = tmp_path / "bad.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    stats = extract_paths([path], tmp_path / "out.csv")
    assert stats.bad_json == 1
    assert stats.rows_kept == 2


def test_known_size_warning(tmp_path: Path):
    path = tmp_path / "cars_submissions.zst"
    _write_zst(path, [_post(id="t", title="Tesla")])
    stats = extract_paths([path], tmp_path / "out.csv")
    assert stats.size_warnings
    assert TORRENT_INFOHASH in stats.size_warnings[0]
    assert str(KNOWN_BYTES["cars_submissions.zst"]) in stats.size_warnings[0]


def test_compare_does_not_overwrite_either_count():
    dump = pd.DataFrame({"subreddit": ["cars"], "year": [2015], "n_posts": [10]})
    api = pd.DataFrame({"subreddit": ["cars", "Autos"], "year": [2015, 2015], "n_posts": [12, 3]})
    compared = compare_counts(dump, api)
    cars = compared.loc[compared["subreddit"] == "cars"].iloc[0]
    assert int(cars["dump_posts"]) == 10
    assert int(cars["api_posts"]) == 12
    assert int(cars["dump_minus_api"]) == -2
    autos = compared.loc[compared["subreddit"] == "Autos"].iloc[0]
    assert pd.isna(autos["dump_posts"])
    assert int(autos["api_posts"]) == 3


def test_torrent_file_table_matches_the_parsed_torrent():
    by_name = {item["path"]: item for item in SUBMISSION_FILES}
    assert by_name["subreddits25/cars_submissions.zst"]["bytes"] == 224_119_795
    assert by_name["subreddits25/Autos_submissions.zst"]["bytes"] == 68_306_712
    assert by_name["subreddits25/carbuying_submissions.zst"]["bytes"] == 7_861_374
    assert by_name["subreddits25/cars_submissions.zst"]["index"] == 53107
    assert PIECE_LENGTH == 67_108_864
    assert TORRENT_INFOHASH == "3e3f64dee22dc304cdd2546254ca1f8e8ae542b4"
