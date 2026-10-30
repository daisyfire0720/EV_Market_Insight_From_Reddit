"""Stream Pushshift / Arctic Shift .zst dumps and keep shopper EV posts.

The files are newline-delimited JSON, compressed with zstd, one Reddit
submission per line. Two layouts are accepted:

- Per-subreddit files from the top-40,000 split, named
  ``<subreddit>_submissions.zst``. The 2005-06 through 2025-12 torrent is
  infohash ``3e3f64dee22dc304cdd2546254ca1f8e8ae542b4``.
- Monthly files named ``RS_YYYY-MM.zst``, which contain every subreddit.
  The subreddit field is what keeps r/cars, r/Autos, and r/carbuying.

The keyword rule is the same word-boundary match used everywhere else in
``v2`` (``ev_v2.textmatch``), on title plus selftext, for
``GENERAL_KEYWORDS``. That is not the Arctic Shift aggregate query that
produced the 16,993 count. The archive's simple keyword search can hit
``ioniq`` inside ``ioniq5`` and ``electric vehicle`` inside ``electric
vehicles``. This filter does not. Counts from the two sources are reported
side by side and are not forced to match.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from ev_v2.arctic import GENERAL_KEYWORDS, GENERAL_SUBREDDITS
from ev_v2.mainstream import MAINSTREAM_END, MAINSTREAM_START
from ev_v2.textmatch import find_keywords

# Parsed from the .torrent recovered on 17 Oct 2026. Indexes are 1-based
# positions in that torrent's file list and apply only to this infohash.
TORRENT_INFOHASH = "3e3f64dee22dc304cdd2546254ca1f8e8ae542b4"
TORRENT_PAGE = f"https://academictorrents.com/details/{TORRENT_INFOHASH}"
TORRENT_WAYBACK = (
    "https://web.archive.org/web/20260430154341/"
    "https://academictorrents.com/download/"
    f"{TORRENT_INFOHASH}.torrent"
)
TORRENT_COMMENT = "Reddit subreddits from 2005-06 to 2025-12"
PIECE_LENGTH = 67_108_864
DUMP_INDEX = "https://github.com/ArthurHeitmann/arctic_shift/blob/master/download_links.md"

# Byte lengths are the ``length`` fields in the torrent, not estimates.
SUBMISSION_FILES = (
    {
        "subreddit": "Autos",
        "path": "subreddits25/Autos_submissions.zst",
        "bytes": 68_306_712,
        "index": 4280,
    },
    {
        "subreddit": "carbuying",
        "path": "subreddits25/carbuying_submissions.zst",
        "bytes": 7_861_374,
        "index": 53039,
    },
    {
        "subreddit": "cars",
        "path": "subreddits25/cars_submissions.zst",
        "bytes": 224_119_795,
        "index": 53107,
    },
)

# Monthly submission file checked by parsing its .torrent on 17 Oct 2026.
# Comments in the same torrent are larger and are not used.
RS_2026_09_BYTES = 22_407_703_827
RS_2026_09_TORRENT = "https://academictorrents.com/details/98b6dbad155750c4d5f65d5923e0c2d11f46da85"

# January–September 2026 are outside the per-subreddit torrent, which stops
# at the end of 2025. These are the monthly torrents listed in download_links.md
# as fetched on 17 Oct 2026. Each one is a full-Reddit month, not three subs.
MONTHLY_2026_TORRENTS = (
    ("2026-01", "https://academictorrents.com/details/8412b89151101d88c915334c45d9c223169a1a60"),
    ("2026-02", "https://academictorrents.com/details/c5ba00048236b60f819dbf010e9034d24fc291fb"),
    ("2026-03", "https://academictorrents.com/details/668087bb8c8c9c763b27a1a4c5e7fcb6add25f2c"),
    ("2026-04", "https://academictorrents.com/details/85d017ddd06920534187e7d45f21c7cec90c9bca"),
    ("2026-05", "https://academictorrents.com/details/55199eff9368cde1f5c1262dd7c1af09f7503ea5"),
    ("2026-06", "https://academictorrents.com/details/3bac8bd352bbb74bbb23df4273cf3da5d66ee5a5"),
    ("2026-07", "https://academictorrents.com/details/e04a4fda12826ab1d181eef6512b36aca63c70ff"),
    ("2026-08", "https://academictorrents.com/details/f4848163a5fb650e4b15eee0202858447bca120a"),
    ("2026-09", RS_2026_09_TORRENT),
)

POST_COLUMNS = [
    "id",
    "created_utc",
    "author",
    "title",
    "selftext",
    "score",
    "num_comments",
    "subreddit",
    "url",
    "link_flair_text",
    "query",
    "sample_after",
    "sample_before",
    "sample_truncated",
    "community_group",
    "sample_stratum",
    "created_date",
    "month",
]

CANONICAL_SUBREDDIT = {name.lower(): name for name in GENERAL_SUBREDDITS}
KNOWN_BYTES = {Path(item["path"]).name: item["bytes"] for item in SUBMISSION_FILES}


@dataclass
class ExtractStats:
    files: int = 0
    rows_scanned: int = 0
    rows_kept: int = 0
    bad_json: int = 0
    skipped_not_submission: int = 0
    skipped_subreddit: int = 0
    skipped_date: int = 0
    skipped_keyword: int = 0
    skipped_duplicate: int = 0
    size_warnings: list[str] = field(default_factory=list)
    by_subreddit_year: dict[tuple[str, int], int] = field(default_factory=dict)

    def add(self, subreddit: str, year: int) -> None:
        key = (subreddit, year)
        self.by_subreddit_year[key] = self.by_subreddit_year.get(key, 0) + 1
        self.rows_kept += 1


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def canonical_subreddit(value) -> str | None:
    name = _as_text(value).strip()
    if not name:
        return None
    return CANONICAL_SUBREDDIT.get(name.lower())


def created_timestamp(value) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _bounds(start: date, end: date) -> tuple[int, int]:
    start_ts = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
    return start_ts, end_ts


def iter_zst_lines(path: Path):
    """Yield decoded lines from a zstd file, including multi-frame streams."""
    import zstandard

    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with path.open("rb") as raw:
        with dctx.stream_reader(raw, read_across_frames=True) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="\n")
            for line in text:
                yield line


def record_from_object(obj: dict, start: date, end: date, stats: ExtractStats) -> dict | None:
    """Return one output row, or None if the object is out of scope."""
    if not isinstance(obj, dict):
        stats.skipped_not_submission += 1
        return None
    # Comment dumps have a body and no title. Do not treat them as submissions.
    if "title" not in obj and "body" in obj:
        stats.skipped_not_submission += 1
        return None
    subreddit = canonical_subreddit(obj.get("subreddit"))
    if subreddit is None:
        stats.skipped_subreddit += 1
        return None
    created = created_timestamp(obj.get("created_utc"))
    if created is None:
        stats.skipped_date += 1
        return None
    start_ts, end_ts = _bounds(start, end)
    if created < start_ts or created >= end_ts:
        stats.skipped_date += 1
        return None
    title = _as_text(obj.get("title"))
    selftext = _as_text(obj.get("selftext"))
    hits = find_keywords(f"{title}\n{selftext}", list(GENERAL_KEYWORDS))
    if not hits:
        stats.skipped_keyword += 1
        return None
    post_id = _as_text(obj.get("id")).strip()
    if not post_id:
        stats.skipped_not_submission += 1
        return None
    moment = datetime.fromtimestamp(created, tz=timezone.utc)
    return {
        "id": post_id,
        "created_utc": created,
        "author": _as_text(obj.get("author")),
        "title": title,
        "selftext": selftext,
        "score": obj.get("score") if obj.get("score") is not None else "",
        "num_comments": obj.get("num_comments") if obj.get("num_comments") is not None else "",
        "subreddit": subreddit,
        "url": _as_text(obj.get("url")),
        "link_flair_text": _as_text(obj.get("link_flair_text")),
        "query": ", ".join(hits),
        "sample_after": start.isoformat(),
        "sample_before": end.isoformat(),
        "sample_truncated": False,
        "community_group": "mainstream_car",
        "sample_stratum": "dump",
        "created_date": moment.date().isoformat(),
        "month": moment.strftime("%Y-%m"),
    }


def check_known_size(path: Path, stats: ExtractStats) -> None:
    expected = KNOWN_BYTES.get(path.name)
    if expected is None:
        return
    actual = path.stat().st_size
    if actual != expected:
        stats.size_warnings.append(
            f"{path.name} is {actual} bytes; the torrent lists {expected} bytes for infohash {TORRENT_INFOHASH}"
        )


# Exact field spellings used in Pushshift / Arctic Shift submission dumps.
# A line that contains none of these is not one of the three shopper subs.
SHOPPER_SUBREDDIT_TOKENS = (
    '"subreddit":"cars"',
    '"subreddit":"Autos"',
    '"subreddit":"carbuying"',
    '"subreddit": "cars"',
    '"subreddit": "Autos"',
    '"subreddit": "carbuying"',
)


def extract_paths(
    paths: list[Path],
    out_csv: Path,
    start: date = MAINSTREAM_START,
    end: date = MAINSTREAM_END,
    fast_subreddit_scan: bool = False,
) -> ExtractStats:
    """Stream ``paths`` and write kept submissions to ``out_csv``.

    ``fast_subreddit_scan`` skips JSON parsing for lines that do not contain
    a shopper subreddit field. It is for the full-Reddit monthly files.
    Per-subreddit files are already one community, so the flag stays off.
    """
    stats = ExtractStats()
    seen: set[str] = set()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for path in paths:
            stats.files += 1
            check_known_size(path, stats)
            for line in iter_zst_lines(path):
                stripped = line.strip()
                if not stripped:
                    continue
                if fast_subreddit_scan and not any(token in stripped for token in SHOPPER_SUBREDDIT_TOKENS):
                    stats.rows_scanned += 1
                    stats.skipped_subreddit += 1
                    if stats.rows_scanned % 2_000_000 == 0:
                        print(f"  scanned {stats.rows_scanned} kept {stats.rows_kept}", flush=True)
                    continue
                stats.rows_scanned += 1
                if stats.rows_scanned % 2_000_000 == 0:
                    print(f"  scanned {stats.rows_scanned} kept {stats.rows_kept}", flush=True)
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    stats.bad_json += 1
                    continue
                row = record_from_object(obj, start, end, stats)
                if row is None:
                    continue
                if row["id"] in seen:
                    stats.skipped_duplicate += 1
                    continue
                seen.add(row["id"])
                writer.writerow(row)
                stats.add(row["subreddit"], int(row["month"][:4]))
    return stats


def counts_frame(stats: ExtractStats):
    import pandas as pd

    rows = [
        {"subreddit": sub, "year": year, "n_posts": n}
        for (sub, year), n in sorted(stats.by_subreddit_year.items())
        if n
    ]
    return pd.DataFrame(rows, columns=["subreddit", "year", "n_posts"])


def compare_counts(dump_counts, api_counts):
    """Side-by-side yearly counts. Neither column is edited to match the other."""
    import pandas as pd

    left = dump_counts.rename(columns={"n_posts": "dump_posts"}) if len(dump_counts) else pd.DataFrame(columns=["subreddit", "year", "dump_posts"])
    right = api_counts.rename(columns={"n_posts": "api_posts"})
    keep = [col for col in ("subreddit", "year", "api_posts") if col in right.columns]
    merged = left.merge(right[keep], on=["subreddit", "year"], how="outer")
    for col in ("dump_posts", "api_posts"):
        if col not in merged.columns:
            merged[col] = pd.NA
    merged["dump_minus_api"] = pd.to_numeric(merged["dump_posts"], errors="coerce") - pd.to_numeric(
        merged["api_posts"], errors="coerce"
    )
    return merged.sort_values(["subreddit", "year"]).reset_index(drop=True)


def render_dump_markdown(counts_path: Path | None = None, compare_path: Path | None = None, stats: ExtractStats | None = None) -> str:
    """Data-card section. Lengths and hashes are from the torrent, not from a download estimate."""
    lines = [
        "## Shopper text from the dumps",
        "",
        "Full text for r/cars, r/Autos, and r/carbuying is read from the zstd dumps, not from the Arctic Shift search API. "
        "`v2/extract_from_dumps.py` streams each file, keeps a submission when the subreddit is one of those three, "
        f"the timestamp is in `{MAINSTREAM_START.isoformat()}` inclusive through `{MAINSTREAM_END.isoformat()}` exclusive, "
        "and the title or selftext matches a keyword on a word boundary. Keywords: "
        + ", ".join(f"`{kw}`" for kw in GENERAL_KEYWORDS)
        + ". Bare `ev` is not a keyword. Output columns match `posts_downloaded.csv`.",
        "",
        "The 16,993 figure is the Arctic Shift aggregate for "
        '`tesla OR rivian OR ioniq OR "electric vehicle" OR "electric car"`. '
        "That search can match a keyword inside a longer token (`ioniq` inside `ioniq5`, `electric vehicle` inside `electric vehicles`). "
        "The dump filter does not. A difference between the two columns is expected. Do not retune the filter to force the totals together.",
        "",
        "### Which torrent",
        "",
        f"Per-subreddit submissions, 2005-06 through 2025-12. Infohash `{TORRENT_INFOHASH}`. "
        f"Torrent comment: `{TORRENT_COMMENT}`. Piece length: {PIECE_LENGTH:,} bytes. "
        "Root folder name in the torrent: `reddit`. File names:",
        "",
        "| File | Subreddit | Bytes in the torrent | 1-based index |",
        "|---|---|---:|---:|",
    ]
    for item in SUBMISSION_FILES:
        lines.append(
            f"| `{item['path']}` | r/{item['subreddit']} | {item['bytes']:,} | {item['index']} |"
        )
    lines.extend(
        [
            "",
            f"The Academic Torrents details page ({TORRENT_PAGE}) returned HTTP 404 on 17 October 2026, "
            "and the Academic Torrents announce URL answered `Torrent not registered with Academic Torrents` for this infohash. "
            "The `.torrent` used to read those lengths was the Internet Archive copy of the old download URL: "
            f"{TORRENT_WAYBACK}. "
            "Public trackers still had peers that day (`udp://tracker.opentrackr.org:1337/announce` among them). "
            "Select the three `*_submissions.zst` files by name. The indexes above are only for this infohash. "
            "Do not select the `*_comments.zst` files. Comments for r/cars alone are 2,164,503,420 bytes in the same torrent. "
            "Because pieces are 64 MiB, a client downloads every piece that overlaps a selected file. "
            "For these three files that is on the order of 512 MiB, not the 3.6 TiB torrent.",
            "",
            "That torrent stops at the end of 2025. January through September 2026 are only in the monthly dumps indexed at "
            f"[download_links.md]({DUMP_INDEX}). "
            f"The 2026-09 torrent ({RS_2026_09_TORRENT}) lists `submissions/RS_2026-09.zst` at {RS_2026_09_BYTES:,} bytes. "
            "A 2025-12 page still up on 17 October 2026 lists `submissions/RS_2025-12.zst` at 18.24 GB. "
            "Keeping all nine submissions files plus their comments files does not fit in about 250 GB free. "
            "One submissions file at a time does fit. The 19 October 2026 check, including peer speed and what was actually streamed, is in the round 5 feasibility section. "
            "The same script streams an `RS_*.zst` file and filters to the three subreddits. "
            "Monthly torrents for January–September 2026:",
            "",
        ]
    )
    for month, url in MONTHLY_2026_TORRENTS:
        lines.append(f"- {month}: {url}")
    lines.extend(
        [
            "",
            "### How to run it",
            "",
            "Install `zstandard` from `v2/requirements.txt`. Point the script at the files you have. It writes a gitignored CSV under `v2/data/raw/` and a small count table under `v2/data_card/`.",
            "",
            "```bash",
            "python v2/extract_from_dumps.py \\",
            "  /path/to/subreddits25/Autos_submissions.zst \\",
            "  /path/to/subreddits25/carbuying_submissions.zst \\",
            "  /path/to/subreddits25/cars_submissions.zst",
            "```",
            "",
            "Add any `RS_2026-01.zst` through `RS_2026-09.zst` files to the same command. "
            "The script warns if a known file name has a different byte length than the torrent. "
            "It does not download the torrent itself.",
            "",
            "With aria2, after the infohash has been checked:",
            "",
            "```bash",
            "aria2c --file-allocation=none --seed-time=0 \\",
            '  --select-file=4280,53039,53107 \\',
            "  --dir=/path/to/dumps \\",
            "  reddit-subreddits.torrent",
            "```",
            "",
            "### Checksums and licensing",
            "",
            f"The integrity check shipped with the torrent is the infohash `{TORRENT_INFOHASH}` plus the per-file lengths in the table above. "
            "Piece hashes inside the `.torrent` cover the bytes a client writes. "
            "There is no separate MD5 or SHA-256 of each `.zst` file on the Academic Torrents page that could be read on 17 October 2026, because that page returned 404. "
            "The live monthly page for December 2025 (https://academictorrents.com/details/481bf2eac43172ae724fd6c75dbcb8e27de77734), fetched on 17 October 2026, says `No license specified, the work may be protected by copyright` in its `dc.rights` meta tag, and the bibtex `license` field on that page is empty. "
            "No open license is stated for these dumps. "
            "A r/pushshift post describes hosts taking Academic Torrents pages down after a request from Reddit: "
            "https://www.reddit.com/r/pushshift/comments/1v50ved/upon_reddits_request_i_am_taking_down_my_academic/. "
            "This repository does not commit the dumps or the full shopper text. Usernames stay in the gitignored CSV. "
            "The committed excerpt is `v2/sample/mainstream_dump_excerpt.csv`: up to two posts per subreddit-year, author hashed, body cut at 400 characters.",
            "",
            "Suggested citation for the per-subreddit split, which is the citation the archived Academic Torrents page carried: "
            f"Watchful1. (2026). Subreddit comments/submissions 2005-06 to 2025-12 [Data set]. Academic Torrents. {TORRENT_PAGE}. "
            "Pushshift collected posts before April 2023. u/raiderbdev collected posts after that. u/Watchful1 split them into per-subreddit files. "
            "Field notes: https://github.com/ArthurHeitmann/arctic_shift/blob/master/file_content_explanations.md.",
            "",
        ]
    )
    if stats is not None:
        lines.append(
            f"Latest local extract: {stats.files} file(s), {stats.rows_scanned:,} lines scanned, {stats.rows_kept:,} posts kept, "
            f"{stats.bad_json:,} bad JSON lines, {stats.skipped_duplicate:,} duplicate ids."
        )
        if stats.size_warnings:
            lines.append("")
            lines.append("Size warnings:")
            for warning in stats.size_warnings:
                lines.append(f"- {warning}")
        lines.append("")
    if counts_path is not None and counts_path.exists():
        import pandas as pd

        counts = pd.read_csv(counts_path)
        if not counts.empty:
            total = int(pd.to_numeric(counts["n_posts"], errors="coerce").fillna(0).sum())
            lines.append(f"Dump keyword posts written by the extractor: **{total:,}**.")
            lines.append("")
            lines.append("| Subreddit | Year | Dump posts |")
            lines.append("|---|---:|---:|")
            for _, row in counts.sort_values(["subreddit", "year"]).iterrows():
                lines.append(f"| r/{row['subreddit']} | {int(row['year'])} | {int(row['n_posts'])} |")
            lines.append("")
    if compare_path is not None and compare_path.exists():
        import pandas as pd

        compared = pd.read_csv(compare_path)
        if not compared.empty and "dump_posts" in compared.columns and "api_posts" in compared.columns:
            dump_sum = int(pd.to_numeric(compared["dump_posts"], errors="coerce").fillna(0).sum())
            api_sum = int(pd.to_numeric(compared["api_posts"], errors="coerce").fillna(0).sum())
            lines.append(
                f"Across the rows in the comparison file, dump posts sum to {dump_sum:,} and API posts sum to {api_sum:,}. "
                "The API sum over a full 2015–September 2026 table is 16,993. "
                "A dump file that ends in December 2025 cannot include the 2026 API rows (r/cars 305, r/Autos 38, r/carbuying 142)."
            )
            lines.append("")
            lines.append("| Subreddit | Year | Dump | API aggregate | Dump minus API |")
            lines.append("|---|---:|---:|---:|---:|")
            for _, row in compared.sort_values(["subreddit", "year"]).iterrows():
                dump_n = "" if pd.isna(row["dump_posts"]) else str(int(row["dump_posts"]))
                api_n = "" if pd.isna(row["api_posts"]) else str(int(row["api_posts"]))
                delta = "" if pd.isna(row["dump_minus_api"]) else str(int(row["dump_minus_api"]))
                lines.append(f"| r/{row['subreddit']} | {int(row['year'])} | {dump_n} | {api_n} | {delta} |")
            lines.append("")
    elif counts_path is None or not counts_path.exists():
        lines.append("No dump extract has been written into this data card yet. The command above is what produces the count table.")
        lines.append("")
    return "\n".join(lines)


def upsert_dump_section(card_path: Path, section: str | None = None, counts_path: Path | None = None, compare_path: Path | None = None, stats: ExtractStats | None = None) -> None:
    start = "<!-- dump-extract-start -->"
    end = "<!-- dump-extract-end -->"
    if section is None:
        section = render_dump_markdown(counts_path=counts_path, compare_path=compare_path, stats=stats)
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    existing = card_path.read_text() if card_path.exists() else ""
    if start in existing and end in existing:
        head, rest = existing.split(start, 1)
        _, tail = rest.split(end, 1)
        card_path.write_text(head.rstrip() + "\n\n" + block + tail.lstrip("\n"))
        return
    card_path.write_text(existing.rstrip() + "\n\n" + block)
