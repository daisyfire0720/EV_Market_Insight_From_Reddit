#!/usr/bin/env python3
"""Filter car-shopper EV posts out of Pushshift / Arctic Shift .zst dumps.

The script does not download torrents. It streams local files. See the
"Shopper text from the dumps" section of ``v2/data_card/DATA_CARD.md`` for
the infohash, the file lengths, and the licensing notes.

Example, from the repository root::

    python v2/extract_from_dumps.py \\
        /path/to/subreddits25/Autos_submissions.zst \\
        /path/to/subreddits25/carbuying_submissions.zst \\
        /path/to/subreddits25/cars_submissions.zst

Monthly ``RS_YYYY-MM.zst`` files are the same command. They are filtered
to r/cars, r/Autos, and r/carbuying.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.dumps import (  # noqa: E402
    compare_counts,
    counts_frame,
    extract_paths,
    upsert_dump_section,
)
from ev_v2.mainstream import MAINSTREAM_END, MAINSTREAM_START  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "raw" / "mainstream_from_dumps.csv"
DEFAULT_COUNTS = ROOT / "data_card" / "dump_keyword_counts.csv"
DEFAULT_COMPARE = ROOT / "data_card" / "dump_vs_api_counts.csv"
DEFAULT_API = ROOT / "data_card" / "mainstream_yearly_counts.csv"
DEFAULT_CARD = ROOT / "data_card" / "DATA_CARD.md"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract EV-keyword shopper submissions from .zst dumps.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Per-subreddit or monthly .zst files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV in the posts_downloaded.csv schema")
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--compare", type=Path, default=DEFAULT_COMPARE)
    parser.add_argument("--api-counts", type=Path, default=DEFAULT_API)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--start", type=_parse_date, default=MAINSTREAM_START)
    parser.add_argument("--end", type=_parse_date, default=MAINSTREAM_END, help="Exclusive end date")
    parser.add_argument("--no-card", action="store_true", help="Do not edit DATA_CARD.md")
    parser.add_argument(
        "--fast-subreddit-scan",
        action="store_true",
        help="Skip JSON parsing on lines that are not r/cars, r/Autos, or r/carbuying. For monthly RS files.",
    )
    args = parser.parse_args(argv)

    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        print("missing files:", ", ".join(missing), file=sys.stderr)
        return 2

    stats = extract_paths(
        list(args.inputs),
        args.out,
        start=args.start,
        end=args.end,
        fast_subreddit_scan=args.fast_subreddit_scan,
    )
    counts = counts_frame(stats)
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.counts, index=False)

    compared = None
    if args.api_counts.is_file():
        import pandas as pd

        compared = compare_counts(counts, pd.read_csv(args.api_counts))
        compared.to_csv(args.compare, index=False)

    for warning in stats.size_warnings:
        print("warning:", warning, flush=True)
    print(
        f"scanned {stats.rows_scanned} lines in {stats.files} file(s); kept {stats.rows_kept}; "
        f"bad json {stats.bad_json}; duplicates {stats.skipped_duplicate}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.counts}", flush=True)
    if compared is not None:
        print(f"wrote {args.compare}", flush=True)
    if not args.no_card and args.card.exists():
        upsert_dump_section(args.card, counts_path=args.counts, compare_path=args.compare if compared is not None else None, stats=stats)
        print(f"updated {args.card}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
