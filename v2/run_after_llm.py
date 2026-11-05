#!/usr/bin/env python3
"""Rebuild the funnel, discontinuity, and transition tables from an LLM label file.

The file is the one ``llm_label.py`` writes. This script does not call a model
and does not invent rows. Human agreement runs only when both coder sheets
are given.

From ``v2/``::

    python run_after_llm.py --llm outputs/llm_labels.csv --out outputs/after_llm
    python run_after_llm.py --llm outputs/llm_smoke.csv --out outputs/smoke_tables --n-boot 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.after_llm import run_tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tables from an instruction-model label file.")
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--small", type=Path, default=ROOT / "outputs" / "framing_labels.csv")
    parser.add_argument("--authors", type=Path, default=ROOT / "outputs" / "post_author_hashes.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coder-a", type=Path, default=None)
    parser.add_argument("--coder-b", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--scores", type=Path, default=ROOT / "outputs" / "post_scores.csv")
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument(
        "--comment-meta",
        type=Path,
        default=ROOT / "outputs" / "comment_meta.csv",
        help="Comment subreddit, parent link, and time. Used only when the label file has comments.",
    )
    parser.add_argument(
        "--comments-rebalanced",
        action="store_true",
        help="Set only after comment windows are rebuilt with equal counts on each side. Batch 1 is not rebalanced.",
    )
    parser.add_argument(
        "--comment-n-boot",
        type=int,
        default=0,
        help="Week-bootstrap draws for the rebalanced comment RDiT. Zero keeps point estimates only.",
    )
    args = parser.parse_args(argv)
    if not args.llm.is_file():
        print(f"No label file at {args.llm}. This script does not invent one.", file=sys.stderr)
        return 2
    score_path = args.scores if args.scores.is_file() else None
    tables = run_tables(
        args.llm,
        args.small,
        args.authors,
        args.out,
        args.coder_a,
        args.coder_b,
        n_boot=args.n_boot,
        score_path=score_path,
        min_score=args.min_score,
        comment_meta_path=args.comment_meta,
        comments_rebalanced=args.comments_rebalanced,
        comment_n_boot=args.comment_n_boot,
    )
    validity = tables["validity"].iloc[0]
    print(
        f"used {int(validity['n_used'])} of {int(validity['n_rows'])} rows "
        f"(bad json {int(validity['n_bad_json'])}, unknown stage {int(validity['n_unknown_stage'])})",
        flush=True,
    )
    for row in tables["drop_log"].itertuples(index=False):
        print(f"  {row.reason}: {int(row.n)}", flush=True)
    usable = tables["comments_sensitivity"].set_index("item").loc["rdit_usable", "value"]
    print(f"comments RDiT usable: {usable}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
