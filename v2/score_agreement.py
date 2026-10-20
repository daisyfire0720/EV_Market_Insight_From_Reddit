#!/usr/bin/env python3
"""Score two coder sheets against each other and against the model key.

The sheets are the blank annotation file after two people have filled the
code columns. This script does not create those labels. Disagreements are
counted and left out of the gold set.

From the repository root::

    python v2/score_agreement.py \\
        --coder-a coder_a.csv --coder-b coder_b.csv \\
        --model-key v2/annotation/annotation_model_key.csv \\
        --corpus v2/outputs/framing_labels.csv \\
        --out v2/outputs/agreement
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.agreement import BARRIER_FROM_MODEL, score_coders  # noqa: E402
from ev_v2.annotation_sample import prepare_label_frame  # noqa: E402


def corpus_rates(path: Path) -> dict[str, float]:
    frame = prepare_label_frame(pd.read_csv(path))
    rates = {}
    for code in BARRIER_FROM_MODEL.values():
        if code not in frame.columns:
            continue
        rates[code] = float(frame[code].mean()) if len(frame) else float("nan")
    if "none" not in rates:
        barrier_cols = [code for code in BARRIER_FROM_MODEL.values() if code in frame.columns]
        if barrier_cols:
            rates["none"] = float((~frame[barrier_cols].any(axis=1)).mean())
    return rates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inter-coder agreement and model-versus-human scores.")
    parser.add_argument("--coder-a", type=Path, required=True)
    parser.add_argument("--coder-b", type=Path, required=True)
    parser.add_argument("--model-key", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None, help="Labeled corpus used for the apparent prevalence.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.coder_a.is_file() or not args.coder_b.is_file():
        print("Both coder files have to exist. This script does not invent labels.", file=sys.stderr)
        return 2
    key = pd.read_csv(args.model_key) if args.model_key and args.model_key.is_file() else None
    rates = corpus_rates(args.corpus) if args.corpus and args.corpus.is_file() else None
    tables = score_coders(
        pd.read_csv(args.coder_a),
        pd.read_csv(args.coder_b),
        model_key=key,
        corpus_prevalence=rates,
        n_boot=args.n_boot,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        path = args.out / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
