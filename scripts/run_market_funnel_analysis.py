"""Run market funnel analysis on LLM-refined topic outputs.

This script reads a topic-level CSV (default from output/ev_refinement),
runs stage assignment + summaries, and writes an Excel workbook to
output/market_funnel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_funnel.market_funnel_analyzer import MarketFunnelAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run market funnel analysis from LLM-refined topic labels.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_market_funnel_analysis.py\n"
            "  python scripts/run_market_funnel_analysis.py --input-file all_subreddits_topic_labels_llm.csv\n"
            "  python scripts/run_market_funnel_analysis.py --exclude-outliers --top-n-topics 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output/ev_refinement",
        help="Directory containing market-funnel input CSV.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="all_subreddits_topic_labels_llm.csv",
        help="Input CSV filename with topic-level LLM labels/summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/market_funnel",
        help="Directory to write market funnel outputs.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="market_funnel_analysis.xlsx",
        help="Output Excel filename.",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=3,
        help="Top topics per stage to include in stage insights.",
    )
    parser.add_argument(
        "--exclude-outliers",
        action="store_true",
        help="Exclude outlier topic rows (topic_id == -1) from analysis.",
    )
    parser.add_argument(
        "--topic-weight-col",
        type=str,
        default=None,
        help="Optional numeric column to use as topic weight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve relative paths from project root so the script works from any cwd.
    project_root = PROJECT_ROOT
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    input_path = input_dir / args.input_file
    output_path = output_dir / args.output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    analyzer = MarketFunnelAnalyzer(topic_weight_col=args.topic_weight_col)
    results = analyzer.run_full_analysis(
        csv_path=str(input_path),
        exclude_outliers=args.exclude_outliers,
        top_n_topics=args.top_n_topics,
    )
    analyzer.export_results(str(output_path))

    print("Market funnel analysis complete.")
    print(f"Input CSV: {input_path}")
    print(f"Output workbook: {output_path}")
    print("\n=== Stage Summary ===")
    print(results["stage_summary"].to_string(index=False))
    print("\n=== Stage Insights ===")
    print(results["stage_insights"].to_string(index=False))


if __name__ == "__main__":
    main()
