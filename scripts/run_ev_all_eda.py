"""Run EDA analysis on BERTopic outputs with LLM topic labels.

This script reads:
1) LLM-refined topic labels CSV
2) Document-topic assignments CSV
3) Optional yearly stats CSV

Then it exports EDA tables and plots to an output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_bertopic.topic_explore_pipeline import RedditTopicEDA
from ev_bertopic.topic_refine_pipeline import TopicRefineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run topic EDA pipeline using final LLM labels.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_ev_all_eda.py\n"
            "  python scripts/run_ev_all_eda.py --show --no-save\n"
            "  python scripts/run_ev_all_eda.py --refinement-dir output/topic_refinement "
            "--extraction-dir output/topic_extraction --output-dir output/topic_exploration"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refinement-dir",
        type=str,
        default="output/topic_refinement",
        help="Directory containing refined/LLM topic label files.",
    )
    parser.add_argument(
        "--extraction-dir",
        type=str,
        default="output/topic_extraction",
        help="Directory containing extraction output files.",
    )
    parser.add_argument(
        "--topic-info-file",
        type=str,
        default="all_subreddits_topic_labels_llm.csv",
        help="LLM-labeled topic info CSV filename.",
    )
    parser.add_argument(
        "--documents-file",
        type=str,
        default="all_subreddits_documents_topics.csv",
        help="Documents + topic assignment CSV filename.",
    )
    parser.add_argument(
        "--yearly-stats-file",
        type=str,
        default="all_subreddits_yearly_stats.csv",
        help="Yearly stats CSV filename. Set to empty string to skip.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/topic_exploration",
        help="Directory to save EDA tables and plots.",
    )
    parser.add_argument(
        "--top-n-overall",
        type=int,
        default=20,
        help="Top topics count for prevalence/heatmap charts.",
    )
    parser.add_argument(
        "--top-n-trend",
        type=int,
        default=10,
        help="Top topics count for trend chart.",
    )
    parser.add_argument(
        "--min-topic-docs-engagement",
        type=int,
        default=20,
        help="Minimum topic document count for engagement metrics.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional minimum topic_probability_max to keep documents.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include topic=-1 outliers in EDA tables/charts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save CSV/PNG outputs; run computations and optional plotting only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    refinement_dir = Path(args.refinement_dir)
    extraction_dir = Path(args.extraction_dir)
    if not refinement_dir.is_absolute():
        refinement_dir = PROJECT_ROOT / refinement_dir
    if not extraction_dir.is_absolute():
        extraction_dir = PROJECT_ROOT / extraction_dir

    topic_info_path = refinement_dir / args.topic_info_file
    documents_path = extraction_dir / args.documents_file
    yearly_stats_path = extraction_dir / args.yearly_stats_file if args.yearly_stats_file else None

    if not topic_info_path.exists():
        raise FileNotFoundError(f"Topic info file not found: {topic_info_path}")
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_path}")
    if yearly_stats_path is not None and not yearly_stats_path.exists():
        raise FileNotFoundError(f"Yearly stats file not found: {yearly_stats_path}")

    cfg = TopicRefineConfig(
        top_n_topics_overall=args.top_n_overall,
        top_n_topics_trend=args.top_n_trend,
        min_topic_docs_for_engagement=args.min_topic_docs_engagement,
        confidence_threshold=args.confidence_threshold,
        exclude_outlier_topic=not args.include_outliers,
    )

    eda = RedditTopicEDA.from_csv(
        topic_info_path=topic_info_path,
        document_topics_path=documents_path,
        yearly_stats_path=yearly_stats_path,
        config=cfg,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    save_outputs = not args.no_save
    save_paths = eda.run_all_eda(outdir=output_dir, show=args.show, save=save_outputs)

    print("EDA run complete.")
    print(f"Topic info input: {topic_info_path}")
    print(f"Documents input: {documents_path}")
    print(f"Yearly stats input: {yearly_stats_path if yearly_stats_path is not None else 'None'}")
    print(f"Save outputs: {save_outputs}")
    if save_outputs:
        print(f"Output directory: {output_dir}")
        for key, path in save_paths.items():
            print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
