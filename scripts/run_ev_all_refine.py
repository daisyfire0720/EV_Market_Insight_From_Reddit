"""Run topic refinement for all-subreddit BERTopic outputs.

Pipeline:
1) Read `all_subreddits_topic_info.csv` from topic extraction output.
2) Apply rule-based topic refinement (`topic_refine_pipeline.py`).
3) Apply LLM refinement (`topic_llm_pipeline.py`).

This script saves both intermediate and final outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_bertopic.topic_llm_pipeline import TopicLLMConfig, TopicLLMPipeline
from ev_bertopic.topic_refine_pipeline import TopicRefinementPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rule-based + LLM topic refinement on all-subreddit BERTopic outputs.",
        epilog=(
            "Examples:\n"
            "  python run_ev_all_refine.py\n"
            "  python run_ev_all_refine.py --extraction-dir output/topic_extraction --refinement-dir output/topic_refinement\n"
            "  python run_ev_all_refine.py --api-key YOUR_GEMINI_KEY\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extraction-dir",
        type=str,
        default="output/topic_extraction",
        help="Directory that contains all_subreddits_topic_info.csv from run_ev_all_extract.py.",
    )
    parser.add_argument(
        "--topic-info-file",
        type=str,
        default="all_subreddits_topic_info.csv",
        help="Topic info CSV filename produced by topic extraction.",
    )
    parser.add_argument(
        "--refinement-dir",
        type=str,
        default="output/topic_refinement",
        help="Directory to save refinement outputs.",
    )
    parser.add_argument(
        "--refined-file",
        type=str,
        default="all_subreddits_topic_labels_refined.csv",
        help="Intermediate CSV output from rule-based refinement.",
    )
    parser.add_argument(
        "--llm-file",
        type=str,
        default="all_subreddits_topic_labels_llm.csv",
        help="Final CSV output from LLM refinement.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Gemini API key. If omitted, SDK/environment defaults are used.",
    )
    parser.add_argument(
        "--call-interval-seconds",
        type=float,
        default=30.0,
        help="Minimum wait between Gemini calls.",
    )
    parser.add_argument(
        "--daily-call-limit",
        type=int,
        default=500,
        help="Maximum Gemini calls allowed per rolling 24h window.",
    )
    parser.add_argument(
        "--call-log-file",
        type=str,
        default="gemini_call_log_all_subreddits.json",
        help="Call log filename saved under refinement-dir.",
    )
    parser.add_argument(
        "--embed-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Embedding device for sentence-transformers in LLM refinement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extraction_dir = Path(args.extraction_dir)
    if not extraction_dir.is_absolute():
        extraction_dir = PROJECT_ROOT / extraction_dir
    topic_info_path = extraction_dir / args.topic_info_file

    if not topic_info_path.exists():
        raise FileNotFoundError(f"Topic info file not found: {topic_info_path}")

    refinement_dir = Path(args.refinement_dir)
    if not refinement_dir.is_absolute():
        refinement_dir = PROJECT_ROOT / refinement_dir
    refinement_dir.mkdir(parents=True, exist_ok=True)

    refined_path = refinement_dir / args.refined_file
    llm_path = refinement_dir / args.llm_file
    call_log_path = refinement_dir / args.call_log_file

    print("[1/3] Running rule-based topic refinement...")
    refine_pipeline = TopicRefinementPipeline.from_csv(topic_info_path)
    refined_df = refine_pipeline.topic_labels_refined()
    refined_df.to_csv(refined_path, index=False)
    print(f"Saved intermediate refined labels: {refined_path}")

    print("[2/3] Running LLM topic refinement...")
    llm_cfg = TopicLLMConfig(
        input_path=str(refined_path),
        output_path=str(llm_path),
        embed_device=args.embed_device,
        gemini_api_key=args.api_key,
        call_interval_seconds=args.call_interval_seconds,
        daily_call_limit=args.daily_call_limit,
        call_log_path=str(call_log_path),
    )
    llm_pipeline = TopicLLMPipeline(llm_cfg)
    llm_pipeline.run()

    print("[3/3] Done")
    print(f"Input topic info: {topic_info_path}")
    print(f"Intermediate refined output: {refined_path}")
    print(f"Final LLM output: {llm_path}")
    print(f"Gemini call log: {call_log_path}")


if __name__ == "__main__":
    main()
