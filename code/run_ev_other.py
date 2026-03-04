from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ev_bertopic import BERTopicConfig, RedditBERTopicPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic pipeline for EV-related submissions from other car subreddits.")
    parser.add_argument(
        "--input-folder",
        type=str,
        default="../data/data_other",
        help="Folder containing *_submissions_ev.csv files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_submissions_ev.csv",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/data_other/output/topics",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--basic-model",
        action="store_true",
        help="Use basic BERTopic model instead of embedding+UMAP+HDBSCAN configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.max_columns = None

    pipeline = RedditBERTopicPipeline(BERTopicConfig())

    df = pipeline.load_csvs_from_glob(args.input_folder, args.pattern)
    cleaned = pipeline.preprocess_submissions(
        df,
        author_col="author",
        remove_deleted_authors=True,
        normalize_alnum=True,
    )
    stats = pipeline.yearly_stats(cleaned)
    documents = pipeline.build_documents(cleaned)

    pipeline.build_model(advanced=not args.basic_model)
    pipeline.fit(documents)
    topics_df = pipeline.topic_info()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "other_subreddits_yearly_stats.csv"
    topics_path = output_dir / "other_subreddits_topic_info.csv"

    stats.to_csv(stats_path)
    topics_df.to_csv(topics_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Documents used: {len(documents)}")
    print(f"Saved yearly stats: {stats_path}")
    print(f"Saved topic info: {topics_path}")


if __name__ == "__main__":
    main()
