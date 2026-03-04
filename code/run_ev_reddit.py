from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ev_bertopic import BERTopicConfig, RedditBERTopicPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic pipeline for r/electricvehicles submissions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/data_evforum/output/electricvehicles_submissions.csv",
        help="Path to r/electricvehicles submissions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/data_evforum/output/topics",
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

    df = pipeline.load_csv(args.input)
    cleaned = pipeline.preprocess_submissions(df)
    stats = pipeline.yearly_stats(cleaned)
    documents = pipeline.build_documents(cleaned)

    pipeline.build_model(advanced=not args.basic_model)
    pipeline.fit(documents)
    topics_df = pipeline.topic_info()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "electricvehicles_yearly_stats.csv"
    topics_path = output_dir / "electricvehicles_topic_info.csv"

    stats.to_csv(stats_path)
    topics_df.to_csv(topics_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Documents used: {len(documents)}")
    print(f"Saved yearly stats: {stats_path}")
    print(f"Saved topic info: {topics_path}")


if __name__ == "__main__":
    main()
