"""Run BERTopic across all EV-related subreddit files in `data_all`.

This script discovers `*_submissions_ev.csv` and `*_comments_ev.csv` pairs,
assigns source tags per subreddit mapping, fits BERTopic, and exports outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ev_bertopic.topic_extraction_pipeline import BERTopicConfig, RedditBERTopicPipeline, RedditDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic pipeline for all EV-related subreddit files in data_all.",
        epilog=(
            "Examples:\n"
            "  python run_ev_all_extract.py\n"
            "  python run_ev_all_extract.py --input-folder data/data_all --output-dir output/topic_extraction\n"
            "  python run_ev_all_extract.py --source-tag electricvehicles=evforum carbuying=carbuying"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="data/data_all",
        help="Folder containing *_submissions_ev.csv and *_comments_ev.csv files.",
    )
    parser.add_argument(
        "--submissions-pattern",
        type=str,
        default="*_submissions_ev.csv",
        help="Glob pattern for submissions files.",
    )
    parser.add_argument(
        "--comments-pattern",
        type=str,
        default="*_comments_ev.csv",
        help="Glob pattern for comments files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/topic_extraction",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--source-tag",
        nargs="*",
        default=["electricvehicles=evforum"],
        help=(
            "Optional key=value mappings of subreddit prefix to source_tag. "
            "Any subreddit without an explicit mapping uses its own prefix as source_tag."
        ),
    )
    return parser.parse_args()


def _prefix_from_stem(stem: str, suffix: str) -> str:
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _parse_source_tag_map(raw_items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --source-tag-map value '{item}'. Expected key=value format."
            )
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid --source-tag-map value '{item}'. Both key and value are required."
            )
        mapping[key] = value
    return mapping


def generate_topic_labels(topic_model, topics_df: pd.DataFrame, top_n_words: int = 8) -> pd.Series:
    if "Topic" not in topics_df.columns:
        return pd.Series([np.nan] * len(topics_df), index=topics_df.index)

    topic_to_label: dict[int, str] = {}
    try:
        generated = topic_model.generate_topic_labels(nr_words=top_n_words, topic_prefix=True)
        for label in generated:
            parts = str(label).split("_", 1)
            if len(parts) == 2:
                try:
                    topic_to_label[int(parts[0])] = parts[1]
                except ValueError:
                    continue
    except Exception:
        topic_to_label = {}

    def _label_for_topic(topic_id: int) -> str:
        if int(topic_id) == -1:
            return "Outlier / Mixed"
        if int(topic_id) in topic_to_label:
            return topic_to_label[int(topic_id)]
        words = topic_model.get_topic(int(topic_id)) or []
        top_words = [w for w, _ in words[:top_n_words]]
        return ", ".join(top_words) if top_words else f"Topic {topic_id}"

    return topics_df["Topic"].apply(_label_for_topic)


def main() -> None:
    args = parse_args()

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.max_columns = None

    cfg = BERTopicConfig()
    builder = RedditDatasetBuilder(cfg)
    pipeline = RedditBERTopicPipeline(cfg)

    source_tag_map = _parse_source_tag_map(args.source_tag)
    input_dir = Path(args.input_folder)

    submission_files = sorted(input_dir.glob(args.submissions_pattern))
    comment_files = sorted(input_dir.glob(args.comments_pattern))

    if not submission_files:
        raise FileNotFoundError(f"No submission files found for pattern: {args.submissions_pattern}")

    comments_by_prefix = {
        _prefix_from_stem(p.stem, "_comments_ev").lower(): p
        for p in comment_files
    }

    canonical_parts: list[pd.DataFrame] = []
    total_submission_rows = 0
    total_comment_rows = 0

    for sub_path in submission_files:
        prefix = _prefix_from_stem(sub_path.stem, "_submissions_ev")
        prefix_key = prefix.lower()
        subreddit = prefix
        source_tag = source_tag_map.get(prefix_key, prefix_key)

        submissions_df = pd.read_csv(sub_path)
        total_submission_rows += len(submissions_df)

        comment_path = comments_by_prefix.get(prefix_key)
        if comment_path is not None and comment_path.exists():
            comments_df = pd.read_csv(comment_path)
        else:
            comments_df = pd.DataFrame(columns=["author", "score", "created", "link", "body"])

        total_comment_rows += len(comments_df)

        canonical_df = builder.build_canonical_df(
            submissions_df=submissions_df,
            comments_df=comments_df,
            subreddit=subreddit,
            source_tag=source_tag,
        )
        canonical_parts.append(canonical_df)

    all_docs_df = pd.concat(canonical_parts, ignore_index=True)
    documents = all_docs_df["text_clean"].fillna("").astype(str).tolist()

    topics, probs, _embeddings = pipeline.fit_transform(documents)
    all_docs_df["topic"] = topics
    if isinstance(probs, np.ndarray):
        if probs.ndim == 2:
            all_docs_df["topic_probability_max"] = probs.max(axis=1)
        elif probs.ndim == 1:
            all_docs_df["topic_probability_max"] = probs
        else:
            all_docs_df["topic_probability_max"] = np.nan
    else:
        all_docs_df["topic_probability_max"] = np.nan

    topics_df = pipeline.model.get_topic_info()
    topics_df["topic_label_bert"] = generate_topic_labels(pipeline.model, topics_df)
    yearly_stats = (
        all_docs_df.groupby(["source_tag", "created_year"], dropna=False)
        .agg(
            doc_count=("doc_id", "count"),
            unique_topics=("topic", lambda s: s[s >= 0].nunique()),
            submissions=("is_submission", lambda s: int(s.sum())),
            comments=("is_submission", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values(["source_tag", "created_year"])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "all_subreddits_yearly_stats.csv"
    topics_path = output_dir / "all_subreddits_topic_info.csv"
    docs_path = output_dir / "all_subreddits_documents_topics.csv"

    yearly_stats.to_csv(stats_path, index=False)
    topics_df.to_csv(topics_path, index=False)
    all_docs_df.to_csv(docs_path, index=False)

    print(f"Subreddit files: {len(submission_files)}")
    print(f"Submission rows: {total_submission_rows}")
    print(f"Comment rows: {total_comment_rows}")
    print(f"Canonical rows: {len(all_docs_df)}")
    print(f"Documents used: {len(documents)}")
    print(f"Source tags: {', '.join(sorted(all_docs_df['source_tag'].dropna().astype(str).unique()))}")
    print(f"Saved yearly stats: {stats_path}")
    print(f"Saved topic info: {topics_path}")
    print(f"Saved documents+topics: {docs_path}")


if __name__ == "__main__":
    main()
