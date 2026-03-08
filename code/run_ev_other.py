"""Run BERTopic for EV-related posts from non-EV car subreddits.

This script loads `*_submissions_ev.csv` and `*_comments_ev.csv` files,
builds a canonical document table, fits BERTopic, and exports topic outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ev_bertopic.pipeline import BERTopicConfig, RedditBERTopicPipeline, RedditDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic pipeline for EV-related submissions + comments from other car subreddits.",
        epilog=(
            "Examples:\n"
            "  python run_ev_other.py\n"
            "  python run_ev_other.py --subreddits carbuying\n"
            "  python run_ev_other.py --subreddits carbuying, autos --output-dir ../data/data_other/output/topics"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="../data/data_other",
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
        default="../data/data_other/output/topics",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="other_subreddits",
        help="Source label to store in canonical output.",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help=(
            "Subreddit prefixes. Accepts comma-separated and/or space-separated values, "
            "e.g. carbuying,autos or carbuying, Autos"
        ),
    )
    return parser.parse_args()


def _prefix_from_stem(stem: str, suffix: str) -> str:
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _parse_target_subreddits(args: argparse.Namespace) -> list[str]:
    targets: list[str] = []

    if args.subreddits:
        raw_values = [args.subreddits] if isinstance(args.subreddits, str) else list(args.subreddits)
        for raw in raw_values:
            for value in str(raw).split(","):
                cleaned = value.strip()
                if cleaned:
                    targets.append(cleaned)

    # Preserve input order while removing duplicates.
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for name in targets:
        if name not in seen:
            seen.add(name)
            ordered_unique.append(name)

    return ordered_unique


def generate_topic_labels(topic_model, topics_df: pd.DataFrame, top_n_words: int = 6) -> pd.Series:
    """Generate labels from BERTopic and align them to topics_df by topic id."""
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

    input_dir = Path(args.input_folder)
    target_subreddits = _parse_target_subreddits(args)
    all_submission_files = sorted(input_dir.glob(args.submissions_pattern))

    if target_subreddits:
        target_set = {s.lower() for s in target_subreddits}
        submission_files = [
            p for p in all_submission_files
            if _prefix_from_stem(p.stem, "_submissions_ev").lower() in target_set
        ]

        found_prefixes = {
            _prefix_from_stem(p.stem, "_submissions_ev").lower()
            for p in submission_files
        }
        missing = [name for name in target_subreddits if name.lower() not in found_prefixes]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(
                f"No submission files matched for subreddit prefix(es): {missing_list} "
                f"using pattern '{args.submissions_pattern}'."
            )
    else:
        submission_files = all_submission_files

    comment_files = sorted(input_dir.glob(args.comments_pattern))

    if not submission_files:
        raise FileNotFoundError(f"No submission files found for pattern: {args.submissions_pattern}")

    comments_by_prefix = {
        _prefix_from_stem(p.stem, "_comments_ev"): p
        for p in comment_files
    }

    canonical_parts: list[pd.DataFrame] = []
    total_submission_rows = 0
    total_comment_rows = 0

    for sub_path in submission_files:
        prefix = _prefix_from_stem(sub_path.stem, "_submissions_ev")
        subreddit = prefix

        submissions_df = pd.read_csv(sub_path)
        total_submission_rows += len(submissions_df)

        comment_path = comments_by_prefix.get(prefix)
        if comment_path is not None and comment_path.exists():
            comments_df = pd.read_csv(comment_path)
        else:
            # Keep pipeline behavior consistent even if comments are missing for one subreddit.
            comments_df = pd.DataFrame(columns=["author", "score", "created", "link", "body"])

        total_comment_rows += len(comments_df)

        canonical_df = builder.build_canonical_df(
            submissions_df=submissions_df,
            comments_df=comments_df,
            subreddit=subreddit,
            source_tag=args.source_tag,
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
    topics_df["topic_label_generated"] = generate_topic_labels(pipeline.model, topics_df)
    yearly_stats = (
        all_docs_df.groupby("created_year", dropna=False)
        .agg(
            doc_count=("doc_id", "count"),
            unique_topics=("topic", lambda s: s[s >= 0].nunique()),
            submissions=("is_submission", lambda s: int(s.sum())),
            comments=("is_submission", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values("created_year")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "other_subreddits_yearly_stats.csv"
    topics_path = output_dir / "other_subreddits_topic_info.csv"
    docs_path = output_dir / "other_subreddits_documents_topics.csv"

    yearly_stats.to_csv(stats_path, index=False)
    topics_df.to_csv(topics_path, index=False)
    all_docs_df.to_csv(docs_path, index=False)

    print(f"Subreddit files: {len(submission_files)}")
    print(f"Submission rows: {total_submission_rows}")
    print(f"Comment rows: {total_comment_rows}")
    print(f"Canonical rows: {len(all_docs_df)}")
    print(f"Documents used: {len(documents)}")
    print(f"Saved yearly stats: {stats_path}")
    print(f"Saved topic info: {topics_path}")
    print(f"Saved documents+topics: {docs_path}")


if __name__ == "__main__":
    main()
