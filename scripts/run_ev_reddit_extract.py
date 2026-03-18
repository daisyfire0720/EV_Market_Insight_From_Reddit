"""Run BERTopic for the r/electricvehicles dataset.

This script reads submission and comment CSV files for one subreddit,
creates canonical documents, fits BERTopic, and writes analysis outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ev_bertopic.topic_extract_pipeline import BERTopicConfig, RedditBERTopicPipeline, RedditDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic pipeline for r/electricvehicles submissions + comments.",
        epilog=(
            "Examples:\n"
            "  python run_ev_reddit_extract.py\n"
            "  python run_ev_reddit_extract.py --submissions data/data_evforum/electricvehicles_submissions.csv "
            "--comments data/data_evforum/electricvehicles_comments.csv --output-dir output/topic_extraction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--submissions",
        type=str,
        default="data/data_evforum/electricvehicles_submissions.csv",
        help="Path to r/electricvehicles submissions CSV.",
    )
    parser.add_argument(
        "--comments",
        type=str,
        default="data/data_evforum/electricvehicles_comments.csv",
        help="Path to r/electricvehicles comments CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/topic_extraction",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        default="electricvehicles",
        help="Subreddit label to store in canonical output.",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="evforum",
        help="Source label to store in canonical output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Embedding device for sentence-transformers.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="all",
        choices=["all", "submissions", "comments"],
        help=(
            "Which document types to include. 'submissions' loads only the submissions file, "
            "'comments' loads only the comments file, 'all' loads both (default)."
        ),
    )
    return parser.parse_args()


def generate_topic_labels(topic_model, topics_df: pd.DataFrame, top_n_words: int = 8) -> pd.Series:
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


def apply_document_sanity_checks(all_docs_df: pd.DataFrame, min_tokens: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows that are invalid for topic modeling and report removal counts."""
    sanity_stats = {
        "input_rows": int(len(all_docs_df)),
        "removed_empty_or_marker": 0,
        "removed_too_short": 0,
        "removed_duplicate_doc_id": 0,
        "output_rows": int(len(all_docs_df)),
    }

    if all_docs_df.empty:
        return all_docs_df, sanity_stats

    text = all_docs_df["text_clean"].fillna("").astype(str).str.strip()
    lowered = text.str.lower()
    marker_values = {"", "nan", "none", "deleted", "removed", "[deleted]", "[removed]"}

    mask_empty_or_marker = lowered.isin(marker_values)
    token_counts = text.str.split().str.len().fillna(0).astype(int)
    mask_too_short = token_counts < int(min_tokens)

    sanity_stats["removed_empty_or_marker"] = int(mask_empty_or_marker.sum())
    sanity_stats["removed_too_short"] = int((~mask_empty_or_marker & mask_too_short).sum())

    keep_mask = ~(mask_empty_or_marker | mask_too_short)
    cleaned = all_docs_df.loc[keep_mask].copy()

    if "doc_id" in cleaned.columns:
        dup_mask = cleaned.duplicated(subset=["doc_id"], keep="first")
        sanity_stats["removed_duplicate_doc_id"] = int(dup_mask.sum())
        cleaned = cleaned.loc[~dup_mask].copy()

    sanity_stats["output_rows"] = int(len(cleaned))
    return cleaned.reset_index(drop=True), sanity_stats


def main() -> None:
    args = parse_args()

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.max_columns = None

    cfg = BERTopicConfig(embedding_device=args.device)
    builder = RedditDatasetBuilder(cfg)
    pipeline = RedditBERTopicPipeline(cfg)

    submissions_path = Path(args.submissions)
    comments_path = Path(args.comments)
    if not submissions_path.is_absolute():
        submissions_path = PROJECT_ROOT / submissions_path
    if not comments_path.is_absolute():
        comments_path = PROJECT_ROOT / comments_path

    submissions_df = pd.read_csv(submissions_path)
    data_type = args.data_type
    if data_type == "submissions":
        submissions_df = pd.read_csv(submissions_path)
        comments_df = pd.DataFrame(columns=["author", "score", "created", "link", "body"])
    elif data_type == "comments":
        submissions_df = pd.DataFrame(columns=["author", "score", "created", "link", "title", "text"])
        comments_df = pd.read_csv(comments_path)
    else:
        submissions_df = pd.read_csv(submissions_path)
        comments_df = pd.read_csv(comments_path)

    canonical_df = builder.build_canonical_df(
        submissions_df=submissions_df,
        comments_df=comments_df,
        subreddit=args.subreddit,
        source_tag=args.source_tag,
    )

    canonical_df, sanity_stats = apply_document_sanity_checks(canonical_df, min_tokens=cfg.min_tokens)
    if canonical_df.empty:
        raise ValueError("No valid documents remain after sanity checks. Adjust filters or inspect source CSV files.")

    documents = canonical_df["text_clean"].fillna("").astype(str).tolist()

    topics, probs, _embeddings = pipeline.fit_transform(documents)
    canonical_df["topic"] = topics
    if isinstance(probs, np.ndarray):
        if probs.ndim == 2:
            canonical_df["topic_probability_max"] = probs.max(axis=1)
        elif probs.ndim == 1:
            canonical_df["topic_probability_max"] = probs
        else:
            canonical_df["topic_probability_max"] = np.nan
    else:
        canonical_df["topic_probability_max"] = np.nan

    topics_df = pipeline.model.get_topic_info()
    topics_df["topic_label_bert"] = generate_topic_labels(pipeline.model, topics_df)
    yearly_stats = (
        canonical_df.groupby("created_year", dropna=False)
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
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "electricvehicles_yearly_stats.csv"
    topics_path = output_dir / "electricvehicles_topic_info.csv"
    docs_path = output_dir / "electricvehicles_documents_topics.csv"

    yearly_stats.to_csv(stats_path, index=False)
    topics_df.to_csv(topics_path, index=False)
    canonical_df.to_csv(docs_path, index=False)

    print(f"Submission rows: {len(submissions_df)}")
    print(f"Data type: {data_type}")
    print(f"Comment rows: {len(comments_df)}")
    print(f"Canonical rows: {len(canonical_df)}")
    print(
        "Sanity checks removed: "
        f"empty/markers={sanity_stats['removed_empty_or_marker']}, "
        f"too_short={sanity_stats['removed_too_short']}, "
        f"duplicate_doc_id={sanity_stats['removed_duplicate_doc_id']}"
    )
    print(f"Documents used: {len(documents)}")
    print(f"Embedding device: {cfg.embedding_device}")
    print(f"Saved yearly stats: {stats_path}")
    print(f"Saved topic info: {topics_path}")
    print(f"Saved documents+topics: {docs_path}")


if __name__ == "__main__":
    main()
