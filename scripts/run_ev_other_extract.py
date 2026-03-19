"""Run BERTopic for EV-related posts from non-EV car subreddits.

This script loads `*_submissions_ev.csv` and `*_comments_ev.csv` files,
builds a canonical document table, fits BERTopic, and exports topic outputs.
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
        description="Run BERTopic pipeline for EV-related submissions + comments from other car subreddits.",
        epilog=(
            "Examples:\n"
            "  python run_ev_other_extract.py\n"
            "  python run_ev_other_extract.py --subreddits carbuying --data-type submissions\n"
            "  python run_ev_other_extract.py --subreddits carbuying, autos --data-type comments\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="data/data_other",
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
            "Which document types to include. 'submissions' loads only submission files, "
            "'comments' loads only comment files, 'all' loads both (default)."
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

    input_dir = Path(args.input_folder)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    target_subreddits = _parse_target_subreddits(args)
    all_submission_files = sorted(input_dir.glob(args.submissions_pattern))
    all_comment_files = sorted(input_dir.glob(args.comments_pattern))

    data_type = args.data_type
    if data_type == "comments":
        all_primary_files = all_comment_files
        primary_suffix = "_comments_ev"
    else:
        all_primary_files = all_submission_files
        primary_suffix = "_submissions_ev"

    if target_subreddits:
        target_set = {s.lower() for s in target_subreddits}
        primary_files = [
            p for p in all_primary_files
            if _prefix_from_stem(p.stem, primary_suffix).lower() in target_set
        ]
        found_prefixes = {
            _prefix_from_stem(p.stem, primary_suffix).lower()
            for p in primary_files
        }
        missing = [name for name in target_subreddits if name.lower() not in found_prefixes]
        if missing:
            missing_list = ", ".join(missing)
            pattern = args.comments_pattern if data_type == "comments" else args.submissions_pattern
            raise FileNotFoundError(
                f"No files matched for subreddit prefix(es): {missing_list} "
                f"using pattern '{pattern}'."
            )
    else:
        primary_files = all_primary_files

    if not primary_files:
        pattern = args.comments_pattern if data_type == "comments" else args.submissions_pattern
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    comments_by_prefix = {
        _prefix_from_stem(p.stem, "_comments_ev"): p
        for p in all_comment_files
    }

    canonical_parts: list[pd.DataFrame] = []
    total_submission_rows = 0
    total_comment_rows = 0

    for primary_path in primary_files:
        prefix = _prefix_from_stem(primary_path.stem, primary_suffix)
        subreddit = prefix

        if data_type == "submissions":
            submissions_df = pd.read_csv(primary_path)
            total_submission_rows += len(submissions_df)
            comments_df = pd.DataFrame(columns=["author", "score", "created", "link", "body"])
        elif data_type == "comments":
            submissions_df = pd.DataFrame(columns=["author", "score", "created", "link", "title", "text"])
            comments_df = pd.read_csv(primary_path)
            total_comment_rows += len(comments_df)
        else:
            submissions_df = pd.read_csv(primary_path)
            total_submission_rows += len(submissions_df)
            comment_path = comments_by_prefix.get(prefix)
            if comment_path is not None and comment_path.exists():
                comments_df = pd.read_csv(comment_path)
            else:
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
    sanity_min_tokens = min(cfg.min_tokens_submission, cfg.min_tokens_comment)
    all_docs_df, sanity_stats = apply_document_sanity_checks(all_docs_df, min_tokens=sanity_min_tokens)
    if all_docs_df.empty:
        raise ValueError("No valid documents remain after sanity checks. Adjust filters or inspect source CSV files.")

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
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "other_subreddits_yearly_stats.csv"
    topics_path = output_dir / "other_subreddits_topic_info.csv"
    docs_path = output_dir / "other_subreddits_documents_topics.csv"

    yearly_stats.to_csv(stats_path, index=False)
    topics_df.to_csv(topics_path, index=False)
    all_docs_df.to_csv(docs_path, index=False)

    print(f"Subreddit files: {len(all_submission_files)}")
    print(f"Data type: {data_type}")
    print(f"Subreddit files: {len(primary_files)}")
    print(f"Submission rows: {total_submission_rows}")
    print(f"Comment rows: {total_comment_rows}")
    print(f"Canonical rows: {len(all_docs_df)}")
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
