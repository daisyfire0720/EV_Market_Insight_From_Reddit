"""Re-fit BERTopic using pre-saved embeddings — skips encoding entirely.

Use this after run_ev_all_extract.py has been run with --save-embeddings.
Loads the embeddings .npy and the documents CSV, then re-fits the topic model
with whatever UMAP/HDBSCAN parameters you specify.

Example usage:

  # First run (generates embeddings):
  python run_ev_all_extract.py --save-embeddings output/topic_extraction/embeddings.npy

  # Re-fit with different clustering params (no GPU / sentence-transformer needed):
  python run_ev_all_reload.py \\
      --embeddings output/topic_extraction/embeddings.npy \\
      --docs      output/topic_extraction/all_subreddits_documents_topics.csv \\
      --output-dir output/topic_extraction_v2 \\
      --min-cluster-size 50 \\
      --n-neighbors 10
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

from ev_bertopic.topic_extract_pipeline import BERTopicConfig, RedditBERTopicPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-fit BERTopic from saved embeddings without re-encoding.",
        epilog=(
            "Examples:\n"
            "  python run_ev_all_reload.py \\\n"
            "      --embeddings output/topic_extraction/embeddings.npy \\\n"
            "      --docs       output/topic_extraction/all_subreddits_documents_topics.csv\n"
            "  python run_ev_all_reload.py \\\n"
            "      --embeddings output/topic_extraction/embeddings.npy \\\n"
            "      --docs       output/topic_extraction/all_subreddits_documents_topics.csv \\\n"
            "      --output-dir output/topic_extraction_v2 \\\n"
            "      --min-cluster-size 50 --n-neighbors 10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the .npy embeddings file saved by run_ev_all_extract.py --save-embeddings.",
    )
    parser.add_argument(
        "--docs",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the documents CSV saved by run_ev_all_extract.py (all_subreddits_documents_topics.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/topic_extraction",
        help="Directory to write re-fitted output CSV files.",
    )
    # UMAP overrides
    parser.add_argument("--n-neighbors", type=int, default=None, help="Override UMAP n_neighbors.")
    parser.add_argument("--n-components", type=int, default=None, help="Override UMAP n_components.")
    parser.add_argument("--min-dist", type=float, default=None, help="Override UMAP min_dist.")
    # HDBSCAN overrides
    parser.add_argument("--min-cluster-size", type=int, default=None, help="Override HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=None, help="Override HDBSCAN min_samples.")
    # BERTopic overrides
    parser.add_argument("--nr-topics", type=int, default=None, help="Override nr_topics (topic reduction).")
    return parser.parse_args()


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

    # ---- resolve paths
    emb_path = Path(args.embeddings)
    if not emb_path.is_absolute():
        emb_path = PROJECT_ROOT / emb_path

    docs_path = Path(args.docs)
    if not docs_path.is_absolute():
        docs_path = PROJECT_ROOT / docs_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # ---- load documents
    print(f"Loading documents from {docs_path} ...")
    all_docs_df = pd.read_csv(docs_path)
    documents = all_docs_df["text_clean"].fillna("").astype(str).tolist()
    print(f"  Loaded {len(documents):,} documents")

    # ---- build config, applying any CLI overrides
    cfg = BERTopicConfig(
        # Disable embedding cache/device — we won't be encoding anything
        embedding_cache_enabled=False,
        embedding_device=None,
    )
    if args.n_neighbors is not None:
        cfg.umap_n_neighbors = args.n_neighbors
    if args.n_components is not None:
        cfg.umap_n_components = args.n_components
    if args.min_dist is not None:
        cfg.umap_min_dist = args.min_dist
    if args.min_cluster_size is not None:
        cfg.hdbscan_min_cluster_size = args.min_cluster_size
    if args.min_samples is not None:
        cfg.hdbscan_min_samples = args.min_samples
    if args.nr_topics is not None:
        cfg.nr_topics = args.nr_topics

    pipeline = RedditBERTopicPipeline(cfg)

    # ---- load embeddings (single np.load, no cache-check loop)
    embeddings = pipeline.load_embeddings(emb_path)

    if embeddings.shape[0] != len(documents):
        raise ValueError(
            f"Embeddings row count ({embeddings.shape[0]:,}) does not match "
            f"document count ({len(documents):,}). "
            "Make sure --embeddings and --docs were produced from the same run."
        )

    # ---- re-fit (no encoding step)
    topics, probs, _ = pipeline.fit_transform(documents, embeddings=embeddings)

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

    # ---- topic reduction if requested
    if cfg.nr_topics is not None:
        print(f"Reducing topics to nr_topics={cfg.nr_topics} ...")
        pipeline.reduce_topics(documents, nr_topics=cfg.nr_topics)
        all_docs_df["topic"] = pipeline.model.topics_

    # ---- outputs
    topics_df = pipeline.model.get_topic_info()
    topics_df["topic_label_bert"] = generate_topic_labels(pipeline.model, topics_df)

    yearly_cols = [c for c in ["source_tag", "created_year"] if c in all_docs_df.columns]
    if yearly_cols:
        yearly_stats = (
            all_docs_df.groupby(yearly_cols, dropna=False)
            .agg(
                doc_count=("doc_id", "count") if "doc_id" in all_docs_df.columns else ("text_clean", "count"),
                unique_topics=("topic", lambda s: s[s >= 0].nunique()),
                submissions=("is_submission", lambda s: int(s.sum())) if "is_submission" in all_docs_df.columns else ("topic", "count"),
                comments=("is_submission", lambda s: int((~s).sum())) if "is_submission" in all_docs_df.columns else ("topic", "count"),
            )
            .reset_index()
            .sort_values(yearly_cols)
        )
    else:
        yearly_stats = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    stats_out = output_dir / "all_subreddits_yearly_stats.csv"
    topics_out = output_dir / "all_subreddits_topic_info.csv"
    docs_out = output_dir / "all_subreddits_documents_topics.csv"

    if not yearly_stats.empty:
        yearly_stats.to_csv(stats_out, index=False)
    topics_df.to_csv(topics_out, index=False)
    all_docs_df.to_csv(docs_out, index=False)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"\nDocuments: {len(documents):,}")
    print(f"Topics found: {n_topics}")
    print(f"UMAP  n_neighbors={cfg.umap_n_neighbors}, n_components={cfg.umap_n_components}, min_dist={cfg.umap_min_dist}")
    print(f"HDBSCAN  min_cluster_size={cfg.hdbscan_min_cluster_size}, min_samples={cfg.hdbscan_min_samples}")
    if not yearly_stats.empty:
        print(f"Saved yearly stats: {stats_out}")
    print(f"Saved topic info:   {topics_out}")
    print(f"Saved documents:    {docs_out}")


if __name__ == "__main__":
    main()
