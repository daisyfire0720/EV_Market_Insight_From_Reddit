"""Normalize topic tables so LLM outputs and BERTopic outputs share columns.

``TopicLLMPipeline`` writes ``topic_id`` and ``doc_count``.
``RedditTopicEDA._attach_topic_info`` reads ``Topic`` and ``Count`` and
raises ``KeyError: 'Topic'`` on the LLM file. ``normalize_topic_info``
accepts either schema.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# Default locations. The v1 funnel runner defaulted to ``output/ev_refinement``
# while the refine step writes ``output/topic_refinement``.
DEFAULT_REFINEMENT_DIR = "output/topic_refinement"
DEFAULT_LLM_TOPIC_FILE = "all_subreddits_topic_labels_llm.csv"
DEFAULT_EXTRACTION_DIR = "output/topic_extraction"


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_topic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``Topic`` and ``Count`` populated when possible.

    Existing ``Topic`` / ``Count`` values are kept. Missing ones are filled
    from ``topic_id`` / ``doc_count`` (and a few other aliases).
    """
    out = df.copy()
    topic_src = _first_present(out, ["Topic", "topic_id", "topic"])
    count_src = _first_present(out, ["Count", "doc_count", "count", "topic_size"])
    if topic_src is None:
        raise KeyError(
            "topic table needs a Topic column (accepted aliases: topic_id, topic)"
        )
    if topic_src != "Topic":
        out["Topic"] = out[topic_src]
    if count_src is not None and count_src != "Count":
        out["Count"] = out[count_src]
    elif "Count" not in out.columns:
        out["Count"] = pd.NA
    out["Topic"] = pd.to_numeric(out["Topic"], errors="coerce")
    out["Count"] = pd.to_numeric(out["Count"], errors="coerce")
    return out


def attach_topic_info(docs: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
    """Left-join documents to topic metadata, accepting either column schema."""
    if "topic" not in docs.columns and "Topic" in docs.columns:
        docs = docs.copy()
        docs["topic"] = docs["Topic"]
    if "topic" not in docs.columns:
        raise KeyError("document table needs a topic column")
    info = normalize_topic_info(topic_info)
    keep = [c for c in ["Topic", "Count", "Name", "topic_label_llm", "topic_keywords_clean", "topic_keybert_clean"] if c in info.columns]
    merged = docs.merge(info[keep], left_on="topic", right_on="Topic", how="left")
    return merged
