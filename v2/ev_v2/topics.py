"""Classical topic model with a probability for every document.

v1 BERTopic ran with ``calculate_probabilities=False``, so
``topic_probability_max`` was always missing. LDA (the model Ruan and Lv
used) returns a distribution over topics for every document. That vector
is what ``topic_probability_max`` is taken from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class LdaResult:
    model: LatentDirichletAllocation
    vectorizer: CountVectorizer
    doc_topic: np.ndarray
    topic_ids: np.ndarray
    topic_probability_max: np.ndarray
    topic_words: pd.DataFrame

    def document_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "topic": self.topic_ids.astype(int),
                "topic_probability_max": self.topic_probability_max.astype(float),
            }
        )


def fit_lda(
    texts: list[str],
    n_topics: int = 8,
    random_state: int = 42,
    max_features: int = 8000,
    min_df: int = 3,
    max_df: float = 0.9,
) -> LdaResult:
    if len(texts) < n_topics:
        raise ValueError(f"need at least {n_topics} documents to fit {n_topics} topics, got {len(texts)}")
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] < n_topics:
        raise ValueError(
            f"vocabulary has {matrix.shape[1]} terms, which is fewer than n_topics={n_topics}"
        )
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=25,
        evaluate_every=0,
    )
    doc_topic = model.fit_transform(matrix)
    # Numerical noise can push a row slightly off 1. Renormalize for the
    # stored probability so the maximum is a share of that document.
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    doc_topic = doc_topic / row_sums
    topic_ids = doc_topic.argmax(axis=1)
    topic_probability_max = doc_topic.max(axis=1)
    if np.isnan(topic_probability_max).any():
        raise RuntimeError("LDA produced NaN document probabilities")
    topic_words = _top_words(model, vectorizer, n_words=12)
    return LdaResult(
        model=model,
        vectorizer=vectorizer,
        doc_topic=doc_topic,
        topic_ids=topic_ids,
        topic_probability_max=topic_probability_max,
        topic_words=topic_words,
    )


def _top_words(model: LatentDirichletAllocation, vectorizer: CountVectorizer, n_words: int) -> pd.DataFrame:
    names = np.array(vectorizer.get_feature_names_out())
    rows = []
    for topic_id, weights in enumerate(model.components_):
        order = weights.argsort()[::-1][:n_words]
        rows.append(
            {
                "topic": topic_id,
                "top_words": ", ".join(names[order]),
                "topic_mass": float(weights.sum()),
            }
        )
    return pd.DataFrame(rows)
