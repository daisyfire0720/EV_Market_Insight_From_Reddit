from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import glob
import os
import re
from typing import Iterable, Sequence

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP


@dataclass
class BERTopicConfig:
    language: str = "english"
    calculate_probabilities: bool = True
    verbose: bool = True
    top_n_words: int = 5
    ngram_range: tuple[int, int] = (1, 2)
    embedding_model_name: str = "all-MiniLM-L6-v2"
    umap_n_neighbors: int = 3
    umap_n_components: int = 3
    umap_min_dist: float = 0.05
    hdbscan_min_cluster_size: int = 80
    hdbscan_min_samples: int = 40
    extra_stopwords: Sequence[str] = field(default_factory=lambda: ["http", "https", "amp", "com"])


class RedditBERTopicPipeline:
    def __init__(self, config: BERTopicConfig | None = None) -> None:
        self.config = config or BERTopicConfig()
        self.model: BERTopic | None = None

    @staticmethod
    def load_csv(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
        return pd.read_csv(csv_path)

    @staticmethod
    def load_csvs_from_glob(folder_path: str | os.PathLike[str], pattern: str) -> pd.DataFrame:
        search_pattern = str(Path(folder_path) / pattern)
        csv_files = glob.glob(search_pattern)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for pattern: {search_pattern}")
        dataframes = [pd.read_csv(file_path) for file_path in csv_files]
        return pd.concat(dataframes, ignore_index=True)

    @staticmethod
    def preprocess_submissions(
        df: pd.DataFrame,
        title_col: str = "title",
        text_col: str = "text",
        created_col: str = "created",
        author_col: str | None = None,
        remove_deleted_authors: bool = False,
        normalize_alnum: bool = False,
    ) -> pd.DataFrame:
        output = df.copy()

        if remove_deleted_authors and author_col and author_col in output.columns:
            output = output[output[author_col] != "u/[deleted]"]

        output = output[output[title_col].notnull()].reset_index(drop=True)
        output[title_col] = output[title_col].astype(str)

        text_not_deleted = output[text_col].notnull() & (output[text_col] != "[deleted]")
        output[text_col] = output[text_col].fillna("").astype(str)
        output["text_use"] = output[title_col]
        output.loc[text_not_deleted, "text_use"] = output.loc[text_not_deleted, title_col] + output.loc[text_not_deleted, text_col]

        output["text_use"] = output["text_use"].str.replace(r"\[removed\]", "", regex=True)
        if normalize_alnum:
            output["text_use"] = output["text_use"].apply(lambda value: re.sub(r"[^a-zA-Z0-9\s]", "", value))

        output["text_use"] = output["text_use"].str.strip()
        output["text_use_len"] = output["text_use"].apply(lambda value: len(value.split()))
        output["created_year"] = output[created_col].astype(str).str.split("-", n=1).str[0]
        return output

    @staticmethod
    def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
        data = df[["created_year", "text_use", "text_use_len", "score"]]
        stats = data.groupby("created_year").agg({"text_use": "count", "score": ["max", "median"]})
        return stats

    @staticmethod
    def build_documents(df: pd.DataFrame, min_length: int = 2) -> list[str]:
        documents = df[df["text_use"].apply(lambda value: len(value) >= min_length)]["text_use"]
        return documents.tolist()

    def _build_stopwords(self) -> list[str]:
        words = list(ENGLISH_STOP_WORDS)
        try:
            from nltk.corpus import stopwords

            words.extend(stopwords.words("english"))
        except Exception:
            pass

        words.extend(self.config.extra_stopwords)
        return sorted(set(words))

    def build_model(self, advanced: bool = True) -> BERTopic:
        if advanced:
            embedding_model = SentenceTransformer(self.config.embedding_model_name)
            umap_model = UMAP(
                n_neighbors=self.config.umap_n_neighbors,
                n_components=self.config.umap_n_components,
                min_dist=self.config.umap_min_dist,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                gen_min_span_tree=True,
                prediction_data=True,
            )
            vectorizer_model = CountVectorizer(
                ngram_range=self.config.ngram_range,
                stop_words=self._build_stopwords(),
            )
            self.model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                top_n_words=self.config.top_n_words,
                language=self.config.language,
                calculate_probabilities=self.config.calculate_probabilities,
                verbose=self.config.verbose,
            )
            return self.model

        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
        self.model = BERTopic(
            vectorizer_model=vectorizer_model,
            language=self.config.language,
            calculate_probabilities=self.config.calculate_probabilities,
            verbose=self.config.verbose,
        )
        return self.model

    def fit(self, documents: Iterable[str]) -> tuple[list[int], list[list[float]] | None]:
        if self.model is None:
            self.build_model(advanced=True)

        topics, probs = self.model.fit_transform(list(documents))
        return topics, probs

    def topic_info(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.get_topic_info()

    def topic_words(self, topic_num: int) -> list[tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.get_topic(topic_num)

    def representative_docs(self, topic_num: int) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.model.get_representative_docs(topic_num)

    def save_topic_info(self, output_path: str | os.PathLike[str]) -> Path:
        topic_df = self.topic_info()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        topic_df.to_csv(target, index=False)
        return target
