from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import html
import json
import hashlib
import inspect
from typing import Optional, Sequence, Dict, Any, List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


# -----------------------------
# Config
# -----------------------------
@dataclass
class BERTopicConfig:
    # General
    language: str = "english"
    calculate_probabilities: bool = True
    verbose: bool = True
    top_n_words: int = 10
    random_state: int = 42

    # Vectorizer / text
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 5
    max_df: float = 0.95
    max_features: Optional[int] = None
    keep_negations: bool = True
    extra_stopwords: Tuple[str, ...] = (
        # web/common artifacts
        "http", "https", "www", "com", "amp", "html",
        # reddit artifacts
        "reddit", "subreddit", "upvote", "downvote",
        "deleted", "removed", "post", "comment", "thread",
        "op", "edit", "tldr", "nbsp", "x200b",
    )

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: Optional[str] = "cuda"   # "cuda", "cpu", or None
    embedding_batch_size: int = 128
    embedding_show_progress: bool = True

    # Embedding cache
    embedding_cache_dir: str = "cache/embeddings"
    embedding_cache_enabled: bool = True
    embedding_cache_version: str = "v1"

    # UMAP
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # HDBSCAN
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    hdbscan_metric: str = "euclidean"

    # Optional BERTopic post-processing
    nr_topics: Optional[int] = None

    # BERTopic extras
    # Backward-compatible override for calculate_probabilities.
    # If set to None, calculate_probabilities is used.
    enable_probabilities: Optional[bool] = None

    # Dataset cleaning / filtering
    min_tokens: int = 3
    drop_automoderator: bool = True
    drop_probable_bots: bool = True
    # Match common bot-style account names (e.g., remindmebot, x_bot).
    bot_author_regex: str = r"(?:^|\b)(?:auto\w*|\w*bot\w*)(?:$|\b)"
    drop_bot_phrases: bool = True
    bot_phrases: Tuple[str, ...] = (
        "i am a bot",
        "this action was performed automatically",
        "please contact the moderators",
        "bot here",
    )

    # Deduplication policy
    dedup_subset: Tuple[str, ...] = ("subreddit", "is_submission", "text_clean")
    dedup_on: Optional[str] = None
    dedup_keep: str = "first"

# -----------------------------
# Utilities
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MD_CODEBLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_MD_QUOTE_RE = re.compile(r"(^|\n)>\s+.*", flags=re.MULTILINE)
_USER_RE = re.compile(r"\bu/[A-Za-z0-9_-]+\b")
_SUB_RE = re.compile(r"\br/[A-Za-z0-9_-]+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt);")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")

_UNICODE_PUNCT_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
})

_DEFAULT_NEGATIONS = {"no", "nor", "not", "never", "without", "n't"}


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Robust datetime parsing:
    - numeric: auto-detect epoch seconds vs milliseconds
    - string: pd.to_datetime with coercion
    """
    s = series.copy()
    # Try numeric conversion
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.5:
        # Heuristic: ms epoch values are ~1e12+, seconds ~1e9+
        median = float(s_num.dropna().median())
        unit = "ms" if median > 1e11 else "s"
        return pd.to_datetime(s_num, unit=unit, errors="coerce", utc=True).dt.tz_convert(None)
    # Fallback to string datetime
    return pd.to_datetime(s, errors="coerce", utc=False)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _repair_common_mojibake(text: str) -> str:
    # Repair common UTF-8 decoded as Latin-1 artifacts (e.g., "â€™", "Ã").
    if not any(marker in text for marker in ("\u00e2", "\u00c3", "\u00c2")):
        return text
    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        return repaired or text
    except Exception:
        return text


def clean_reddit_text(text: str) -> str:
    """
    Reddit-aware text cleaning:
    - remove urls
    - remove markdown code blocks
    - remove quote blocks (lines starting with ">")
    - normalize markdown links [label](url) -> label
    - strip reddit handles r/... and u/...
    - basic html entity normalization
    """
    if text is None:
        return ""
    t = str(text)

    # Normalize common mojibake and unicode punctuation artifacts.
    t = _repair_common_mojibake(t)
    t = html.unescape(t)
    t = t.replace("\\'", "'").replace('\\"', '"')
    t = t.translate(_UNICODE_PUNCT_TRANSLATION)
    # Preserve contractions like I`m / they`re before removing markdown ticks.
    t = re.sub(r"(?<=\w)`(?=\w)", "'", t)
    t = _ZERO_WIDTH_RE.sub(" ", t)

    t = _MD_CODEBLOCK_RE.sub(" ", t)
    t = _MD_QUOTE_RE.sub(" ", t)
    t = _MARKDOWN_LINK_RE.sub(r"\1", t)
    t = _URL_RE.sub(" ", t)
    t = _USER_RE.sub(" ", t)
    t = _SUB_RE.sub(" ", t)
    t = _HTML_ENTITY_RE.sub(" ", t)

    # Remove stray markdown tokens
    t = t.replace("*", " ").replace("_", " ").replace("`", " ")
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def default_stopwords(cfg: BERTopicConfig) -> set[str]:
    sw = set(ENGLISH_STOP_WORDS)
    sw.update({w.lower() for w in cfg.extra_stopwords})
    if cfg.keep_negations:
        sw = sw.difference(_DEFAULT_NEGATIONS)
    return sw


def _patch_sklearn_check_array_compat() -> None:
    """
    Compatibility shim for environments where scikit-learn renamed
    check_array(force_all_finite=...) to ensure_all_finite=...
    but hdbscan still calls the old keyword.
    """
    try:
        from sklearn.utils import validation as sk_validation
        import sklearn.utils as sk_utils
    except Exception:
        return

    try:
        params = inspect.signature(sk_validation.check_array).parameters
    except (TypeError, ValueError):
        return

    if "force_all_finite" in params:
        return
    if "ensure_all_finite" not in params:
        return

    original_check_array = sk_validation.check_array

    def check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return original_check_array(*args, **kwargs)

    sk_validation.check_array = check_array_compat
    sk_utils.check_array = check_array_compat


def _patch_sentence_transformers_static_embedding_compat() -> None:
    """
    Compatibility shim for BERTopic versions that import
    sentence_transformers.models.StaticEmbedding when running against
    older sentence-transformers releases where that class is absent.
    """
    try:
        import sentence_transformers.models as st_models
    except Exception:
        return

    if hasattr(st_models, "StaticEmbedding"):
        return

    class StaticEmbedding:  # pragma: no cover - compatibility shim
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "StaticEmbedding is unavailable in this sentence-transformers version. "
                "Upgrade sentence-transformers to use static embeddings."
            )

    st_models.StaticEmbedding = StaticEmbedding


# -----------------------------
# Dataset builder
# -----------------------------
class RedditDatasetBuilder:
    def __init__(self, cfg: BERTopicConfig):
        self.cfg = cfg
        self._bot_author_re = re.compile(cfg.bot_author_regex, flags=re.IGNORECASE)

    def _is_probable_bot(self, author: str) -> bool:
        if author is None:
            return False
        a = str(author).strip().lower()
        if self.cfg.drop_automoderator and a == "automoderator":
            return True
        if self.cfg.drop_probable_bots and self._bot_author_re.search(a):
            return True
        return False

    def _contains_bot_phrase(self, text: str) -> bool:
        if not self.cfg.drop_bot_phrases:
            return False
        t = (text or "").lower()
        return any(p in t for p in self.cfg.bot_phrases)

    @staticmethod
    def _token_count(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def build_canonical_df(
        self,
        submissions_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        subreddit: str,
        source_tag: str,
        # Submissions schema
        sub_author_col: str = "author",
        sub_title_col: str = "title",
        sub_selftext_col: str = "text",   # your sample uses "text"
        sub_created_col: str = "created",
        sub_score_col: str = "score",
        sub_link_col: str = "link",
        sub_url_col: str = "url",
        # Comments schema
        com_author_col: str = "author",
        com_body_col: str = "body",
        com_created_col: str = "created",
        com_score_col: str = "score",
        com_link_col: str = "link",
    ) -> pd.DataFrame:
        cfg = self.cfg

        # ---- submissions
        s = submissions_df.copy()
        s["is_submission"] = True
        s["subreddit"] = subreddit
        s["source_tag"] = source_tag
        s["author"] = s.get(sub_author_col)
        s["title"] = s.get(sub_title_col)
        s["selftext"] = s.get(sub_selftext_col)
        s["created_dt"] = _safe_to_datetime(s.get(sub_created_col))
        s["score"] = pd.to_numeric(s.get(sub_score_col), errors="coerce")
        s["link"] = s.get(sub_link_col)
        s["url"] = s.get(sub_url_col)

        # Build raw text: title + selftext if exists and not [deleted]/[removed]
        title = s["title"].fillna("").astype(str)
        body = s["selftext"].fillna("").astype(str)

        # common deleted markers
        bad_markers = {"[deleted]", "[removed]", "nan", "none"}
        body_norm = body.str.strip().str.lower()
        body = body.where(~body_norm.isin(bad_markers), "")

        s["text_raw"] = (title.str.strip() + " " + body.str.strip()).str.strip()
        s["text_raw"] = s["text_raw"].fillna("")

        # ---- comments
        c = comments_df.copy()
        c["is_submission"] = False
        c["subreddit"] = subreddit
        c["source_tag"] = source_tag
        c["author"] = c.get(com_author_col)
        c["title"] = np.nan
        c["selftext"] = np.nan
        c["created_dt"] = _safe_to_datetime(c.get(com_created_col))
        c["score"] = pd.to_numeric(c.get(com_score_col), errors="coerce")
        c["link"] = c.get(com_link_col)
        c["url"] = np.nan
        c["text_raw"] = c.get(com_body_col).fillna("").astype(str)

        # ---- unify
        df = pd.concat([s, c], ignore_index=True, sort=False)

        # Author flags
        df["author"] = df["author"].astype(str).fillna("")
        df["author_deleted"] = df["author"].str.strip().str.lower().isin({"[deleted]", "deleted", "nan", "none", ""})

        # Clean text
        df["text_clean"] = df["text_raw"].apply(clean_reddit_text)

        # Basic token filter
        df["n_tokens"] = df["text_clean"].apply(self._token_count)
        df = df[df["n_tokens"] >= cfg.min_tokens].copy()

        # Created time derived fields
        df["created_year"] = df["created_dt"].dt.year
        df["created_month"] = df["created_dt"].dt.to_period("M").astype(str)

        # Bot/boilerplate filtering
        df["is_bot"] = df["author"].apply(self._is_probable_bot) | df["text_raw"].apply(self._contains_bot_phrase)
        df = df[~df["is_bot"]].copy()

        # Dedup
        subset = list(cfg.dedup_subset) if cfg.dedup_subset else None
        if subset:
            df = df.drop_duplicates(subset=subset, keep=cfg.dedup_keep).copy()
        elif cfg.dedup_on:
            df = df.drop_duplicates(subset=[cfg.dedup_on], keep=cfg.dedup_keep).copy()

        # doc_id (stable-ish)
        df["doc_id"] = df.apply(
            lambda r: _hash_text(
                f"{r.get('source_tag','')}|{r.get('subreddit','')}|{int(bool(r.get('is_submission')))}|{r.get('created_dt','')}|{r.get('author','')}|{r.get('text_clean','')}"
            ),
            axis=1,
        )

        # Select + order
        cols = [
            "doc_id", "is_submission", "subreddit", "source_tag",
            "created_dt", "created_year", "created_month",
            "author", "author_deleted", "score",
            "link", "url", "title", "selftext",
            "text_raw", "text_clean", "n_tokens",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[cols].reset_index(drop=True)

        return df


# -----------------------------
# Embedding cache + BERTopic
# -----------------------------
class EmbeddingCache:
    """
    Disk cache keyed by SHA1(text_clean) + model name + cache version.
    Stores np.float32 arrays in .npy with sidecar meta.
    """
    def __init__(self, cache_dir: str, model_name: str, version: str = "v1"):
        self.cache_dir = _ensure_dir(cache_dir)
        self.model_name = model_name
        self.version = version

    def _key(self, text: str) -> str:
        h = _hash_text(text)
        m = _hash_text(self.model_name)[:10]
        v = _hash_text(self.version)[:8]
        return f"{v}_{m}_{h}"

    def get_path(self, text: str) -> Path:
        return self.cache_dir / f"{self._key(text)}.npy"

    def has(self, text: str) -> bool:
        return self.get_path(text).exists()

    def load(self, text: str) -> Optional[np.ndarray]:
        p = self.get_path(text)
        if not p.exists():
            return None
        arr = np.load(p)
        return arr

    def save(self, text: str, emb: np.ndarray) -> None:
        p = self.get_path(text)
        np.save(p, emb.astype(np.float32), allow_pickle=False)


class RedditBERTopicPipeline:
    def __init__(self, cfg: BERTopicConfig):
        self.cfg = cfg
        self._topic_model = None
        self._vectorizer = None
        self._embedding_model = None
        self._stopwords = default_stopwords(cfg)

        # cache
        self._cache = None
        if cfg.embedding_cache_enabled:
            self._cache = EmbeddingCache(cfg.embedding_cache_dir, cfg.embedding_model_name, cfg.embedding_cache_version)

    # ---- vectorizer
    def build_vectorizer(self) -> CountVectorizer:
        cfg = self.cfg
        # CountVectorizer has its own token pattern; keep simple and let our cleaning handle reddit junk
        vec = CountVectorizer(
            stop_words=list(self._stopwords),
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            max_features=cfg.max_features,
        )
        self._vectorizer = vec
        return vec

    # ---- embeddings (reproducible + cached)
    def _get_embedding_model(self):
        """Lazily initialize and reuse the sentence-transformers model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.cfg.embedding_model_name,
                device=self.cfg.embedding_device,
            )
        return self._embedding_model

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        cfg = self.cfg
        model = self._get_embedding_model()

        # Two-pass: load cached where possible, encode remaining in batches
        embs = [None] * len(texts)
        missing_idx = []

        if self._cache is not None:
            for i, t in enumerate(texts):
                cached = self._cache.load(t)
                if cached is None:
                    missing_idx.append(i)
                else:
                    embs[i] = cached
        else:
            missing_idx = list(range(len(texts)))

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = model.encode(
                missing_texts,
                batch_size=cfg.embedding_batch_size,
                show_progress_bar=cfg.embedding_show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            for j, i in enumerate(missing_idx):
                embs[i] = new_embs[j]
                if self._cache is not None:
                    self._cache.save(texts[i], embs[i])

        return np.vstack(embs).astype(np.float32)

    # ---- model build
    def build_topic_model(self):
        cfg = self.cfg

        # Keep hdbscan compatible with newer scikit-learn check_array kwargs.
        _patch_sklearn_check_array_compat()
        # Keep BERTopic import compatible with older sentence-transformers.
        _patch_sentence_transformers_static_embedding_compat()

        # Lazy imports so dataset building doesn't require topic deps
        from bertopic import BERTopic
        import umap
        import hdbscan
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

        # Reproducibility
        np.random.seed(cfg.random_state)

        umap_model = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=cfg.umap_n_components,
            min_dist=cfg.umap_min_dist,
            metric="cosine",
            random_state=cfg.random_state,
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        )

        vectorizer_model = self.build_vectorizer()

        # Better labels: combine KeyBERTInspired + MMR to reduce redundant ngrams
        representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
        }

        calculate_probabilities = (
            cfg.enable_probabilities
            if cfg.enable_probabilities is not None
            else cfg.calculate_probabilities
        )

        topic_model = BERTopic(
            language=cfg.language,
            embedding_model=self._get_embedding_model(),
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            calculate_probabilities=calculate_probabilities,
            verbose=True,
        )
        self._topic_model = topic_model
        return topic_model

    # ---- fit/transform
    def fit_transform(
        self,
        docs: Sequence[str],
        embeddings: Optional[np.ndarray] = None,
    ):
        if self._topic_model is None:
            self.build_topic_model()
        if embeddings is None:
            embeddings = self.encode_texts(docs)
        topics, probs = self._topic_model.fit_transform(docs, embeddings)
        return topics, probs, embeddings

    # ---- BERTopic features you weren’t using yet
    def topics_over_time(
        self,
        docs: Sequence[str],
        timestamps: Sequence[Any],
        topics: Optional[Sequence[int]] = None,
        nr_bins: Optional[int] = None,
        global_tuning: bool = True,
    ) -> pd.DataFrame:
        """
        Wrapper around BERTopic.topics_over_time.
        timestamps can be years, datetimes, etc.
        """
        if self._topic_model is None:
            raise RuntimeError("Topic model not built. Call fit_transform() first.")
        return self._topic_model.topics_over_time(
            docs=docs,
            timestamps=timestamps,
            topics=topics,
            nr_bins=nr_bins,
            global_tuning=global_tuning,
        )

    def document_info(self, docs: Sequence[str]) -> pd.DataFrame:
        """
        BERTopic.get_document_info
        """
        if self._topic_model is None:
            raise RuntimeError("Topic model not built. Call fit_transform() first.")
        return self._topic_model.get_document_info(docs)

    def hierarchical_topics(self, docs: Sequence[str]) -> pd.DataFrame:
        """
        BERTopic.hierarchical_topics
        """
        if self._topic_model is None:
            raise RuntimeError("Topic model not built. Call fit_transform() first.")
        return self._topic_model.hierarchical_topics(docs)

    def reduce_topics(self, docs: Sequence[str], nr_topics: int | str = "auto") -> Any:
        """
        Reduce number of topics ("auto" uses HDBSCAN/UMAP structure).
        """
        if self._topic_model is None:
            raise RuntimeError("Topic model not built. Call fit_transform() first.")
        self._topic_model.reduce_topics(docs, nr_topics=nr_topics)
        return self._topic_model

    @property
    def model(self):
        return self._topic_model
