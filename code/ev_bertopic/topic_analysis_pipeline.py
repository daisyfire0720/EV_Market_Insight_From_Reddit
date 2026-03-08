from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TopicAnalysisConfig:
    top_n_topics_overall: int = 20
    top_n_topics_trend: int = 10
    min_topic_docs_for_engagement: int = 20
    confidence_threshold: Optional[float] = None
    exclude_outlier_topic: bool = True
    topic_label_max_words: int = 6
    label_generic_words: Tuple[str, ...] = (
        "ev", "car", "cars", "vehicle", "vehicles", "buy", "buying", "used",
        "new", "year", "miles", "mile", "thing", "things", "good", "bad",
        "better", "best", "need", "want", "like", "people", "look", "looking",
    )
    extra_stopwords: Tuple[str, ...] = (
        "reddit", "subreddit", "thread", "post", "comment", "deleted", "removed",
        "automoderator", "amp", "nbsp", "x200b",
    )
    hierarchy_similarity_threshold: float = 0.55
    hierarchy_top_pairs: int = 25
    subreddit_group_map: Dict[str, str] = field(default_factory=dict)


class TopicLabelRefiner:
    """Create more human-readable labels by combining Name, Representation, and KeyBERT.

    Notes:
    - Uses Name as the strongest signal because it often already captures the topic concept.
    - Uses Representation/KeyBERT to refine and choose a cleaner phrase.
    - Deterministic and fully offline.
    """

    def __init__(self, config: Optional[TopicAnalysisConfig] = None):
        self.config = config or TopicAnalysisConfig()
        self.stopwords = set(ENGLISH_STOP_WORDS).union(self.config.extra_stopwords)

    @staticmethod
    def _safe_list_parse(value: Any) -> List[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        s = str(value).strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # fallback: split on comma if it looks like a flat string
        if "," in s:
            return [x.strip(" []'\"") for x in s.split(",") if x.strip(" []'\"")]
        return [s.strip(" []'\"")]

    @staticmethod
    def _remove_topic_prefix(name: str) -> str:
        s = str(name)
        if "_" in s and re.match(r"^-?\d+_", s):
            return s.split("_", 1)[1]
        return s

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        t = str(text).lower()
        t = t.replace("_", " ")
        t = re.sub(r"[^a-z0-9\s\-+/]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _simple_singular(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    def _canonical_tokens(self, phrase: str) -> List[str]:
        toks = [self._simple_singular(x) for x in self._normalize_phrase(phrase).split()]
        return [t for t in toks if t and t not in self.stopwords]

    def _is_generic_phrase(self, phrase: str) -> bool:
        toks = self._canonical_tokens(phrase)
        if not toks:
            return True
        generic = set(self.config.label_generic_words)
        return all(t in generic for t in toks)

    def _deduplicate_phrases(self, phrases: Sequence[str]) -> List[str]:
        kept: List[str] = []
        seen: List[Tuple[str, ...]] = []
        for phrase in phrases:
            toks = tuple(self._canonical_tokens(phrase))
            if not toks:
                continue
            is_dup = False
            for prev in seen:
                overlap = len(set(toks).intersection(prev)) / max(1, len(set(toks).union(prev)))
                if overlap >= 0.8:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(phrase)
                seen.append(toks)
        return kept

    def _candidate_phrases(self, row: pd.Series) -> List[str]:
        name_raw = self._remove_topic_prefix(row.get("Name", ""))
        rep = self._safe_list_parse(row.get("Representation"))
        keybert = self._safe_list_parse(row.get("KeyBERT"))
        mmr = self._safe_list_parse(row.get("MMR"))

        name_terms = [x for x in name_raw.split("_") if x]
        name_phrase_2 = " ".join(name_terms[:2]).strip()
        name_phrase_3 = " ".join(name_terms[:3]).strip()
        name_phrase_4 = " ".join(name_terms[:4]).strip()

        candidates: List[str] = []
        # Prioritize human-like phrases first.
        candidates.extend([name_phrase_2, name_phrase_3, name_phrase_4])
        candidates.extend(keybert[:5])
        candidates.extend(mmr[:5])

        # Add combined phrase from top representation words if helpful.
        rep_terms = [self._normalize_phrase(x) for x in rep[:5]]
        rep_terms = [x for x in rep_terms if x]
        if len(rep_terms) >= 2:
            candidates.append(" ".join(rep_terms[:2]))
        if len(rep_terms) >= 3:
            candidates.append(" ".join(rep_terms[:3]))

        # Add unigram phrases from name as a fallback.
        candidates.extend(name_terms[:4])

        candidates = [self._normalize_phrase(x) for x in candidates if str(x).strip()]
        candidates = self._deduplicate_phrases(candidates)
        return candidates

    def choose_label(self, row: pd.Series) -> str:
        topic = row.get("Topic")
        if topic == -1:
            return "Outlier / Mixed"

        # Build source token inventory.
        name_raw = self._remove_topic_prefix(row.get("Name", ""))
        name_terms = [t for t in self._normalize_phrase(name_raw).split() if t]
        rep_terms = [self._normalize_phrase(x) for x in self._safe_list_parse(row.get("Representation"))[:10]]
        keybert_terms = [self._normalize_phrase(x) for x in self._safe_list_parse(row.get("KeyBERT"))[:10]]
        mmr_terms = [self._normalize_phrase(x) for x in self._safe_list_parse(row.get("MMR"))[:10]]
        candidates = self._candidate_phrases(row)

        # Concept-driven templates. These make labels much more semantic than raw joins.
        combined_text = " | ".join([name_raw, " ".join(rep_terms), " ".join(keybert_terms), " ".join(mmr_terms)]).lower()
        token_set = set(self._canonical_tokens(combined_text))
        phrase_set = set(keybert_terms + rep_terms + mmr_terms + candidates)

        def has_token(*tokens: str) -> bool:
            return all(self._simple_singular(t.lower()) in token_set for t in tokens)

        def has_any(*tokens: str) -> bool:
            return any(self._simple_singular(t.lower()) in token_set for t in tokens)

        def has_phrase_contains(*needles: str) -> bool:
            return any(all(n in p for n in needles) for p in phrase_set)

        if has_token("tax", "credit"):
            return "Federal Tax Credit" if has_token("federal") else "Tax Credit"
        if has_any("lease", "loan", "payment") and has_any("loan", "payment", "finance", "financing"):
            return "Lease And Financing"
        if has_token("battery") and has_any("replacement", "warranty", "life", "degradation"):
            return "Battery Replacement"
        if has_any("charging", "charger", "station", "network"):
            if has_any("public", "station", "network"):
                return "Public Charging"
            return "EV Charging"
        if has_any("range") and has_any("mile", "miles", "trip", "trips", "daily"):
            if has_any("phev"):
                return "PHEV Range"
            return "Driving Range"
        if has_any("hybrid") and has_phrase_contains("non hybrid"):
            return "Hybrid Vs Non-Hybrid"
        if has_any("msrp", "dealer", "dealers", "markup") and has_any("rav4", "toyota"):
            return "RAV4 Pricing"
        if has_any("toyota", "camry", "rav4", "corolla", "prius") and has_any("hybrid"):
            return "Toyota Hybrid Shopping"
        if has_any("new car", "buying", "advice") or has_phrase_contains("car buying"):
            return "Car Buying Advice"

        if not candidates:
            return f"Topic {topic}"

        def score_phrase(phrase: str) -> Tuple[float, float, float]:
            toks = self._canonical_tokens(phrase)
            n = len(toks)
            score = 0.0
            if 2 <= n <= self.config.topic_label_max_words:
                score += 4.0
            elif n == 1:
                score += 1.0
            else:
                score += max(0.0, 2.5 - 0.3 * abs(n - 3))
            if self._is_generic_phrase(phrase):
                score -= 2.5
            if any(any(ch.isdigit() for ch in t) for t in toks):
                score -= 1.5
            # Prefer keybert phrases with informative concepts.
            if phrase in keybert_terms:
                score += 1.8
            if phrase in mmr_terms:
                score += 0.8
            # Penalize duplicate concepts like "ev evs".
            score += len(set(toks)) / max(1, len(toks))
            if len(set(toks)) < len(toks):
                score -= 1.2
            return score, -abs(n - 2), -len(phrase)

        best = max(candidates, key=score_phrase)
        toks = self._canonical_tokens(best)[: self.config.topic_label_max_words]
        if not toks:
            return f"Topic {topic}"
        label = " ".join(toks).title()
        label = re.sub(r"\bEv\b", "EV", label)
        label = re.sub(r"\bPhev\b", "PHEV", label)
        label = re.sub(r"\bBev\b", "BEV", label)
        label = re.sub(r"\bMsrp\b", "MSRP", label)
        return label

    def build_topic_label_table(self, topic_info: pd.DataFrame) -> pd.DataFrame:
        out = topic_info.copy()
        out["topic_label"] = out.apply(self.choose_label, axis=1)
        out["topic_keywords_clean"] = out["Representation"].apply(
            lambda x: ", ".join(self._safe_list_parse(x)[:8])
        )
        out["topic_keybert_clean"] = out["KeyBERT"].apply(
            lambda x: ", ".join(self._safe_list_parse(x)[:8])
        )
        return out


class TopicHierarchyExplorer:
    """Hierarchy / merge helper.

    Exact BERTopic hierarchy/merge requires the trained model object.
    When only CSV outputs are available, this class falls back to an approximate
    similarity-based merge recommendation using topic text representations.
    """

    def __init__(self, config: Optional[TopicAnalysisConfig] = None):
        self.config = config or TopicAnalysisConfig()

    @staticmethod
    def _topic_text(topic_info: pd.DataFrame) -> pd.Series:
        def row_text(row: pd.Series) -> str:
            parts = [
                str(row.get("topic_label", "")),
                str(row.get("Name", "")),
                str(row.get("topic_keywords_clean", "")),
                str(row.get("topic_keybert_clean", "")),
                str(row.get("MMR", "")),
            ]
            return " ".join(parts)
        return topic_info.apply(row_text, axis=1)

    def recommend_merges_from_csv(self, topic_info: pd.DataFrame) -> pd.DataFrame:
        use = topic_info.copy()
        if self.config.exclude_outlier_topic:
            use = use[use["Topic"] != -1].copy()
        if use.empty:
            return pd.DataFrame(columns=["topic_a", "label_a", "topic_b", "label_b", "similarity"])

        text = self._topic_text(use)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = vec.fit_transform(text)
        sim = cosine_similarity(X)

        rows: List[Dict[str, Any]] = []
        idx = list(use.index)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                s = float(sim[i, j])
                if s >= self.config.hierarchy_similarity_threshold:
                    rows.append(
                        {
                            "topic_a": int(use.iloc[i]["Topic"]),
                            "label_a": use.iloc[i].get("topic_label"),
                            "topic_b": int(use.iloc[j]["Topic"]),
                            "label_b": use.iloc[j].get("topic_label"),
                            "similarity": round(s, 4),
                        }
                    )
        if not rows:
            return pd.DataFrame(columns=["topic_a", "label_a", "topic_b", "label_b", "similarity"])
        out = pd.DataFrame(rows).sort_values("similarity", ascending=False)
        return out.head(self.config.hierarchy_top_pairs).reset_index(drop=True)

    def hierarchical_topics_from_model(self, topic_model: Any, docs: Sequence[str]) -> pd.DataFrame:
        if topic_model is None:
            raise ValueError("topic_model is required for exact BERTopic hierarchy.")
        return topic_model.hierarchical_topics(list(docs))

    def reduce_topics_from_model(
        self,
        topic_model: Any,
        docs: Sequence[str],
        nr_topics: int | str = "auto",
    ) -> Any:
        if topic_model is None:
            raise ValueError("topic_model is required for exact BERTopic topic reduction.")
        topic_model.reduce_topics(list(docs), nr_topics=nr_topics)
        return topic_model

class RedditTopicEDA:
    def __init__(
        self,
        topic_info: pd.DataFrame,
        document_topics: pd.DataFrame,
        yearly_stats: Optional[pd.DataFrame] = None,
        config: Optional[TopicAnalysisConfig] = None,
    ):
        self.config = config or TopicAnalysisConfig()
        self.labeler = TopicLabelRefiner(self.config)
        self.hierarchy = TopicHierarchyExplorer(self.config)

        self.topic_info_raw = topic_info.copy()
        self.document_topics_raw = document_topics.copy()
        self.yearly_stats = yearly_stats.copy() if yearly_stats is not None else None

        self.topic_info = self.labeler.build_topic_label_table(self.topic_info_raw)
        self.documents = self._prepare_documents(self.document_topics_raw)
        self.documents_enriched = self._attach_topic_info(self.documents, self.topic_info)

    @classmethod
    def from_csv(
        cls,
        topic_info_path: str | Path,
        document_topics_path: str | Path,
        yearly_stats_path: Optional[str | Path] = None,
        config: Optional[TopicAnalysisConfig] = None,
    ) -> "RedditTopicEDA":
        topic_info = pd.read_csv(topic_info_path)
        document_topics = pd.read_csv(document_topics_path)
        yearly_stats = pd.read_csv(yearly_stats_path) if yearly_stats_path else None
        return cls(topic_info, document_topics, yearly_stats, config=config)

    def _prepare_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "created_dt" in out.columns:
            out["created_dt"] = pd.to_datetime(out["created_dt"], errors="coerce")
        if "created_year" not in out.columns and "created_dt" in out.columns:
            out["created_year"] = out["created_dt"].dt.year
        if "score" in out.columns:
            out["score"] = pd.to_numeric(out["score"], errors="coerce")
        if "n_tokens" in out.columns:
            out["n_tokens"] = pd.to_numeric(out["n_tokens"], errors="coerce")
        if "topic_probability_max" in out.columns:
            out["topic_probability_max"] = pd.to_numeric(out["topic_probability_max"], errors="coerce")
        if self.config.confidence_threshold is not None and "topic_probability_max" in out.columns:
            out = out[
                out["topic_probability_max"].isna() | (out["topic_probability_max"] >= self.config.confidence_threshold)
            ].copy()
        if self.config.subreddit_group_map and "subreddit" in out.columns:
            out["subreddit_group"] = out["subreddit"].map(self.config.subreddit_group_map).fillna("other")
        return out

    def _attach_topic_info(self, docs: pd.DataFrame, topic_info: pd.DataFrame) -> pd.DataFrame:
        right_cols = [
            c for c in ["Topic", "Count", "Name", "topic_label", "topic_keywords_clean", "topic_keybert_clean"]
            if c in topic_info.columns
        ]
        merged = docs.merge(topic_info[right_cols], left_on="topic", right_on="Topic", how="left")
        return merged

    def _filter_topic_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.config.exclude_outlier_topic and "topic" in out.columns:
            out = out[out["topic"] != -1].copy()
        return out

    def topic_prevalence(self) -> pd.DataFrame:
        df = self._filter_topic_rows(self.documents_enriched)
        out = (
            df.groupby(["topic", "topic_label"], dropna=False)
            .size()
            .reset_index(name="doc_count")
            .sort_values("doc_count", ascending=False)
            .reset_index(drop=True)
        )
        total = out["doc_count"].sum()
        out["share"] = out["doc_count"] / total if total else np.nan
        return out

    def topic_trends(self) -> pd.DataFrame:
        df = self._filter_topic_rows(self.documents_enriched)
        out = (
            df.groupby(["created_year", "topic", "topic_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["created_year", "count"], ascending=[True, False])
        )
        out["share_within_year"] = out.groupby("created_year")["count"].transform(lambda x: x / x.sum())
        return out.reset_index(drop=True)

    def subreddit_difference(self, level: str = "subreddit") -> pd.DataFrame:
        if level not in self.documents_enriched.columns:
            raise ValueError(f"{level} not found in document table.")
        df = self._filter_topic_rows(self.documents_enriched)
        out = (
            df.groupby([level, "topic", "topic_label"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        out["share_within_group"] = out.groupby(level)["count"].transform(lambda x: x / x.sum())
        return out.sort_values([level, "count"], ascending=[True, False]).reset_index(drop=True)

    def engagement_analysis(self) -> pd.DataFrame:
        df = self._filter_topic_rows(self.documents_enriched)
        if "score" not in df.columns:
            raise ValueError("score column not found in document table.")
        out = (
            df.groupby(["topic", "topic_label"], dropna=False)
            .agg(
                doc_count=("score", "size"),
                avg_score=("score", "mean"),
                median_score=("score", "median"),
                p75_score=("score", lambda x: np.nanpercentile(x, 75)),
                avg_tokens=("n_tokens", "mean") if "n_tokens" in df.columns else ("score", "size"),
            )
            .reset_index()
        )
        out = out[out["doc_count"] >= self.config.min_topic_docs_for_engagement].copy()
        return out.sort_values("avg_score", ascending=False).reset_index(drop=True)

    def confidence_summary(self) -> pd.DataFrame:
        if "topic_probability_max" not in self.documents_enriched.columns:
            return pd.DataFrame()
        s = self.documents_enriched["topic_probability_max"].dropna()
        if s.empty:
            return pd.DataFrame(columns=["metric", "value"])
        return pd.DataFrame(
            {
                "metric": ["count", "mean", "median", "p25", "p75", "min", "max"],
                "value": [
                    s.size,
                    s.mean(),
                    s.median(),
                    s.quantile(0.25),
                    s.quantile(0.75),
                    s.min(),
                    s.max(),
                ],
            }
        )

    def merge_recommendations(self) -> pd.DataFrame:
        return self.hierarchy.recommend_merges_from_csv(self.topic_info)

    def export_analysis_tables(self, outdir: str | Path) -> Dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "topic_labels": outdir / "topic_labels_refined.csv",
            "topic_prevalence": outdir / "eda_topic_prevalence.csv",
            "topic_trends": outdir / "eda_topic_trends.csv",
            "subreddit_difference": outdir / "eda_subreddit_difference.csv",
            "engagement": outdir / "eda_topic_engagement.csv",
            "merge_recommendations": outdir / "eda_topic_merge_recommendations.csv",
            "documents_enriched": outdir / "documents_topics_enriched.csv",
        }
        self.topic_info.to_csv(paths["topic_labels"], index=False)
        self.topic_prevalence().to_csv(paths["topic_prevalence"], index=False)
        self.topic_trends().to_csv(paths["topic_trends"], index=False)
        self.subreddit_difference().to_csv(paths["subreddit_difference"], index=False)
        self.engagement_analysis().to_csv(paths["engagement"], index=False)
        self.merge_recommendations().to_csv(paths["merge_recommendations"], index=False)
        self.documents_enriched.to_csv(paths["documents_enriched"], index=False)
        return paths

    # -------------------------
    # Visualization helpers
    # -------------------------
    def _save_or_show(
        self,
        fig: plt.Figure,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> None:
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_topic_prevalence(
        self,
        top_n: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        top_n = top_n or self.config.top_n_topics_overall
        data = self.topic_prevalence().head(top_n).sort_values("doc_count", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(data))))
        ax.barh(data["topic_label"], data["doc_count"])
        ax.set_title("Top Topic Prevalence")
        ax.set_xlabel("Document Count")
        ax.set_ylabel("Topic")
        self._save_or_show(fig, save_path, show=show)
        return data

    def plot_topic_trends(
        self,
        top_n: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        top_n = top_n or self.config.top_n_topics_trend
        top_topics = self.topic_prevalence().head(top_n)[["topic", "topic_label"]]
        data = self.topic_trends().merge(top_topics, on=["topic", "topic_label"], how="inner")
        fig, ax = plt.subplots(figsize=(11, 6))
        for topic_id, d in data.groupby("topic"):
            d = d.sort_values("created_year")
            ax.plot(d["created_year"], d["share_within_year"], marker="o", label=d["topic_label"].iloc[0])
        ax.set_title("Topic Share Trends Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Share Within Year")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        self._save_or_show(fig, save_path, show=show)
        return data

    def plot_subreddit_heatmap(
        self,
        top_n: Optional[int] = None,
        level: str = "subreddit",
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        top_n = top_n or self.config.top_n_topics_overall
        top_labels = self.topic_prevalence().head(top_n)["topic_label"].tolist()
        data = self.subreddit_difference(level=level)
        pivot = (
            data[data["topic_label"].isin(top_labels)]
            .pivot_table(index="topic_label", columns=level, values="share_within_group", fill_value=0.0)
        )
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * pivot.shape[1]), max(5, 0.45 * pivot.shape[0])))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Topic Share by {level.title()}")
        fig.colorbar(im, ax=ax, label="Share Within Group")
        self._save_or_show(fig, save_path, show=show)
        return pivot.reset_index()

    def plot_topic_engagement(
        self,
        top_n: int = 15,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        data = self.engagement_analysis().head(top_n).sort_values("avg_score", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(data))))
        ax.barh(data["topic_label"], data["avg_score"])
        ax.set_title("Most Engaging Topics")
        ax.set_xlabel("Average Reddit Score")
        ax.set_ylabel("Topic")
        self._save_or_show(fig, save_path, show=show)
        return data

    def plot_confidence_histogram(
        self,
        bins: int = 30,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        if "topic_probability_max" not in self.documents_enriched.columns:
            return pd.DataFrame()
        s = self.documents_enriched["topic_probability_max"].dropna()
        if s.empty:
            return pd.DataFrame()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(s.values, bins=bins)
        ax.set_title("Topic Assignment Confidence")
        ax.set_xlabel("topic_probability_max")
        ax.set_ylabel("Document Count")
        self._save_or_show(fig, save_path, show=show)
        return self.confidence_summary()

    def plot_outlier_share_by_year(
        self,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        if "created_year" not in self.documents_enriched.columns:
            return pd.DataFrame()
        df = self.documents_enriched.copy()
        out = (
            df.assign(is_outlier=df["topic"].eq(-1).astype(int))
            .groupby("created_year")
            .agg(doc_count=("topic", "size"), outlier_count=("is_outlier", "sum"))
            .reset_index()
        )
        out["outlier_share"] = out["outlier_count"] / out["doc_count"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(out["created_year"], out["outlier_share"], marker="o")
        ax.set_title("Outlier Share by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Outlier Share")
        self._save_or_show(fig, save_path, show=show)
        return out

    def run_all_eda(
        self,
        outdir: str | Path,
        show: bool = False,
        save: bool = True,
    ) -> Dict[str, Path]:
        save_paths: Dict[str, Path] = {}
        outdir_path = Path(outdir)

        if save:
            outdir_path.mkdir(parents=True, exist_ok=True)
            self.export_analysis_tables(outdir_path)
            save_paths = {
                "topic_prevalence_png": outdir_path / "topic_prevalence.png",
                "topic_trends_png": outdir_path / "topic_trends.png",
                "subreddit_heatmap_png": outdir_path / "subreddit_heatmap.png",
                "topic_engagement_png": outdir_path / "topic_engagement.png",
                "confidence_hist_png": outdir_path / "topic_confidence_histogram.png",
                "outlier_share_png": outdir_path / "outlier_share_by_year.png",
            }

        self.plot_topic_prevalence(save_path=save_paths.get("topic_prevalence_png"), show=show)
        self.plot_topic_trends(save_path=save_paths.get("topic_trends_png"), show=show)
        self.plot_subreddit_heatmap(save_path=save_paths.get("subreddit_heatmap_png"), show=show)
        self.plot_topic_engagement(save_path=save_paths.get("topic_engagement_png"), show=show)
        self.plot_confidence_histogram(save_path=save_paths.get("confidence_hist_png"), show=show)
        self.plot_outlier_share_by_year(save_path=save_paths.get("outlier_share_png"), show=show)
        return save_paths
