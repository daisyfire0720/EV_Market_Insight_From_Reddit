from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .topic_refine_pipeline import (
	TopicRefineConfig,
)


class TopicHierarchyExplorer:
	"""Hierarchy / merge helper.

	Exact BERTopic hierarchy/merge requires the trained model object.
	When only CSV outputs are available, this class falls back to an approximate
	similarity-based merge recommendation using topic text representations.
	"""

	def __init__(self, config: Optional[TopicRefineConfig] = None):
		self.config = config or TopicRefineConfig()

	@staticmethod
	def _topic_text(topic_info: pd.DataFrame) -> pd.Series:
		def row_text(row: pd.Series) -> str:
			parts = [
				str(row.get("topic_label_llm", "")),
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

		rows = []
		idx = list(use.index)
		for i in range(len(idx)):
			for j in range(i + 1, len(idx)):
				s = float(sim[i, j])
				if s >= self.config.hierarchy_similarity_threshold:
					rows.append(
						{
							"topic_a": int(use.iloc[i]["Topic"]),
							"label_a": use.iloc[i].get("topic_label_llm"),
							"topic_b": int(use.iloc[j]["Topic"]),
							"label_b": use.iloc[j].get("topic_label_llm"),
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


class RedditTopicEDA:
	def __init__(
		self,
		topic_info: pd.DataFrame,
		document_topics: pd.DataFrame,
		yearly_stats: Optional[pd.DataFrame] = None,
		config: Optional[TopicRefineConfig] = None,
	):
		self.config = config or TopicRefineConfig()
		self.hierarchy = TopicHierarchyExplorer(self.config)

		self.topic_info_raw = topic_info.copy()
		self.document_topics_raw = document_topics.copy()
		self.yearly_stats = yearly_stats.copy() if yearly_stats is not None else None

		self.topic_info = self.topic_info_raw.copy()
		# Use final LLM labels for exploration; keep backward compatibility with older outputs.
		if "topic_label_llm" not in self.topic_info.columns and "topic_label_refined" in self.topic_info.columns:
			self.topic_info["topic_label_llm"] = self.topic_info["topic_label_refined"]
		elif "topic_label_llm" not in self.topic_info.columns:
			self.topic_info["topic_label_llm"] = self.topic_info.get("topic_label_bert", "")
		self.documents = self._prepare_documents(self.document_topics_raw)
		self.documents_enriched = self._attach_topic_info(self.documents, self.topic_info)

	@classmethod
	def from_csv(
		cls,
		topic_info_path: str | Path,
		document_topics_path: str | Path,
		yearly_stats_path: Optional[str | Path] = None,
		config: Optional[TopicRefineConfig] = None,
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
			c for c in ["Topic", "Count", "Name", "topic_label_llm", "topic_keywords_clean", "topic_keybert_clean"]
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
			df.groupby(["topic", "topic_label_llm"], dropna=False)
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
			df.groupby(["created_year", "topic", "topic_label_llm"], dropna=False)
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
			df.groupby([level, "topic", "topic_label_llm"], dropna=False)
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
			df.groupby(["topic", "topic_label_llm"], dropna=False)
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
			"topic_labels": outdir / "topic_labels_llm.csv",
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
		ax.barh(data["topic_label_llm"], data["doc_count"])
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
		top_topics = self.topic_prevalence().head(top_n)[["topic", "topic_label_llm"]]
		data = self.topic_trends().merge(top_topics, on=["topic", "topic_label_llm"], how="inner")
		fig, ax = plt.subplots(figsize=(11, 6))
		for _topic_id, d in data.groupby("topic"):
			d = d.sort_values("created_year")
			ax.plot(d["created_year"], d["share_within_year"], marker="o", label=d["topic_label_llm"].iloc[0])
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
		top_labels = self.topic_prevalence().head(top_n)["topic_label_llm"].tolist()
		data = self.subreddit_difference(level=level)
		pivot = (
			data[data["topic_label_llm"].isin(top_labels)]
			.pivot_table(index="topic_label_llm", columns=level, values="share_within_group", fill_value=0.0)
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
		ax.barh(data["topic_label_llm"], data["avg_score"])
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
