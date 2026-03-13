from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FunnelStageSpec:
    """Definition of a market funnel stage and its keyword signals."""

    name: str
    description: str
    include_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    priority: int = 0


class MarketFunnelAnalyzer:
    """
    Analyze topic-level outputs and map them into market funnel stages,
    then surface stage-specific pain points and opportunity areas.

    Expected input:
        A CSV / DataFrame with topic-level rows such as:
        - topic_id
        - topic_label_llm
        - topic_summary_llm
        - rep_doc_centroid
        - rep_doc_farthest
        - optional Count / doc_count / topic_size

    Typical usage:
        analyzer = MarketFunnelAnalyzer(topic_weight_col="Count")
        results = analyzer.run_full_analysis(csv_path="topic_labels_llm.csv")
        analyzer.export_results("market_funnel_analysis.xlsx")
    """

    DEFAULT_STAGE_SPECS = [
        FunnelStageSpec(
            name="Awareness",
            description="Users define needs, learn categories, and build an initial mental shortlist.",
            include_keywords=[
                "what car", "which car", "new car", "used car", "guide", "research",
                "beginner", "best", "recommend", "need a car", "first car", "buying guide",
                "worth it", "should i buy",
            ],
            priority=1,
        ),
        FunnelStageSpec(
            name="Consideration",
            description="Users narrow options and compare broad sets of brands, models, or powertrains.",
            include_keywords=[
                "compare", "comparison", "versus", "vs", "hybrid", "electric", "ev", "suv",
                "sedan", "reliability", "features", "resale", "warranty", "maintenance",
                "driving experience",
            ],
            priority=2,
        ),
        FunnelStageSpec(
            name="Evaluation",
            description="Users evaluate finalists with more concrete tradeoffs, economics, or fit-to-needs questions.",
            include_keywords=[
                "total cost", "ownership", "credit", "loan", "finance", "payment", "apr",
                "maintenance cost", "fuel efficiency", "safety", "test drive", "long term",
                "battery", "tax credit", "incentive",
            ],
            priority=3,
        ),
        FunnelStageSpec(
            name="Purchase",
            description="Users are actively transacting and discussing dealer execution, negotiation, or inventory availability.",
            include_keywords=[
                "dealer", "dealership", "markup", "msrp", "market adjustment", "doc fee",
                "allocation", "wait list", "deposit", "negotiat", "purchase", "bought",
                "on the lot", "inventory", "delivery", "out the door",
            ],
            priority=4,
        ),
        FunnelStageSpec(
            name="Ownership",
            description="Users discuss post-purchase experience, service, running costs, and satisfaction after buying.",
            include_keywords=[
                "owned", "ownership experience", "service", "repair", "problem", "issue",
                "warranty claim", "after buying", "over time", "long-term", "maintenance",
                "battery replacement",
            ],
            priority=5,
        ),
    ]

    PAIN_POINT_RULES = {
        "price_markup": {
            "name": "Price Markup / Overpricing",
            "keywords": ["markup", "market adjustment", "over msrp", "adm", "overpriced", "msrp", "doc fee"],
            "stage_hint": ["Purchase", "Evaluation"],
            "severity": 5,
        },
        "dealer_friction": {
            "name": "Dealer Friction / Trust Issues",
            "keywords": ["dealer", "dealership", "salesperson", "lied", "pressure", "bait", "finance office", "upsell"],
            "stage_hint": ["Purchase"],
            "severity": 4,
        },
        "inventory_wait": {
            "name": "Inventory / Wait-Time Constraints",
            "keywords": ["inventory", "allocation", "wait list", "deposit", "backorder", "delivery", "on the lot", "availability"],
            "stage_hint": ["Purchase", "Evaluation"],
            "severity": 4,
        },
        "financing_affordability": {
            "name": "Financing / Affordability",
            "keywords": ["loan", "apr", "payment", "monthly payment", "credit", "finance", "lease", "afford"],
            "stage_hint": ["Evaluation", "Purchase"],
            "severity": 4,
        },
        "tco_uncertainty": {
            "name": "Total Cost / Incentive Uncertainty",
            "keywords": ["tax credit", "incentive", "total cost", "ownership cost", "fuel savings", "insurance", "depreciation"],
            "stage_hint": ["Evaluation"],
            "severity": 3,
        },
        "charging_access": {
            "name": "Charging Access / Infrastructure",
            "keywords": ["charger", "charging", "supercharger", "range", "apartment", "garage", "240v", "fast charging"],
            "stage_hint": ["Evaluation", "Ownership"],
            "severity": 4,
        },
        "reliability_risk": {
            "name": "Reliability / Maintenance Risk",
            "keywords": ["reliability", "reliable", "maintenance", "repair", "issue", "problem", "warranty", "battery replacement"],
            "stage_hint": ["Consideration", "Ownership", "Evaluation"],
            "severity": 4,
        },
        "feature_fit": {
            "name": "Feature / Fit Tradeoff",
            "keywords": ["interior", "comfort", "safety", "cargo", "space", "awd", "feature", "trim", "visibility"],
            "stage_hint": ["Consideration"],
            "severity": 3,
        },
        "model_confusion": {
            "name": "Comparison Overload / Decision Complexity",
            "keywords": ["compare", "versus", "vs", "narrowed down", "which should i buy", "decide", "between"],
            "stage_hint": ["Awareness", "Consideration"],
            "severity": 3,
        },
    }

    NEGATIVE_SIGNAL_WORDS = {
        "problem", "issue", "concern", "worry", "worried", "bad", "hard", "difficult", "friction",
        "expensive", "overpriced", "markup", "delay", "limited", "confusing", "uncertain", "lied",
        "pressure", "regret", "risk", "stuck", "cannot", "can't", "worse", "unavailable",
    }

    QUESTION_SIGNAL_WORDS = {
        "should", "which", "what", "worth", "how", "can i", "do i", "is it", "would", "recommend",
    }

    REQUIRED_MIN_COLUMNS = ["topic_id"]
    TEXT_PRIORITY_COLUMNS = [
        "topic_label_llm", "topic_summary_llm", "topic_label_refined", "topic_label_bert",
        "rep_doc_centroid", "rep_doc_farthest",
    ]

    def __init__(
        self,
        stage_specs: Optional[List[FunnelStageSpec]] = None,
        topic_weight_col: Optional[str] = None,
        outlier_topic_id: int = -1,
    ) -> None:
        self.stage_specs = sorted(stage_specs or self.DEFAULT_STAGE_SPECS, key=lambda x: x.priority)
        self.topic_weight_col = topic_weight_col
        self.outlier_topic_id = outlier_topic_id
        self.df: Optional[pd.DataFrame] = None
        self.stage_df: Optional[pd.DataFrame] = None
        self.stage_summary_df: Optional[pd.DataFrame] = None
        self.pain_point_df: Optional[pd.DataFrame] = None
        self.pain_point_stage_summary_df: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------------
    # Loading / validation
    # ---------------------------------------------------------------------
    def load_csv(self, csv_path: str, **read_csv_kwargs) -> pd.DataFrame:
        self.df = pd.read_csv(csv_path, **read_csv_kwargs)
        self._validate_input(self.df)
        self.df = self._prepare_base_columns(self.df)
        return self.df.copy()

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        self.df = self._prepare_base_columns(df.copy())
        return self.df.copy()

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_MIN_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        text_cols_present = [c for c in self.TEXT_PRIORITY_COLUMNS if c in df.columns]
        if not text_cols_present:
            raise ValueError(
                "No supported text columns found. Need at least one of: "
                f"{self.TEXT_PRIORITY_COLUMNS}"
            )

        if self.topic_weight_col is not None and self.topic_weight_col not in df.columns:
            raise ValueError(
                f"topic_weight_col='{self.topic_weight_col}' not found in dataframe columns."
            )

    def _prepare_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.TEXT_PRIORITY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["topic_weight"] = df[self.topic_weight_col].astype(float) if self.topic_weight_col is not None else 1.0
        df["is_outlier"] = df["topic_id"].astype(str) == str(self.outlier_topic_id)
        df["combined_text"] = df.apply(self._combine_text_fields, axis=1)
        df["negative_signal_score"] = df["combined_text"].apply(self._compute_negative_signal_score)
        df["question_signal_score"] = df["combined_text"].apply(self._compute_question_signal_score)
        return df

    def _combine_text_fields(self, row: pd.Series) -> str:
        ordered = [row.get(c, "") for c in self.TEXT_PRIORITY_COLUMNS]
        text = " | ".join([x for x in ordered if str(x).strip()])
        return self._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text).lower()
        text = text.replace("â€™", "'")
        text = text.replace("’", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ---------------------------------------------------------------------
    # Stage assignment
    # ---------------------------------------------------------------------
    def assign_funnel_stages(self, exclude_outliers: bool = True) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Use load_csv() or load_dataframe() first.")

        df = self.df.copy()
        if exclude_outliers:
            df = df.loc[~df["is_outlier"]].copy()

        assignments = df.apply(self._score_row_to_stage, axis=1, result_type="expand")
        assignments.columns = [
            "funnel_stage", "stage_confidence", "stage_match_score", "stage_reason", "stage_keyword_hits",
        ]
        df = pd.concat([df.reset_index(drop=True), assignments.reset_index(drop=True)], axis=1)
        df["topic_display"] = df.apply(self._build_topic_display, axis=1)
        df["primary_signal"] = df["stage_reason"].str.split(";").str[0].fillna("")

        self.stage_df = df.sort_values(
            ["funnel_stage", "stage_confidence", "topic_weight"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        return self.stage_df.copy()

    def _score_row_to_stage(self, row: pd.Series) -> Tuple[str, float, float, str, str]:
        text = row["combined_text"]
        stage_scores = []

        for spec in self.stage_specs:
            hits = []
            score = 0.0

            for kw in spec.include_keywords:
                if kw in text:
                    hits.append(kw)
                    score += 1.0

            for kw in spec.exclude_keywords:
                if kw in text:
                    score -= 1.0

            label_text = self._clean_text(row.get("topic_label_llm", ""))
            summary_text = self._clean_text(row.get("topic_summary_llm", ""))
            centroid_text = self._clean_text(row.get("rep_doc_centroid", ""))

            label_hits = sum(1 for kw in spec.include_keywords if kw in label_text)
            summary_hits = sum(1 for kw in spec.include_keywords if kw in summary_text)
            centroid_hits = sum(1 for kw in spec.include_keywords if kw in centroid_text)
            score += 1.2 * label_hits + 0.8 * summary_hits + 0.5 * centroid_hits

            stage_scores.append((spec.name, score, hits, spec.description))

        stage_scores = sorted(stage_scores, key=lambda x: x[1], reverse=True)
        best_name, best_score, best_hits, _ = stage_scores[0]
        second_score = stage_scores[1][1] if len(stage_scores) > 1 else 0.0

        if best_score <= 0:
            best_name = "Unclear / Cross-Stage"
            confidence = 0.25
            reason = "No strong keyword evidence; topic appears broad or mixed"
            return best_name, confidence, float(best_score), reason, ""

        confidence = self._score_to_confidence(best_score, second_score)
        reason = self._build_stage_reason(best_name, best_hits, best_score, second_score)
        keyword_hits = ", ".join(sorted(set(best_hits)))[:500]
        return best_name, confidence, float(best_score), reason, keyword_hits

    @staticmethod
    def _score_to_confidence(best_score: float, second_score: float) -> float:
        margin = best_score - second_score
        raw = 0.45 + 0.08 * best_score + 0.10 * margin
        return float(max(0.0, min(0.99, raw)))

    @staticmethod
    def _build_stage_reason(stage_name: str, hits: List[str], best_score: float, second_score: float) -> str:
        if hits:
            hit_text = ", ".join(sorted(set(hits))[:6])
            return (
                f"Assigned to {stage_name} due to signals: {hit_text}; "
                f"score={best_score:.2f}, margin_vs_next={best_score - second_score:.2f}"
            )
        return (
            f"Assigned to {stage_name} based on overall topic wording; "
            f"score={best_score:.2f}, margin_vs_next={best_score - second_score:.2f}"
        )

    @staticmethod
    def _build_topic_display(row: pd.Series) -> str:
        label = row.get("topic_label_llm", "") or row.get("topic_label_refined", "")
        summary = row.get("topic_summary_llm", "")
        summary = summary[:180] + "..." if len(summary) > 180 else summary
        return f"{label} | {summary}".strip(" |")

    # ---------------------------------------------------------------------
    # Analysis outputs
    # ---------------------------------------------------------------------
    def build_stage_summary(self) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before build_stage_summary().")

        df = self.stage_df.copy()
        total_weight = df["topic_weight"].sum()

        grouped = (
            df.groupby("funnel_stage", dropna=False)
            .agg(
                n_topics=("topic_id", "count"),
                weighted_topics=("topic_weight", "sum"),
                avg_confidence=("stage_confidence", "mean"),
                avg_match_score=("stage_match_score", "mean"),
                avg_negative_signal=("negative_signal_score", "mean"),
                avg_question_signal=("question_signal_score", "mean"),
            )
            .reset_index()
        )

        grouped["weighted_share"] = np.where(total_weight > 0, grouped["weighted_topics"] / total_weight, np.nan)
        grouped["stage_role"] = grouped["funnel_stage"].map(self._stage_role_lookup())
        grouped["stage_health_flag"] = grouped.apply(self._stage_health_flag, axis=1)

        self.stage_summary_df = grouped.sort_values(["weighted_topics", "avg_confidence"], ascending=[False, False]).reset_index(drop=True)
        return self.stage_summary_df.copy()

    @staticmethod
    def _stage_role_lookup() -> Dict[str, str]:
        return {
            "Awareness": "Top-of-funnel demand formation",
            "Consideration": "Mid-funnel option narrowing",
            "Evaluation": "Decision-quality and economics validation",
            "Purchase": "Conversion friction and transaction execution",
            "Ownership": "Post-purchase retention and advocacy risk",
            "Unclear / Cross-Stage": "Mixed signal bucket",
        }

    @staticmethod
    def _stage_health_flag(row: pd.Series) -> str:
        share = row["weighted_share"]
        conf = row["avg_confidence"]
        if pd.isna(share):
            return "Unknown"
        if share >= 0.30 and conf >= 0.70:
            return "High signal concentration"
        if share >= 0.20:
            return "Meaningful stage presence"
        if conf < 0.55:
            return "Needs manual review"
        return "Secondary stage"

    def generate_stage_insights(self, top_n_topics: int = 3) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before generate_stage_insights().")

        insight_rows = []
        for stage, g in self.stage_df.groupby("funnel_stage", dropna=False):
            g = g.sort_values(["topic_weight", "stage_confidence"], ascending=[False, False])
            top_topics = g.head(top_n_topics)

            topic_list = " || ".join(top_topics["topic_label_llm"].fillna(top_topics["topic_label_refined"]).astype(str))
            key_signals = self._extract_common_terms(top_topics["combined_text"].tolist(), top_k=10)
            friction = self._infer_stage_friction(stage, " ".join(top_topics["combined_text"].tolist()))
            implication = self._infer_business_implication(stage, friction, key_signals)

            insight_rows.append(
                {
                    "funnel_stage": stage,
                    "top_topics": topic_list,
                    "common_signals": ", ".join(key_signals),
                    "core_friction_or_need": friction,
                    "business_implication": implication,
                    "n_topics_in_stage": len(g),
                    "weighted_topics": g["topic_weight"].sum(),
                    "avg_confidence": g["stage_confidence"].mean(),
                    "avg_negative_signal": g["negative_signal_score"].mean(),
                }
            )

        return pd.DataFrame(insight_rows).sort_values(["weighted_topics", "avg_confidence"], ascending=[False, False]).reset_index(drop=True)

    def build_topic_deep_dive(self) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before build_topic_deep_dive().")

        rows = []
        for _, row in self.stage_df.iterrows():
            text = row["combined_text"]
            primary_pain = self._score_pain_points_for_text(text, row["funnel_stage"])[0]
            rows.append(
                {
                    "topic_id": row["topic_id"],
                    "topic_label_llm": row.get("topic_label_llm", ""),
                    "funnel_stage": row["funnel_stage"],
                    "topic_weight": row["topic_weight"],
                    "stage_confidence": row["stage_confidence"],
                    "decision_signals": self._extract_signal_snippets(text),
                    "customer_need": self._infer_customer_need(text),
                    "purchase_barrier": self._infer_purchase_barrier(text),
                    "primary_pain_point": primary_pain["pain_point_name"],
                    "primary_pain_score": primary_pain["pain_score"],
                    "suggested_action": self._infer_suggested_action(row["funnel_stage"], text),
                }
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # Pain-point analyzer
    # ---------------------------------------------------------------------
    def build_pain_point_table(self, top_k_per_topic: int = 3) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before build_pain_point_table().")

        rows = []
        for _, row in self.stage_df.iterrows():
            scored = self._score_pain_points_for_text(row["combined_text"], row["funnel_stage"])
            for item in scored[:top_k_per_topic]:
                rows.append(
                    {
                        "topic_id": row["topic_id"],
                        "topic_label_llm": row.get("topic_label_llm", ""),
                        "funnel_stage": row["funnel_stage"],
                        "topic_weight": row["topic_weight"],
                        "stage_confidence": row["stage_confidence"],
                        "pain_point_code": item["pain_point_code"],
                        "pain_point_name": item["pain_point_name"],
                        "pain_keyword_hits": item["pain_keyword_hits"],
                        "pain_score": item["pain_score"],
                        "pain_severity": item["pain_severity"],
                        "negative_signal_score": row["negative_signal_score"],
                        "question_signal_score": row["question_signal_score"],
                        "opportunity_score": self._compute_opportunity_score(
                            weight=row["topic_weight"],
                            pain_score=item["pain_score"],
                            negative_signal=row["negative_signal_score"],
                            question_signal=row["question_signal_score"],
                        ),
                        "evidence_snippet": self._extract_evidence_snippet(row["combined_text"], item["matched_keywords"]),
                        "suggested_response": self._pain_point_suggested_response(item["pain_point_code"], row["funnel_stage"]),
                    }
                )

        self.pain_point_df = pd.DataFrame(rows).sort_values(
            ["opportunity_score", "topic_weight", "pain_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return self.pain_point_df.copy()

    def summarize_pain_points_by_stage(self, top_n: int = 5) -> pd.DataFrame:
        if self.pain_point_df is None:
            self.build_pain_point_table()

        grouped = (
            self.pain_point_df.groupby(["funnel_stage", "pain_point_name"], dropna=False)
            .agg(
                n_topic_hits=("topic_id", "nunique"),
                weighted_volume=("topic_weight", "sum"),
                avg_pain_score=("pain_score", "mean"),
                avg_negative_signal=("negative_signal_score", "mean"),
                avg_question_signal=("question_signal_score", "mean"),
                opportunity_score=("opportunity_score", "sum"),
            )
            .reset_index()
        )
        grouped["rank_within_stage"] = grouped.groupby("funnel_stage")["opportunity_score"].rank(method="dense", ascending=False)
        grouped["priority_flag"] = grouped.apply(self._priority_flag_from_opportunity, axis=1)
        grouped = grouped.sort_values(["funnel_stage", "opportunity_score"], ascending=[True, False]).reset_index(drop=True)
        self.pain_point_stage_summary_df = grouped.copy()
        if top_n is not None:
            grouped = grouped[grouped["rank_within_stage"] <= top_n].copy()
        return grouped.reset_index(drop=True)

    def generate_pain_point_insights(self, top_n_per_stage: int = 3) -> pd.DataFrame:
        summary = self.summarize_pain_points_by_stage(top_n=top_n_per_stage)
        rows = []
        for stage, g in summary.groupby("funnel_stage", dropna=False):
            g = g.sort_values("opportunity_score", ascending=False)
            top = g.head(top_n_per_stage)
            rows.append(
                {
                    "funnel_stage": stage,
                    "top_pain_points": " || ".join(top["pain_point_name"].tolist()),
                    "top_priority_flags": " || ".join(top["priority_flag"].tolist()),
                    "stage_opportunity_score": top["opportunity_score"].sum(),
                    "recommended_focus": self._stage_recommended_focus(stage, top["pain_point_name"].tolist()),
                }
            )
        return pd.DataFrame(rows).sort_values("stage_opportunity_score", ascending=False).reset_index(drop=True)

    def _score_pain_points_for_text(self, text: str, stage: str) -> List[Dict[str, object]]:
        scored = []
        lowered = self._clean_text(text)
        for code, rule in self.PAIN_POINT_RULES.items():
            matched = [kw for kw in rule["keywords"] if kw in lowered]
            stage_bonus = 0.75 if stage in rule.get("stage_hint", []) else 0.0
            score = len(matched) + stage_bonus
            if score <= 0:
                continue
            scored.append(
                {
                    "pain_point_code": code,
                    "pain_point_name": rule["name"],
                    "pain_keyword_hits": len(matched),
                    "matched_keywords": matched,
                    "pain_score": float(score),
                    "pain_severity": rule["severity"],
                }
            )
        if not scored:
            scored = [{
                "pain_point_code": "general_complexity",
                "pain_point_name": "General Decision Complexity",
                "pain_keyword_hits": 0,
                "matched_keywords": [],
                "pain_score": 0.5,
                "pain_severity": 2,
            }]
        return sorted(scored, key=lambda x: (x["pain_score"], x["pain_severity"]), reverse=True)

    def _compute_negative_signal_score(self, text: str) -> float:
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        hits = sum(1 for tok in tokens if tok in self.NEGATIVE_SIGNAL_WORDS)
        return round(hits / max(len(tokens), 1) * 20, 4)

    def _compute_question_signal_score(self, text: str) -> float:
        lowered = self._clean_text(text)
        qmarks = lowered.count("?")
        phrase_hits = sum(1 for p in self.QUESTION_SIGNAL_WORDS if p in lowered)
        return round(min(1.0, 0.2 * qmarks + 0.15 * phrase_hits), 4)

    @staticmethod
    def _compute_opportunity_score(weight: float, pain_score: float, negative_signal: float, question_signal: float) -> float:
        return float(round(weight * (1 + pain_score) * (1 + negative_signal) * (1 + question_signal), 4))

    @staticmethod
    def _extract_evidence_snippet(text: str, keywords: List[str], max_len: int = 220) -> str:
        if not keywords:
            return text[:max_len]
        for kw in keywords:
            idx = text.find(kw)
            if idx >= 0:
                start = max(0, idx - 70)
                end = min(len(text), idx + len(kw) + 120)
                snippet = text[start:end].strip()
                return snippet[:max_len] + ("..." if len(snippet) > max_len else "")
        return text[:max_len]

    @staticmethod
    def _priority_flag_from_opportunity(row: pd.Series) -> str:
        score = row["opportunity_score"]
        if score >= 500:
            return "Critical intervention opportunity"
        if score >= 150:
            return "High-priority friction"
        if score >= 50:
            return "Meaningful improvement area"
        return "Secondary issue"

    @staticmethod
    def _stage_recommended_focus(stage: str, pain_points: List[str]) -> str:
        joined = ", ".join(pain_points[:3])
        if stage == "Purchase":
            return f"Reduce conversion leakage by addressing {joined}" if joined else "Reduce transaction friction"
        if stage == "Evaluation":
            return f"Improve economic proof and decision support around {joined}" if joined else "Improve calculators and proof points"
        if stage == "Consideration":
            return f"Clarify comparison and fit tradeoffs around {joined}" if joined else "Clarify comparison experience"
        if stage == "Ownership":
            return f"Strengthen retention messaging and support around {joined}" if joined else "Strengthen post-purchase support"
        return f"Improve top-of-funnel education around {joined}" if joined else "Improve customer education"

    @staticmethod
    def _pain_point_suggested_response(code: str, stage: str) -> str:
        mapping = {
            "price_markup": "Add transparent pricing, MSRP guidance, and anti-markup education",
            "dealer_friction": "Create dealer-playbook content and negotiation guidance",
            "inventory_wait": "Improve availability visibility, order tracking, and substitute recommendations",
            "financing_affordability": "Add payment calculators, APR explainers, and lease-vs-buy tools",
            "tco_uncertainty": "Provide incentive explainers and cost-of-ownership calculators",
            "charging_access": "Offer charging eligibility guidance, apartment/home charging education, and range-fit tools",
            "reliability_risk": "Strengthen long-term reliability proof, warranty clarity, and maintenance education",
            "feature_fit": "Create trim-level comparison and use-case-based fit guidance",
            "model_confusion": "Build guided comparison flows and shortlist recommendation tools",
            "general_complexity": "Review manually and consider adding a more specific pain-point rule",
        }
        return mapping.get(code, f"Address {code} through stage-specific messaging and support in {stage}")

    # ---------------------------------------------------------------------
    # Insight helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "they", "have", "will", "their",
            "about", "into", "just", "like", "what", "your", "when", "which", "were", "them", "than",
            "then", "would", "there", "also", "more", "really", "because", "while", "could", "should",
            "over", "most", "some", "much", "very", "only", "such", "does", "been", "being", "through",
            "after", "before", "where", "topic", "users", "discussion", "explores", "including", "focus",
            "vehicle", "vehicles", "cars", "car",
        }
        return [t for t in tokens if t not in stop and len(t) > 2]

    def _extract_common_terms(self, texts: List[str], top_k: int = 10) -> List[str]:
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in self._tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [k for k, _ in ranked[:top_k]]

    @staticmethod
    def _extract_signal_snippets(text: str, max_items: int = 5) -> str:
        patterns = [
            r"\b(?:msrp|markup|dealer|dealership|loan|credit|reliability|maintenance|hybrid|ev|tax credit|wait list|inventory)\b",
        ]
        hits = []
        for pat in patterns:
            hits.extend(re.findall(pat, text.lower()))
        hits = list(dict.fromkeys(hits))[:max_items]
        return ", ".join(hits)

    @staticmethod
    def _infer_stage_friction(stage: str, text: str) -> str:
        text = text.lower()
        if stage == "Awareness":
            return "Users need category education, trusted starting points, and simpler option framing"
        if stage == "Consideration":
            if "reliability" in text or "maintenance" in text:
                return "Shortlist decisions are constrained by perceived reliability and ownership tradeoffs"
            return "Users struggle to compare alternatives across too many overlapping attributes"
        if stage == "Evaluation":
            if "loan" in text or "payment" in text or "credit" in text:
                return "Economic feasibility and financing readiness are key blockers"
            return "Users need proof on total cost, incentives, and long-term fit"
        if stage == "Purchase":
            return "Dealer friction, markups, low inventory, and negotiation complexity block conversion"
        if stage == "Ownership":
            return "Post-purchase satisfaction depends on service, maintenance, and long-run cost clarity"
        return "Topic spans multiple stages and likely needs manual interpretation"

    @staticmethod
    def _infer_business_implication(stage: str, friction: str, key_signals: List[str]) -> str:
        signals = ", ".join(key_signals[:4])
        if stage == "Awareness":
            return f"Invest in educational content and entry-level segmentation; strongest cues: {signals}"
        if stage == "Consideration":
            return f"Provide side-by-side comparison tools and clearer differentiation; strongest cues: {signals}"
        if stage == "Evaluation":
            return f"Support calculators, financing guidance, and incentive explainers; strongest cues: {signals}"
        if stage == "Purchase":
            return f"Reduce conversion leakage by addressing price transparency and dealer friction; strongest cues: {signals}"
        if stage == "Ownership":
            return f"Strengthen retention through service communication and cost-of-ownership education; strongest cues: {signals}"
        return friction

    @staticmethod
    def _infer_customer_need(text: str) -> str:
        text = text.lower()
        if any(x in text for x in ["compare", "vs", "versus"]):
            return "Customer needs comparative clarity across options"
        if any(x in text for x in ["loan", "payment", "credit", "apr"]):
            return "Customer needs affordability and financing clarity"
        if any(x in text for x in ["dealer", "markup", "msrp", "inventory"]):
            return "Customer needs transparent transaction conditions"
        if any(x in text for x in ["reliability", "maintenance", "service", "repair"]):
            return "Customer needs confidence in long-term ownership outcomes"
        return "Customer needs clearer decision support"

    @staticmethod
    def _infer_purchase_barrier(text: str) -> str:
        text = text.lower()
        if "markup" in text or "market adjustment" in text:
            return "Excess dealer markup"
        if "wait list" in text or "allocation" in text or "inventory" in text:
            return "Limited supply / availability"
        if "loan" in text or "credit" in text or "payment" in text:
            return "Financing uncertainty"
        if "reliability" in text or "maintenance" in text:
            return "Performance or ownership risk concerns"
        return "General decision complexity"

    @staticmethod
    def _infer_suggested_action(stage: str, text: str) -> str:
        if stage == "Awareness":
            return "Create top-of-funnel explainers, buyer guides, and category entry content"
        if stage == "Consideration":
            return "Build comparison pages and feature/value differentiation messaging"
        if stage == "Evaluation":
            return "Add TCO calculators, financing FAQs, and incentive estimators"
        if stage == "Purchase":
            return "Improve price transparency, inventory visibility, and negotiation support"
        if stage == "Ownership":
            return "Use after-sales service messaging and lifecycle support content"
        return "Review manually and refine stage taxonomy"

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------
    def export_results(self, output_path: str) -> None:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before export_results().")

        stage_summary = self.stage_summary_df if self.stage_summary_df is not None else self.build_stage_summary()
        stage_insights = self.generate_stage_insights(top_n_topics=3)
        topic_deep_dive = self.build_topic_deep_dive()
        pain_point_table = self.pain_point_df if self.pain_point_df is not None else self.build_pain_point_table()
        pain_point_summary = self.pain_point_stage_summary_df if self.pain_point_stage_summary_df is not None else self.summarize_pain_points_by_stage()
        pain_point_insights = self.generate_pain_point_insights()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            self.stage_df.to_excel(writer, sheet_name="topic_stage_mapping", index=False)
            stage_summary.to_excel(writer, sheet_name="stage_summary", index=False)
            stage_insights.to_excel(writer, sheet_name="stage_insights", index=False)
            topic_deep_dive.to_excel(writer, sheet_name="topic_deep_dive", index=False)
            pain_point_table.to_excel(writer, sheet_name="pain_point_table", index=False)
            pain_point_summary.to_excel(writer, sheet_name="pain_point_summary", index=False)
            pain_point_insights.to_excel(writer, sheet_name="pain_point_insights", index=False)

    # ---------------------------------------------------------------------
    # Convenience pipeline
    # ---------------------------------------------------------------------
    def run_full_analysis(
        self,
        csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        exclude_outliers: bool = True,
        top_n_topics: int = 3,
        top_k_pain_points_per_topic: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        if csv_path is not None:
            self.load_csv(csv_path)
        elif df is not None:
            self.load_dataframe(df)
        elif self.df is None:
            raise ValueError("Provide csv_path, df, or load data first.")

        stage_df = self.assign_funnel_stages(exclude_outliers=exclude_outliers)
        stage_summary = self.build_stage_summary()
        stage_insights = self.generate_stage_insights(top_n_topics=top_n_topics)
        topic_deep_dive = self.build_topic_deep_dive()
        pain_point_table = self.build_pain_point_table(top_k_per_topic=top_k_pain_points_per_topic)
        pain_point_summary = self.summarize_pain_points_by_stage()
        pain_point_insights = self.generate_pain_point_insights()

        return {
            "topic_stage_mapping": stage_df,
            "stage_summary": stage_summary,
            "stage_insights": stage_insights,
            "topic_deep_dive": topic_deep_dive,
            "pain_point_table": pain_point_table,
            "pain_point_summary": pain_point_summary,
            "pain_point_insights": pain_point_insights,
        }
