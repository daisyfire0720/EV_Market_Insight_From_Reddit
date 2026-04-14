from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FunnelStageSpec:
    """Definition of a market funnel stage and its keyword / journey signals."""

    name: str
    description: str
    include_keywords: List[str] = field(default_factory=list)
    ownership_cues: List[str] = field(default_factory=list)
    deciding_cues: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class TopicFamilySpec:
    """Definition of a topic family from the PDF taxonomy."""

    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    priority: int = 0


class MarketFunnelAnalyzer:
    """
    EV market funnel analyzer aligned to the reference taxonomy PDF.

    What changed vs older versions:
    - Uses the PDF-aligned 6-stage EV funnel as the default stage system.
    - Adds an 8-family topic taxonomy so each topic can receive BOTH a stage
      and a family label.
    - Uses journey-position cues to distinguish ambiguous themes such as
      charging in Evaluation vs Onboarding.
    - Preserves the older public API where practical.

    Required minimum input:
    - topic_id
    - at least one supported text column

    Typical usage:
        analyzer = MarketFunnelAnalyzer(topic_weight_col="Count")
        outputs = analyzer.run_full_analysis(csv_path="topic_labels_llm.csv")
        analyzer.export_results("market_funnel_analysis.xlsx")
    """

    STAGE_ORDER = [
        "Awareness / Need Formation",
        "Consideration / Shortlisting",
        "Evaluation / Practical Fit",
        "Purchase / Transaction",
        "Onboarding / Early Ownership",
        "Long-Term Ownership / Retention",
        "Unclear / Cross-Stage",
    ]

    FAMILY_ORDER = [
        "EV Purchase and Vehicle Selection",
        "Charging and Energy Access",
        "Battery, Range, and Real-World Performance",
        "Cost, Incentives, and Financial Feasibility",
        "Dealer, Purchase Process, and Market Access",
        "Reliability, Service, and Ownership Experience",
        "Policy, Regulation, and Market Environment",
        "ICE / Hybrid / EV Transition Narratives",
        "General / Mixed EV Discussion",
    ]

    DEFAULT_STAGE_SPECS = [
        FunnelStageSpec(
            name="Awareness / Need Formation",
            description=(
                "Earliest discovery stage. Users ask whether an EV is worth considering "
                "and what problem or lifestyle need it might solve."
            ),
            include_keywords=[
                "first ev", "what is an ev", "should i buy an ev", "worth considering",
                "general ev curiosity", "new to ev", "ev worth it", "why buy an ev",
                "thinking about an ev", "curious about ev", "ev vs regular car",
                "beginner", "starter guide", "buying guide", "research",
            ],
            deciding_cues=["thinking about", "considering", "worth it", "beginner", "curious", "researching"],
            priority=1,
        ),
        FunnelStageSpec(
            name="Consideration / Shortlisting",
            description=(
                "Users begin narrowing options and comparing brands, models, classes, or broad alternatives."
            ),
            include_keywords=[
                "tesla vs", "ioniq 5", "model y", "best ev suv", "used bolt", "leaf",
                "hybrid vs ev", "vs", "versus", "compare", "comparison", "shortlist",
                "which ev", "which one", "best for commuting", "family car", "body class",
                "sedan", "suv", "hatchback", "trim", "awd",
            ],
            deciding_cues=["between", "shortlist", "compare", "versus", "vs", "which one"],
            priority=2,
        ),
        FunnelStageSpec(
            name="Evaluation / Practical Fit",
            description=(
                "Users test whether an EV works for their life economically and operationally."
            ),
            include_keywords=[
                "apartment charging", "street park", "range fit", "winter performance",
                "insurance", "incentives", "tax credit", "rebate", "tco", "total cost",
                "monthly payment", "apr", "loan", "lease", "home charging", "public charging",
                "can i make an ev work", "charging feasibility", "road trip", "cold weather",
                "battery degradation", "efficiency", "miles per kwh", "financial feasibility",
            ],
            deciding_cues=[
                "can i", "would an ev work", "fit my life", "practical", "feasibility",
                "before buying", "if i buy", "thinking of buying", "cost me",
            ],
            ownership_cues=[],
            priority=3,
        ),
        FunnelStageSpec(
            name="Purchase / Transaction",
            description=(
                "The discussion shifts from deciding to executing the purchase."
            ),
            include_keywords=[
                "dealer friction", "dealer", "dealership", "markup", "market adjustment",
                "adm", "financing", "lease terms", "order process", "inventory", "delivery timing",
                "delivery", "out the door", "doc fee", "msrp", "wait list", "allocation",
                "deposit", "purchase process", "buying now", "signed papers", "otd",
            ],
            deciding_cues=["order", "deposit", "buying", "purchasing", "dealer"],
            priority=4,
        ),
        FunnelStageSpec(
            name="Onboarding / Early Ownership",
            description=(
                "Immediate post-purchase adaptation stage where users learn the product and ecosystem."
            ),
            include_keywords=[
                "just bought", "i bought", "new owner", "first charging setup", "set up home charging",
                "set up charging", "app confusion", "learning features", "first service", "first week",
                "first month", "delivery day", "picked up", "owner tips", "setup the app",
                "charging app", "initial impressions", "new ev owner",
            ],
            ownership_cues=["just bought", "i bought", "new owner", "picked up", "delivery day", "first week", "first month"],
            priority=5,
        ),
        FunnelStageSpec(
            name="Long-Term Ownership / Retention",
            description=(
                "Later-stage ownership experience including satisfaction, reliability, maintenance, and replacement decisions."
            ),
            include_keywords=[
                "after two winters", "battery degradation", "maintenance", "warranty", "service center",
                "long-term satisfaction", "regret", "resale", "repair", "service delays",
                "ownership experience", "years later", "after a year", "after two years",
                "battery range seems worse", "replacement decision", "keep or sell", "degradation",
                "long term", "long-term", "over time",
            ],
            ownership_cues=["after a year", "after two years", "over time", "long-term", "owned for", "my ev now", "years later"],
            priority=6,
        ),
    ]

    DEFAULT_TOPIC_FAMILY_SPECS = [
        TopicFamilySpec(
            name="EV Purchase and Vehicle Selection",
            description="Brand/model choice, body class, fit-for-use, shortlist logic, and new vs used EV selection.",
            keywords=[
                "best ev", "which ev", "model y", "ioniq 5", "used ev", "used bolt", "leaf",
                "family car", "commuting", "vehicle selection", "recommendation", "body class",
                "sedan", "suv", "hatchback", "trim", "choice", "compare models",
            ],
            priority=1,
        ),
        TopicFamilySpec(
            name="Charging and Energy Access",
            description="Home charging, public charging, apartment access, compatibility, and convenience.",
            keywords=[
                "charging", "charger", "home charging", "public charging", "dc fast", "fast charging",
                "level 2", "apartment charging", "street park", "garage", "plug", "supercharger",
                "charging network", "energy access", "charging setup",
            ],
            priority=2,
        ),
        TopicFamilySpec(
            name="Battery, Range, and Real-World Performance",
            description="Battery health, range anxiety, winter range, efficiency, and real-world performance.",
            keywords=[
                "battery", "range", "battery degradation", "winter range", "road trip", "efficiency",
                "cold weather", "miles per kwh", "kwh", "state of charge", "highway efficiency",
                "real-world performance",
            ],
            priority=3,
        ),
        TopicFamilySpec(
            name="Cost, Incentives, and Financial Feasibility",
            description="Price, payment, insurance, tax credits, rebates, financing, leasing, and TCO.",
            keywords=[
                "price", "payment", "monthly payment", "apr", "loan", "lease", "insurance",
                "tax credit", "rebate", "incentive", "tco", "total cost", "financial feasibility",
                "afford", "cost",
            ],
            priority=4,
        ),
        TopicFamilySpec(
            name="Dealer, Purchase Process, and Market Access",
            description="Dealer interactions, order flow, wait times, markup, inventory access, and transaction friction.",
            keywords=[
                "dealer", "dealership", "markup", "market adjustment", "inventory", "wait list",
                "allocation", "delivery timing", "delivery", "doc fee", "out the door", "msrp",
                "order process", "market access", "purchase process", "deposit",
            ],
            priority=5,
        ),
        TopicFamilySpec(
            name="Reliability, Service, and Ownership Experience",
            description="Maintenance, repair, service quality, warranty, software issues after purchase, and ownership sentiment.",
            keywords=[
                "reliability", "service", "repair", "maintenance", "warranty", "ownership experience",
                "service center", "service delays", "quality", "problem", "issue", "app confusion",
                "software update", "ownership sentiment",
            ],
            priority=6,
        ),
        TopicFamilySpec(
            name="Policy, Regulation, and Market Environment",
            description="Government policy, eligibility rules, infrastructure policy, and market environment discourse.",
            keywords=[
                "policy", "regulation", "eligibility", "rule changes", "federal tax credit",
                "state regulation", "emissions regulation", "charging policy", "government",
                "market environment", "mandate",
            ],
            priority=7,
        ),
        TopicFamilySpec(
            name="ICE / Hybrid / EV Transition Narratives",
            description="Transition stories, skepticism, education, habit shifts, and cross-powertrain comparison.",
            keywords=[
                "ice", "hybrid", "ev vs ice", "hybrid vs ev", "switching from ice", "transition",
                "skeptic", "conversion concerns", "habit shifts", "gas car", "regular car",
            ],
            priority=8,
        ),
    ]

    PAIN_POINT_RULES = {
        "price_markup": {
            "name": "Price Markup / Overpricing",
            "keywords": ["markup", "market adjustment", "adm", "over msrp", "doc fee", "overpriced"],
            "stage_hint": ["Purchase / Transaction", "Evaluation / Practical Fit"],
            "severity": 5,
        },
        "dealer_friction": {
            "name": "Dealer Friction / Trust Issues",
            "keywords": ["dealer", "dealership", "salesperson", "lied", "pressure", "bait", "upsell"],
            "stage_hint": ["Purchase / Transaction"],
            "severity": 4,
        },
        "inventory_wait": {
            "name": "Inventory / Wait-Time Constraints",
            "keywords": ["inventory", "allocation", "wait list", "delivery", "availability", "backorder"],
            "stage_hint": ["Purchase / Transaction"],
            "severity": 4,
        },
        "charging_access": {
            "name": "Charging Access / Feasibility",
            "keywords": ["charging", "charger", "home charging", "apartment", "street park", "garage", "level 2"],
            "stage_hint": ["Evaluation / Practical Fit", "Onboarding / Early Ownership"],
            "severity": 4,
        },
        "tco_uncertainty": {
            "name": "Cost / Incentive Uncertainty",
            "keywords": ["tax credit", "rebate", "incentive", "insurance", "tco", "total cost", "payment", "apr", "loan", "lease"],
            "stage_hint": ["Evaluation / Practical Fit", "Purchase / Transaction"],
            "severity": 4,
        },
        "range_fit": {
            "name": "Range / Practical-Fit Concern",
            "keywords": ["range", "road trip", "winter", "cold weather", "battery degradation", "efficiency"],
            "stage_hint": ["Evaluation / Practical Fit", "Long-Term Ownership / Retention"],
            "severity": 4,
        },
        "reliability_service": {
            "name": "Reliability / Service Friction",
            "keywords": ["reliability", "service", "repair", "maintenance", "warranty", "issue", "problem"],
            "stage_hint": ["Onboarding / Early Ownership", "Long-Term Ownership / Retention"],
            "severity": 4,
        },
        "comparison_complexity": {
            "name": "Comparison Overload / Decision Complexity",
            "keywords": ["vs", "versus", "compare", "between", "which one", "shortlist"],
            "stage_hint": ["Consideration / Shortlisting", "Awareness / Need Formation"],
            "severity": 3,
        },
    }

    NEGATIVE_SIGNAL_WORDS = {
        "problem", "issue", "concern", "worry", "worried", "bad", "hard", "difficult", "friction",
        "expensive", "overpriced", "markup", "delay", "limited", "confusing", "uncertain", "lied",
        "pressure", "regret", "risk", "stuck", "cannot", "can't", "worse", "unavailable", "degradation",
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
        topic_family_specs: Optional[List[TopicFamilySpec]] = None,
        topic_weight_col: Optional[str] = None,
        outlier_topic_id: int = -1,
    ) -> None:
        self.stage_specs = sorted(stage_specs or self.DEFAULT_STAGE_SPECS, key=lambda x: x.priority)
        self.topic_family_specs = sorted(topic_family_specs or self.DEFAULT_TOPIC_FAMILY_SPECS, key=lambda x: x.priority)
        self.topic_weight_col = topic_weight_col
        self.outlier_topic_id = outlier_topic_id

        self.df: Optional[pd.DataFrame] = None
        self.stage_df: Optional[pd.DataFrame] = None
        self.stage_summary_df: Optional[pd.DataFrame] = None
        self.family_summary_df: Optional[pd.DataFrame] = None
        self.pain_point_df: Optional[pd.DataFrame] = None
        self.pain_point_stage_summary_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------
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
            raise ValueError(f"topic_weight_col='{self.topic_weight_col}' not found in dataframe columns.")

    def _prepare_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.TEXT_PRIORITY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["topic_weight"] = (
            pd.to_numeric(df[self.topic_weight_col], errors="coerce").fillna(0.0)
            if self.topic_weight_col is not None else 1.0
        )
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

    # ------------------------------------------------------------------
    # Stage / family assignment
    # ------------------------------------------------------------------
    def assign_funnel_stages(self, exclude_outliers: bool = True) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Use load_csv() or load_dataframe() first.")

        df = self.df.copy()
        if exclude_outliers:
            df = df.loc[~df["is_outlier"]].copy()

        stage_assignments = df.apply(self._score_row_to_stage, axis=1, result_type="expand")
        stage_assignments.columns = [
            "funnel_stage", "stage_confidence", "stage_match_score", "stage_reason", "stage_keyword_hits",
        ]
        df = pd.concat([df.reset_index(drop=True), stage_assignments.reset_index(drop=True)], axis=1)

        family_assignments = df.apply(self._score_row_to_family, axis=1, result_type="expand")
        family_assignments.columns = [
            "topic_family", "topic_family_confidence", "topic_family_score", "topic_family_reason", "topic_family_keyword_hits",
        ]
        df = pd.concat([df.reset_index(drop=True), family_assignments.reset_index(drop=True)], axis=1)

        df["topic_display"] = df.apply(self._build_topic_display, axis=1)
        df["primary_signal"] = df["stage_reason"].str.split(";").str[0].fillna("")
        df["stage_sort_order"] = df["funnel_stage"].map({name: i for i, name in enumerate(self.STAGE_ORDER)})
        df["family_sort_order"] = df["topic_family"].map({name: i for i, name in enumerate(self.FAMILY_ORDER)})

        self.stage_df = df.sort_values(
            ["stage_sort_order", "family_sort_order", "stage_confidence", "topic_weight"],
            ascending=[True, True, False, False],
            na_position="last",
        ).reset_index(drop=True)
        return self.stage_df.copy()

    def _score_row_to_stage(self, row: pd.Series) -> Tuple[str, float, float, str, str]:
        text = row["combined_text"]
        stage_scores: List[Tuple[str, float, List[str], str]] = []
        ownership_state = self._infer_ownership_state(text)

        for spec in self.stage_specs:
            score = 0.0
            hits: List[str] = []

            include_hits = [kw for kw in spec.include_keywords if kw in text]
            deciding_hits = [kw for kw in spec.deciding_cues if kw in text]
            ownership_hits = [kw for kw in spec.ownership_cues if kw in text]
            exclude_hits = [kw for kw in spec.exclude_keywords if kw in text]

            score += 1.0 * len(include_hits)
            score += 0.5 * len(deciding_hits)
            score += 0.8 * len(ownership_hits)
            score -= 1.0 * len(exclude_hits)
            hits.extend(include_hits + deciding_hits + ownership_hits)

            label_text = self._clean_text(row.get("topic_label_llm", ""))
            summary_text = self._clean_text(row.get("topic_summary_llm", ""))
            centroid_text = self._clean_text(row.get("rep_doc_centroid", ""))

            score += 1.5 * sum(1 for kw in spec.include_keywords if kw in label_text)
            score += 0.9 * sum(1 for kw in spec.include_keywords if kw in summary_text)
            score += 0.5 * sum(1 for kw in spec.include_keywords if kw in centroid_text)

            # Journey-position bump for ambiguous EV topics such as charging / battery.
            if spec.name == "Evaluation / Practical Fit" and ownership_state == "pre_purchase":
                score += 1.25
            if spec.name == "Onboarding / Early Ownership" and ownership_state == "new_owner":
                score += 1.75
            if spec.name == "Long-Term Ownership / Retention" and ownership_state == "long_term_owner":
                score += 1.75
            if spec.name == "Purchase / Transaction" and any(x in text for x in ["dealer", "markup", "inventory", "doc fee", "delivery", "deposit"]):
                score += 1.25
            if spec.name == "Consideration / Shortlisting" and any(x in text for x in ["vs", "versus", "compare", "which one", "shortlist"]):
                score += 1.0
            if spec.name == "Awareness / Need Formation" and any(x in text for x in ["new to ev", "should i buy an ev", "worth considering", "curious about ev"]):
                score += 1.25

            stage_scores.append((spec.name, score, sorted(set(hits)), spec.description))

        stage_scores = sorted(stage_scores, key=lambda x: x[1], reverse=True)
        best_name, best_score, best_hits, _ = stage_scores[0]
        second_score = stage_scores[1][1] if len(stage_scores) > 1 else 0.0

        if best_score <= 0:
            return (
                "Unclear / Cross-Stage",
                0.25,
                float(best_score),
                "No strong EV journey evidence; topic appears broad, weakly labeled, or mixed",
                "",
            )

        confidence = self._score_to_confidence(best_score, second_score)
        reason = self._build_stage_reason(best_name, best_hits, best_score, second_score, ownership_state)
        keyword_hits = ", ".join(best_hits[:15])
        return best_name, confidence, float(best_score), reason, keyword_hits

    def _score_row_to_family(self, row: pd.Series) -> Tuple[str, float, float, str, str]:
        text = row["combined_text"]
        family_scores: List[Tuple[str, float, List[str], str]] = []
        for spec in self.topic_family_specs:
            hits = [kw for kw in spec.keywords if kw in text]
            score = float(len(hits))

            label_text = self._clean_text(row.get("topic_label_llm", ""))
            summary_text = self._clean_text(row.get("topic_summary_llm", ""))
            score += 1.25 * sum(1 for kw in spec.keywords if kw in label_text)
            score += 0.75 * sum(1 for kw in spec.keywords if kw in summary_text)
            family_scores.append((spec.name, score, sorted(set(hits)), spec.description))

        family_scores = sorted(family_scores, key=lambda x: x[1], reverse=True)
        best_name, best_score, best_hits, _ = family_scores[0]
        second_score = family_scores[1][1] if len(family_scores) > 1 else 0.0

        if best_score <= 0:
            return (
                "General / Mixed EV Discussion",
                0.25,
                float(best_score),
                "No strong subject-matter evidence for a single topic family",
                "",
            )

        confidence = self._score_to_confidence(best_score, second_score)
        reason = (
            f"Assigned to {best_name} due to subject-matter signals: {', '.join(best_hits[:8])}; "
            f"score={best_score:.2f}, margin_vs_next={best_score - second_score:.2f}"
        )
        return best_name, confidence, float(best_score), reason, ", ".join(best_hits[:15])

    @staticmethod
    def _infer_ownership_state(text: str) -> str:
        text = text.lower()
        long_term_patterns = [
            "after a year", "after two years", "after two winters", "over time", "long-term", "long term",
            "owned for", "years later", "battery degradation", "resale", "keep or sell",
        ]
        new_owner_patterns = [
            "just bought", "i bought", "new owner", "picked up", "delivery day", "first week",
            "first month", "set up home charging", "setup the app", "learning features",
        ]
        pre_purchase_patterns = [
            "should i buy", "thinking about", "considering", "can i make an ev work", "would an ev work",
            "fit my life", "if i buy", "before buying", "compare", "vs", "versus", "which ev",
        ]

        if any(p in text for p in long_term_patterns):
            return "long_term_owner"
        if any(p in text for p in new_owner_patterns):
            return "new_owner"
        if any(p in text for p in pre_purchase_patterns):
            return "pre_purchase"
        return "unknown"

    @staticmethod
    def _score_to_confidence(best_score: float, second_score: float) -> float:
        margin = best_score - second_score
        raw = 0.42 + 0.08 * best_score + 0.10 * margin
        return float(max(0.0, min(0.99, raw)))

    @staticmethod
    def _build_stage_reason(stage_name: str, hits: List[str], best_score: float, second_score: float, ownership_state: str) -> str:
        hit_text = ", ".join(hits[:8]) if hits else "overall topic wording"
        ownership_text = f"journey_state={ownership_state}"
        return (
            f"Assigned to {stage_name} due to signals: {hit_text}; {ownership_text}; "
            f"score={best_score:.2f}, margin_vs_next={best_score - second_score:.2f}"
        )

    @staticmethod
    def _build_topic_display(row: pd.Series) -> str:
        label = row.get("topic_label_llm", "") or row.get("topic_label_refined", "")
        summary = row.get("topic_summary_llm", "")
        summary = summary[:180] + "..." if len(summary) > 180 else summary
        return f"{label} | {summary}".strip(" |")

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
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
        grouped["dominant_topic_family"] = grouped["funnel_stage"].map(self._dominant_family_by_stage(df))
        grouped["stage_sort_order"] = grouped["funnel_stage"].map({name: i for i, name in enumerate(self.STAGE_ORDER)})
        self.stage_summary_df = grouped.sort_values("stage_sort_order").drop(columns="stage_sort_order").reset_index(drop=True)
        return self.stage_summary_df.copy()

    def build_topic_family_summary(self) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before build_topic_family_summary().")

        df = self.stage_df.copy()
        total_weight = df["topic_weight"].sum()
        grouped = (
            df.groupby("topic_family", dropna=False)
            .agg(
                n_topics=("topic_id", "count"),
                weighted_topics=("topic_weight", "sum"),
                avg_family_confidence=("topic_family_confidence", "mean"),
                avg_negative_signal=("negative_signal_score", "mean"),
                avg_question_signal=("question_signal_score", "mean"),
            )
            .reset_index()
        )
        grouped["weighted_share"] = np.where(total_weight > 0, grouped["weighted_topics"] / total_weight, np.nan)
        grouped["family_sort_order"] = grouped["topic_family"].map({name: i for i, name in enumerate(self.FAMILY_ORDER)})
        self.family_summary_df = grouped.sort_values(["family_sort_order", "weighted_topics"], ascending=[True, False]).drop(columns="family_sort_order").reset_index(drop=True)
        return self.family_summary_df.copy()

    @staticmethod
    def _stage_role_lookup() -> Dict[str, str]:
        return {
            "Awareness / Need Formation": "Earliest discovery and problem framing",
            "Consideration / Shortlisting": "Option narrowing and shortlist building",
            "Evaluation / Practical Fit": "Economic and operational fit validation",
            "Purchase / Transaction": "Dealer execution and conversion friction",
            "Onboarding / Early Ownership": "Immediate post-purchase learning and setup",
            "Long-Term Ownership / Retention": "Retention, reliability, and lifecycle experience",
            "Unclear / Cross-Stage": "Mixed or weak journey evidence",
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

    @staticmethod
    def _dominant_family_by_stage(df: pd.DataFrame) -> Dict[str, str]:
        if df.empty or "topic_family" not in df.columns:
            return {}
        dominant = (
            df.groupby(["funnel_stage", "topic_family"])["topic_weight"]
            .sum()
            .reset_index()
            .sort_values(["funnel_stage", "topic_weight"], ascending=[True, False])
            .drop_duplicates("funnel_stage")
        )
        return dict(zip(dominant["funnel_stage"], dominant["topic_family"]))

    def generate_stage_insights(self, top_n_topics: int = 3) -> pd.DataFrame:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before generate_stage_insights().")

        rows = []
        for stage, g in self.stage_df.groupby("funnel_stage", dropna=False):
            g = g.sort_values(["topic_weight", "stage_confidence"], ascending=[False, False])
            top_topics = g.head(top_n_topics)
            topic_list = " || ".join(top_topics["topic_label_llm"].fillna(top_topics["topic_label_refined"]).astype(str))
            key_signals = self._extract_common_terms(top_topics["combined_text"].tolist(), top_k=10)
            dominant_family_mix = self._top_family_mix(top_topics)
            friction = self._infer_stage_friction(stage, " ".join(top_topics["combined_text"].tolist()))
            implication = self._infer_business_implication(stage, key_signals)

            rows.append(
                {
                    "funnel_stage": stage,
                    "top_topics": topic_list,
                    "common_signals": ", ".join(key_signals),
                    "dominant_topic_families": dominant_family_mix,
                    "core_friction_or_need": friction,
                    "business_implication": implication,
                    "n_topics_in_stage": len(g),
                    "weighted_topics": g["topic_weight"].sum(),
                    "avg_confidence": g["stage_confidence"].mean(),
                    "avg_negative_signal": g["negative_signal_score"].mean(),
                }
            )

        out = pd.DataFrame(rows)
        out["stage_sort_order"] = out["funnel_stage"].map({name: i for i, name in enumerate(self.STAGE_ORDER)})
        return out.sort_values(["stage_sort_order", "weighted_topics"], ascending=[True, False]).drop(columns="stage_sort_order").reset_index(drop=True)

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
                    "topic_family": row.get("topic_family", ""),
                    "topic_weight": row["topic_weight"],
                    "stage_confidence": row["stage_confidence"],
                    "topic_family_confidence": row.get("topic_family_confidence", np.nan),
                    "decision_signals": self._extract_signal_snippets(text),
                    "customer_need": self._infer_customer_need(text),
                    "purchase_barrier": self._infer_purchase_barrier(text),
                    "primary_pain_point": primary_pain["pain_point_name"],
                    "primary_pain_score": primary_pain["pain_score"],
                    "suggested_action": self._infer_suggested_action(row["funnel_stage"], row.get("topic_family", ""), text),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pain-point analyzer
    # ------------------------------------------------------------------
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
                        "topic_family": row.get("topic_family", ""),
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
        grouped["stage_sort_order"] = grouped["funnel_stage"].map({name: i for i, name in enumerate(self.STAGE_ORDER)})
        grouped = grouped.sort_values(["stage_sort_order", "opportunity_score"], ascending=[True, False]).drop(columns="stage_sort_order").reset_index(drop=True)
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
        out = pd.DataFrame(rows)
        out["stage_sort_order"] = out["funnel_stage"].map({name: i for i, name in enumerate(self.STAGE_ORDER)})
        return out.sort_values(["stage_sort_order", "stage_opportunity_score"], ascending=[True, False]).drop(columns="stage_sort_order").reset_index(drop=True)

    def _score_pain_points_for_text(self, text: str, stage: str) -> List[Dict[str, object]]:
        lowered = self._clean_text(text)
        scored = []
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
        mapping = {
            "Awareness / Need Formation": "Improve entry-level EV education",
            "Consideration / Shortlisting": "Clarify shortlist and comparison tradeoffs",
            "Evaluation / Practical Fit": "Improve operational-fit and cost proof",
            "Purchase / Transaction": "Reduce purchase friction and leakage",
            "Onboarding / Early Ownership": "Smooth first-month setup and learning",
            "Long-Term Ownership / Retention": "Protect retention with service and reliability support",
        }
        prefix = mapping.get(stage, "Review mixed-stage friction")
        return f"{prefix}: {joined}" if joined else prefix

    @staticmethod
    def _pain_point_suggested_response(code: str, stage: str) -> str:
        mapping = {
            "price_markup": "Add transparent pricing, MSRP guidance, and anti-markup education",
            "dealer_friction": "Create dealer-playbook content and negotiation guidance",
            "inventory_wait": "Improve availability visibility, order tracking, and substitute recommendations",
            "charging_access": "Offer apartment/home charging guidance and setup tools",
            "tco_uncertainty": "Provide incentive explainers and total-cost calculators",
            "range_fit": "Show realistic range scenarios and winter / road-trip guidance",
            "reliability_service": "Clarify warranty, service expectations, and maintenance realities",
            "comparison_complexity": "Build guided comparison flows and shortlist recommendation tools",
            "general_complexity": "Review manually and consider adding a more specific pain-point rule",
        }
        return mapping.get(code, f"Address {code} with stage-specific support in {stage}")

    # ------------------------------------------------------------------
    # Insight helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "they", "have", "will", "their",
            "about", "into", "just", "like", "what", "your", "when", "which", "were", "them", "than",
            "then", "would", "there", "also", "more", "really", "because", "while", "could", "should",
            "over", "most", "some", "much", "very", "only", "such", "does", "been", "being", "through",
            "after", "before", "where", "topic", "users", "discussion", "explores", "including", "focus",
            "vehicle", "vehicles", "cars", "car", "ev", "evs",
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
    def _extract_signal_snippets(text: str, max_items: int = 7) -> str:
        patterns = [
            r"\b(?:markup|dealer|dealership|loan|credit|reliability|maintenance|hybrid|tax credit|wait list|inventory|charging|battery|range|warranty)\b",
        ]
        hits: List[str] = []
        for pat in patterns:
            hits.extend(re.findall(pat, text.lower()))
        hits = list(dict.fromkeys(hits))[:max_items]
        return ", ".join(hits)

    @staticmethod
    def _top_family_mix(df: pd.DataFrame, top_n: int = 3) -> str:
        if df.empty or "topic_family" not in df.columns:
            return ""
        mix = (
            df.groupby("topic_family")["topic_weight"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        return " || ".join(f"{idx} ({val:.1f})" for idx, val in mix.items())

    @staticmethod
    def _infer_stage_friction(stage: str, text: str) -> str:
        text = text.lower()
        if stage == "Awareness / Need Formation":
            return "Users need EV basics, relevance framing, and trusted starting points"
        if stage == "Consideration / Shortlisting":
            return "Users are narrowing options and need cleaner side-by-side differentiation"
        if stage == "Evaluation / Practical Fit":
            return "Users need proof that EV ownership works economically and operationally in their situation"
        if stage == "Purchase / Transaction":
            return "Dealer friction, markup, financing, inventory, and delivery execution are the main blockers"
        if stage == "Onboarding / Early Ownership":
            return "New owners need help with setup, charging routines, apps, and early feature learning"
        if stage == "Long-Term Ownership / Retention":
            return "Long-run satisfaction depends on reliability, service quality, battery/range confidence, and resale"
        return "Topic spans multiple stages and likely needs manual review"

    @staticmethod
    def _infer_business_implication(stage: str, key_signals: List[str]) -> str:
        signals = ", ".join(key_signals[:4])
        if stage == "Awareness / Need Formation":
            return f"Invest in introductory education and EV relevance framing; strongest cues: {signals}"
        if stage == "Consideration / Shortlisting":
            return f"Provide comparison tools and shortlist guidance; strongest cues: {signals}"
        if stage == "Evaluation / Practical Fit":
            return f"Support fit validation with charging, cost, and range calculators; strongest cues: {signals}"
        if stage == "Purchase / Transaction":
            return f"Reduce conversion leakage with dealer, pricing, and delivery transparency; strongest cues: {signals}"
        if stage == "Onboarding / Early Ownership":
            return f"Improve first-30-day onboarding content and setup flows; strongest cues: {signals}"
        if stage == "Long-Term Ownership / Retention":
            return f"Protect retention with reliability proof and support content; strongest cues: {signals}"
        return signals

    @staticmethod
    def _infer_customer_need(text: str) -> str:
        text = text.lower()
        if any(x in text for x in ["compare", "vs", "versus", "shortlist"]):
            return "Customer needs comparative clarity across EV options"
        if any(x in text for x in ["apartment charging", "street park", "home charging", "public charging"]):
            return "Customer needs charging feasibility clarity"
        if any(x in text for x in ["loan", "payment", "credit", "apr", "lease", "tax credit", "insurance"]):
            return "Customer needs affordability and incentive clarity"
        if any(x in text for x in ["dealer", "markup", "inventory", "delivery", "doc fee"]):
            return "Customer needs transparent transaction conditions"
        if any(x in text for x in ["reliability", "maintenance", "service", "repair", "warranty"]):
            return "Customer needs confidence in long-term ownership outcomes"
        return "Customer needs clearer EV decision support"

    @staticmethod
    def _infer_purchase_barrier(text: str) -> str:
        text = text.lower()
        if any(x in text for x in ["markup", "market adjustment", "doc fee"]):
            return "Dealer overpricing / fee friction"
        if any(x in text for x in ["wait list", "allocation", "inventory", "delivery"]):
            return "Supply / delivery constraint"
        if any(x in text for x in ["loan", "credit", "payment", "apr", "lease"]):
            return "Financing / affordability uncertainty"
        if any(x in text for x in ["apartment charging", "street park", "home charging", "public charging"]):
            return "Charging feasibility uncertainty"
        if any(x in text for x in ["range", "winter", "battery degradation", "insurance"]):
            return "Practical-fit / cost uncertainty"
        if any(x in text for x in ["reliability", "maintenance", "warranty", "service"]):
            return "Ownership risk concern"
        return "General decision complexity"

    @staticmethod
    def _infer_suggested_action(stage: str, family: str, text: str) -> str:
        if stage == "Awareness / Need Formation":
            return "Create EV 101 explainers, entry guides, and simple myth-vs-reality content"
        if stage == "Consideration / Shortlisting":
            return "Build side-by-side compare pages and use-case-based recommendations"
        if stage == "Evaluation / Practical Fit":
            return "Add charging-fit, range-fit, TCO, and incentive calculators"
        if stage == "Purchase / Transaction":
            return "Improve pricing transparency, dealer guidance, and inventory / delivery visibility"
        if stage == "Onboarding / Early Ownership":
            return "Create first-week setup checklists, charging setup guides, and app walkthroughs"
        if stage == "Long-Term Ownership / Retention":
            return "Publish service, warranty, degradation, and long-term cost guidance"
        return f"Review manually and refine rules for family={family}"

    # ------------------------------------------------------------------
    # Export / pipeline
    # ------------------------------------------------------------------
    def export_results(self, output_path: str) -> None:
        if self.stage_df is None:
            raise ValueError("Run assign_funnel_stages() before export_results().")

        stage_summary = self.stage_summary_df if self.stage_summary_df is not None else self.build_stage_summary()
        family_summary = self.family_summary_df if self.family_summary_df is not None else self.build_topic_family_summary()
        stage_insights = self.generate_stage_insights(top_n_topics=3)
        topic_deep_dive = self.build_topic_deep_dive()
        pain_point_table = self.pain_point_df if self.pain_point_df is not None else self.build_pain_point_table()
        pain_point_summary = self.pain_point_stage_summary_df if self.pain_point_stage_summary_df is not None else self.summarize_pain_points_by_stage()
        pain_point_insights = self.generate_pain_point_insights()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            self.stage_df.to_excel(writer, sheet_name="topic_stage_mapping", index=False)
            stage_summary.to_excel(writer, sheet_name="stage_summary", index=False)
            family_summary.to_excel(writer, sheet_name="topic_family_summary", index=False)
            stage_insights.to_excel(writer, sheet_name="stage_insights", index=False)
            topic_deep_dive.to_excel(writer, sheet_name="topic_deep_dive", index=False)
            pain_point_table.to_excel(writer, sheet_name="pain_point_table", index=False)
            pain_point_summary.to_excel(writer, sheet_name="pain_point_summary", index=False)
            pain_point_insights.to_excel(writer, sheet_name="pain_point_insights", index=False)

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
        family_summary = self.build_topic_family_summary()
        stage_insights = self.generate_stage_insights(top_n_topics=top_n_topics)
        topic_deep_dive = self.build_topic_deep_dive()
        pain_point_table = self.build_pain_point_table(top_k_per_topic=top_k_pain_points_per_topic)
        pain_point_summary = self.summarize_pain_points_by_stage()
        pain_point_insights = self.generate_pain_point_insights()

        return {
            "topic_stage_mapping": stage_df,
            "stage_summary": stage_summary,
            "topic_family_summary": family_summary,
            "stage_insights": stage_insights,
            "topic_deep_dive": topic_deep_dive,
            "pain_point_table": pain_point_table,
            "pain_point_summary": pain_point_summary,
            "pain_point_insights": pain_point_insights,
        }


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {"topic_id": 1, "topic_label_llm": "Should I buy an EV for commuting?", "topic_summary_llm": "New to EVs and wondering if an EV is worth considering for my daily commute.", "Count": 10},
            {"topic_id": 2, "topic_label_llm": "Model Y vs Ioniq 5 for family car", "topic_summary_llm": "Comparing shortlisted EV SUVs for comfort, cargo, and commuting.", "Count": 8},
            {"topic_id": 3, "topic_label_llm": "Can I make an EV work if I live in an apartment?", "topic_summary_llm": "Charging feasibility, street parking, insurance, and tax credit questions before buying.", "Count": 12},
            {"topic_id": 4, "topic_label_llm": "Dealer added markup and delivery keeps slipping", "topic_summary_llm": "Inventory, markup, and dealership friction during purchase.", "Count": 6},
            {"topic_id": 5, "topic_label_llm": "I just bought a Bolt - how do I set up charging and the app?", "topic_summary_llm": "New owner asking about first charging setup and app confusion.", "Count": 5},
            {"topic_id": 6, "topic_label_llm": "After two winters my EV range seems worse", "topic_summary_llm": "Battery degradation and long-term ownership concerns after owning the car for years.", "Count": 4},
        ]
    )
    analyzer = MarketFunnelAnalyzer(topic_weight_col="Count")
    results = analyzer.run_full_analysis(df=sample)
    for name, df_out in results.items():
        print(f"\n=== {name} ===")
        print(df_out.head())
