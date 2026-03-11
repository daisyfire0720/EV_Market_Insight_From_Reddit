from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass
class TopicRefineConfig:
    top_n_topics_overall: int = 20
    top_n_topics_trend: int = 10
    min_topic_docs_for_engagement: int = 20
    confidence_threshold: Optional[float] = None
    exclude_outlier_topic: bool = True
    topic_label_max_words: int = 8
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

    def __init__(self, config: Optional[TopicRefineConfig] = None):
        self.config = config or TopicRefineConfig()
        self.stopwords = set(ENGLISH_STOP_WORDS).union(self.config.extra_stopwords)
        self.generic_tokens = {
            self._simple_singular(w.lower()) for w in self.config.label_generic_words
        }

    @staticmethod
    def _safe_list_parse(value: Any) -> List[str]:
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        if isinstance(value, (list, tuple, set, np.ndarray)):
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
        return all(t in self.generic_tokens for t in toks)

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

        keybert_terms = [self._normalize_phrase(x) for x in self._safe_list_parse(row.get("KeyBERT"))[:10]]
        mmr_terms = [self._normalize_phrase(x) for x in self._safe_list_parse(row.get("MMR"))[:10]]
        candidates = self._candidate_phrases(row)

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
        out["topic_label_refined"] = out.apply(self.choose_label, axis=1)
        out["topic_keywords_clean"] = out["Representation"].apply(
            lambda x: ", ".join(self._safe_list_parse(x)[:8])
        )
        out["topic_keybert_clean"] = out["KeyBERT"].apply(
            lambda x: ", ".join(self._safe_list_parse(x)[:8])
        )
        return out


class RepresentativeDocCleaner:
    """Normalize representative-doc text artifacts in BERTopic topic tables."""

    _ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
    _WHITESPACE_RE = re.compile(r"\s+")

    _UNICODE_PUNCT_TRANSLATION = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u2026": "...",
            "\u00a0": " ",
        }
    )

    @staticmethod
    def _repair_common_mojibake(text: str) -> str:
        # Repair common UTF-8 decoded as Latin-1 artifacts (e.g., "â€™", "Ã").
        if not any(marker in text for marker in ("\u00e2", "\u00c3", "\u00c2")):
            return text
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            return repaired or text
        except Exception:
            return text

    def clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value)
        text = self._repair_common_mojibake(text)       
        text = html.unescape(text)
        text = text.replace("\\'", "'").replace('\\"', '"')
        text = text.translate(self._UNICODE_PUNCT_TRANSLATION)
        text = re.sub(r"(?<=\w)`(?=\w)", "'", text)
        text = self._ZERO_WIDTH_RE.sub(" ", text)
        text = self._WHITESPACE_RE.sub(" ", text).strip()
        return text

    def _safe_list_parse(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, np.ndarray)):
            out = [self.clean_text(x) for x in value]
            return [x for x in out if x]
        s = str(value).strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                out = [self.clean_text(x) for x in parsed]
                return [x for x in out if x]
        except Exception:
            pass
        cleaned = self.clean_text(s)
        return [cleaned] if cleaned else []

    def clean_topic_info(self, topic_info: pd.DataFrame) -> pd.DataFrame:
        out = topic_info.copy()
        if "Representative_Docs" in out.columns:
            out["Representative_Docs"] = out["Representative_Docs"].apply(
                lambda x: self._safe_list_parse(x)
            )
        return out


class TopicRefinementPipeline:
    """Refine BERTopic topic labels and export a single refined table."""

    def __init__(
        self,
        topic_info: pd.DataFrame,
        config: Optional[TopicRefineConfig] = None,
    ):
        self.config = config or TopicRefineConfig()
        self.labeler = TopicLabelRefiner(self.config)
        self.rep_doc_cleaner = RepresentativeDocCleaner()

        self.topic_info_raw = self.rep_doc_cleaner.clean_topic_info(topic_info.copy())
        self.topic_info = self.labeler.build_topic_label_table(self.topic_info_raw)

    @classmethod
    def from_csv(
        cls,
        topic_info_path: str | Path,
        config: Optional[TopicRefineConfig] = None,
    ) -> "TopicRefinementPipeline":
        topic_info = pd.read_csv(topic_info_path)
        return cls(topic_info, config=config)

    def topic_labels_refined(self) -> pd.DataFrame:
        return self.topic_info.copy()

    def export_topic_labels(self, outdir: str | Path) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / "topic_labels_refined.csv"
        self.topic_labels_refined().to_csv(out_path, index=False)
        return out_path
