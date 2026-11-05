"""Purchase-journey keyword scorer with word-boundary matching.

Stage names and keyword lists are the five stages in
``src/ev_funnel/market_funnel_analyzer.py``. Scoring uses token boundaries
so generic tokens such as ``ev`` and ``vs`` do not fire inside unrelated
words. Substring scoring is still available so the two can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

from ev_v2.textmatch import contains_keyword


@dataclass(frozen=True)
class StageSpec:
    name: str
    keywords: tuple[str, ...]


# Copied from MarketFunnelAnalyzer.DEFAULT_STAGE_SPECS include_keywords.
STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        "Awareness",
        (
            "what car", "which car", "new car", "used car", "guide", "research",
            "beginner", "best", "recommend", "need a car", "first car", "buying guide",
            "worth it", "should i buy",
        ),
    ),
    StageSpec(
        "Consideration",
        (
            "compare", "comparison", "versus", "vs", "hybrid", "electric", "ev", "suv",
            "sedan", "reliability", "features", "resale", "warranty", "maintenance",
            "driving experience",
        ),
    ),
    StageSpec(
        "Evaluation",
        (
            "total cost", "ownership", "credit", "loan", "finance", "payment", "apr",
            "maintenance cost", "fuel efficiency", "safety", "test drive", "long term",
            "battery", "tax credit", "incentive",
        ),
    ),
    StageSpec(
        "Purchase",
        (
            "dealer", "dealership", "markup", "msrp", "market adjustment", "doc fee",
            "allocation", "wait list", "deposit", "negotiat", "purchase", "bought",
            "on the lot", "inventory", "delivery", "out the door",
        ),
    ),
    StageSpec(
        "Ownership",
        (
            "owned", "ownership experience", "service", "repair", "problem", "issue",
            "warranty claim", "after buying", "over time", "long-term", "maintenance",
            "battery replacement",
        ),
    ),
)


def _hits(text: str, keywords: tuple[str, ...], mode: str) -> list[str]:
    blob = "" if text is None else str(text).lower()
    found: list[str] = []
    for kw in keywords:
        if mode == "boundary":
            if contains_keyword(blob, kw):
                found.append(kw)
        elif mode == "substring":
            if kw.lower() in blob:
                found.append(kw)
        else:
            raise ValueError(f"unknown match mode: {mode}")
    return found


def assign_stage(text: str, mode: str = "boundary") -> tuple[str, list[str]]:
    """Return (stage, keywords that fired). Ties go to the earlier stage."""
    best_name = "Unclear"
    best_hits: list[str] = []
    best_score = 0
    for spec in STAGE_SPECS:
        hits = _hits(text, spec.keywords, mode)
        if len(hits) > best_score:
            best_name = spec.name
            best_hits = hits
            best_score = len(hits)
    return best_name, best_hits
