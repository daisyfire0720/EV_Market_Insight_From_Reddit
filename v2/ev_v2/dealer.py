"""Dealer-friction decision.

The zero-shot model often scores a dealer markup as vehicle price and leaves
the dealer label off. A friction lexicon covers that miss. The raw NLI score
is kept. The lexicon does not fire on a bare mention of a dealer, a service
visit, or a sticker price with no fee or markup language.

Hypothesis variants that reworded the dealer label were worse on a fixed set
of invented sentences: they raised fair price quotes and missed inventory
waits. Those comparisons are in ``v2/outputs/dealer_rule_before_after.csv``
after ``v2/eval_dealer_rule.py`` runs. This module is the rule that file
evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ev_v2.textmatch import contains_keyword

# Phrases that are dealer friction even when the word "dealer" is missing.
_PHRASES = (
    "market adjustment",
    "market adjusted",
    "dealer markup",
    "dealership markup",
    "dealer mark-up",
    "doc fee",
    "documentation fee",
    "dealer fee",
    "dealer fees",
    "dealer addendum",
    "bait and switch",
    "bait-and-switch",
    "stealership",
)

# Tokens that count only together with a dealer mention.
_WITH_DEALER = (
    "markup",
    "mark-up",
    "marked up",
    "addendum",
    "no inventory",
    "none in stock",
    "not in stock",
    "waitlist",
    "wait list",
)

_DEALER_WORDS = ("dealer", "dealership", "dealerships", "stealership")


def mentions_dealer(text: str) -> bool:
    return any(contains_keyword(text, word) for word in _DEALER_WORDS)


def friction_lexicon(text: str) -> bool:
    """True when the text uses markup, fee, or inventory language."""
    if any(contains_keyword(text, phrase) for phrase in _PHRASES):
        return True
    if not mentions_dealer(text):
        return False
    return any(contains_keyword(text, token) for token in _WITH_DEALER)


@dataclass(frozen=True)
class DealerDecision:
    on: bool
    reason: str


def _ordinary_service_or_charging_location(text: str) -> bool:
    """A dealer mention that is a service visit or a place to charge.

    The codebook says ordinary service is not dealer friction. A charger that
    happens to sit at a dealer is charging access. Either one is ignored only
    when the friction lexicon did not match.
    """
    if friction_lexicon(text) or not mentions_dealer(text):
        return False
    if contains_keyword(text, "service"):
        return True
    return contains_keyword(text, "charging") or contains_keyword(text, "charger")


def decide_dealer(text: str, dealer_score: float, threshold: float = 0.5) -> DealerDecision:
    """Combine the NLI score with the lexicon.

    ``reason`` is ``nli``, ``lexicon``, ``nli+lexicon``, ``off``, or
    ``suppressed_service_or_charging_location``.
    """
    blob = text or ""
    nli_on = float(dealer_score) >= threshold
    lexicon_on = friction_lexicon(blob)
    if nli_on and lexicon_on:
        return DealerDecision(True, "nli+lexicon")
    if lexicon_on:
        return DealerDecision(True, "lexicon")
    if nli_on and _ordinary_service_or_charging_location(blob):
        return DealerDecision(False, "suppressed_service_or_charging_location")
    if nli_on:
        return DealerDecision(True, "nli")
    return DealerDecision(False, "off")
