"""Funnel stages, topic families, and the finer barrier aspects.

The definitions are the ones in Daisy's reference taxonomy note
(``v2/docs/reference_taxonomy_6stage_8family.pdf``). A post gets one stage,
from the poster's place in the journey, and one or more families, from what
the discussion is about. The seven barriers already in the label file stay
as the finer aspects.

The sampling helpers below are not codes. They only decide which posts are
drawn onto the blank sheet. A lexicon hit is not a label.
"""

from __future__ import annotations

import re

STAGES: tuple[tuple[str, str], ...] = (
    ("awareness", "Awareness / Need Formation"),
    ("consideration", "Consideration / Shortlisting"),
    ("evaluation", "Evaluation / Practical Fit"),
    ("purchase", "Purchase / Transaction"),
    ("onboarding", "Onboarding / Early Ownership"),
    ("long_term_ownership", "Long-Term Ownership / Retention"),
)

STAGE_CODES: tuple[str, ...] = tuple(code for code, _ in STAGES) + ("unclear",)

# Ordinal used by the transition scaffold. Unclear is not on the ladder.
STAGE_ORDER = {code: index + 1 for index, (code, _) in enumerate(STAGES)}

FAMILIES: tuple[tuple[str, str], ...] = (
    ("purchase_selection", "EV Purchase and Vehicle Selection"),
    ("charging_energy", "Charging and Energy Access"),
    ("battery_range", "Battery, Range, and Real-World Performance"),
    ("cost_incentives", "Cost, Incentives, and Financial Feasibility"),
    ("dealer_access", "Dealer, Purchase Process, and Market Access"),
    ("reliability_service", "Reliability, Service, and Ownership Experience"),
    ("policy_market", "Policy, Regulation, and Market Environment"),
    ("transition_narratives", "ICE / Hybrid / EV Transition Narratives"),
)

FAMILY_CODES: tuple[str, ...] = tuple(code for code, _ in FAMILIES)

# Finer aspects. Same seven barriers as the small-model file, plus an explicit none.
ASPECTS: tuple[tuple[str, str], ...] = (
    ("charging_access", "charging access"),
    ("range", "driving range"),
    ("price_tco", "vehicle price or total cost"),
    ("incentives_policy", "purchase incentives or tax credits"),
    ("battery_reliability", "battery reliability"),
    ("dealer_friction", "dealer or buying friction"),
    ("environment", "environmental impact"),
)

ASPECT_CODES: tuple[str, ...] = tuple(code for code, _ in ASPECTS) + ("none",)

STANCE_CODES: tuple[str, ...] = ("support", "oppose", "mixed", "no_stance")

# Small-model stage strings, mapped only as far as that model can go.
# It has one ownership class. It does not separate onboarding from long-term ownership.
SMALL_MODEL_STAGE = {
    "learning about electric vehicles": "awareness",
    "comparing electric vehicle options": "consideration",
    "evaluating cost or fit before buying": "evaluation",
    "buying an electric vehicle": "purchase",
    "owning an electric vehicle": "ownership",
    "no clear purchase stage": "unclear",
}

# Which aspect columns suggest a family, for the sampling draw only.
ASPECT_TO_FAMILIES = {
    "charging_access": ("charging_energy",),
    "range": ("battery_range",),
    "battery_reliability": ("battery_range",),
    "price_tco": ("cost_incentives",),
    "incentives_policy": ("cost_incentives", "policy_market"),
    "dealer_friction": ("dealer_access",),
    "environment": ("policy_market",),
}

_ONBOARDING = re.compile(
    r"\b(just (bought|purchased|picked up|got my|got the)|took delivery|new owner|"
    r"first (charge|week|night|ev)|picked it up|got it yesterday|"
    r"how (do|should) i set up|set up (the )?(app|charger|home charg))\b",
    re.IGNORECASE,
)
_LONG_TERM = re.compile(
    r"\b(degrad\w*|years of ownership|after \d+ (years|winters)|warranty|resale|"
    r"service cent(er|re)|long-term|battery health|owned (it|mine|this) for)\b",
    re.IGNORECASE,
)
_RELIABILITY = re.compile(
    r"\b(service cent(er|re)|warranty|repair|maintenance|recall)\b",
    re.IGNORECASE,
)
_TRANSITION = re.compile(
    r"\b(hybrid|gasoline|gas car|petrol|internal combustion|switching from|came from)\b",
    re.IGNORECASE,
)
_SELECTION = re.compile(
    r"\b(versus|vs\.?|which (ev|model|one)|recommend a|shortlist)\b",
    re.IGNORECASE,
)


def provisional_ownership_split(text: str | None) -> str | None:
    """Return ``onboarding`` or ``long_term_ownership`` when the wording is one-sided.

    Used only to oversample the sheet. Both patterns, or neither, returns None.
    """
    if not text:
        return None
    early = _ONBOARDING.search(text) is not None
    later = _LONG_TERM.search(text) is not None
    if early and not later:
        return "onboarding"
    if later and not early:
        return "long_term_ownership"
    return None


def sampling_stage(stage_label: str | None, text: str | None = None) -> str:
    """Stage stratum for the draw. Ownership stays unsplit unless the lexicon is one-sided."""
    mapped = SMALL_MODEL_STAGE.get(str(stage_label or ""), "unclear")
    if mapped != "ownership":
        return mapped
    split = provisional_ownership_split(text)
    return split or "ownership_unsplit"


def sampling_families(aspect_flags: dict[str, bool], stage_label: str | None, text: str | None = None) -> set[str]:
    """Family strata implied by the small-model aspects, plus a few lexicon hints."""
    found: set[str] = set()
    for aspect, families in ASPECT_TO_FAMILIES.items():
        if aspect_flags.get(aspect):
            found.update(families)
    if str(stage_label or "") == "comparing electric vehicle options":
        found.add("purchase_selection")
    blob = text or ""
    if _SELECTION.search(blob):
        found.add("purchase_selection")
    if _RELIABILITY.search(blob):
        found.add("reliability_service")
    if _TRANSITION.search(blob):
        found.add("transition_narratives")
    return found
