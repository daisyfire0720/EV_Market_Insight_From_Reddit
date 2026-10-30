"""Word-boundary keyword matching.

The v1 funnel scored keywords with ``kw in text``. That treats ``ev`` as a
hit inside ``every`` and ``vs`` as a hit inside ``evs``. Matching here
requires the keyword to sit on token boundaries.
"""

from __future__ import annotations

import re


def _boundary_pattern(keyword: str) -> re.Pattern[str]:
    kw = keyword.strip()
    if not kw:
        raise ValueError("keyword must be non-empty")
    # (?<!\w) / (?!\w) rather than \b so a keyword is not a prefix or suffix
    # of a longer token (ev/every, vs/evs) and still matches next to punctuation.
    return re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", flags=re.IGNORECASE)


def contains_keyword(text: str, keyword: str) -> bool:
    if text is None:
        return False
    return _boundary_pattern(keyword).search(str(text)) is not None


def find_keywords(text: str, keywords: list[str]) -> list[str]:
    found: list[str] = []
    blob = "" if text is None else str(text)
    for kw in keywords:
        if contains_keyword(blob, kw):
            found.append(kw)
    return found


def contains_any(text: str, keywords: list[str]) -> bool:
    return bool(find_keywords(text, keywords))
