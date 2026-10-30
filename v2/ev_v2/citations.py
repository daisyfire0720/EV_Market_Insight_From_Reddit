"""Check that every citation key in the paper draft is defined in the BibTeX file.

A key is the token inside ``[@key]``. The check does not invent a reference
and does not decide whether a DOI is correct. It only fails when the draft
names a key the ``.bib`` file does not define.
"""

from __future__ import annotations

import re
from pathlib import Path

CITE_KEY = re.compile(r"\[@([A-Za-z0-9:_-]+)\]")
BIB_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")


def citation_keys(text: str) -> set[str]:
    return set(CITE_KEY.findall(text))


def bib_keys(text: str) -> set[str]:
    return set(BIB_KEY.findall(text))


def missing_citation_keys(paper_dir: Path, bib_name: str = "references.bib") -> list[str]:
    paper_dir = Path(paper_dir)
    bib_path = paper_dir / bib_name
    defined = bib_keys(bib_path.read_text(encoding="utf-8")) if bib_path.is_file() else set()
    used: set[str] = set()
    for path in sorted(paper_dir.glob("*.md")):
        used.update(citation_keys(path.read_text(encoding="utf-8")))
    return sorted(used - defined)
