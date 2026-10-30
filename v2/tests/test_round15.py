"""Citation keys in the draft must exist in the BibTeX file.

The current title and abstract stay. The alternative framing is a separate file.
"""

from pathlib import Path

from ev_v2.citations import bib_keys, citation_keys, missing_citation_keys
from ev_v2.manuscript import render_manuscript


def test_every_cited_key_is_defined_in_the_bib():
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper"
    missing = missing_citation_keys(paper)
    assert missing == []
    rendered = render_manuscript(root / "outputs")
    defined = bib_keys((paper / "references.bib").read_text(encoding="utf-8"))
    used = citation_keys(rendered)
    assert used
    assert used <= defined
    assert "[CITATION NEEDED]" not in rendered


def test_the_working_title_and_abstract_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    text = render_manuscript(root / "outputs")
    assert text.startswith("# Stage and topic in electric-vehicle talk on Reddit, 2015–2026\n")
    assert "Working title used below: option 1." in text
    assert "1. Where the barrier sits in the purchase:" in text
    alt = (root / "paper" / "ALT_FRAMING.md").read_text(encoding="utf-8")
    assert "not a decision" in alt
    assert "Stage before shock:" in alt
    assert "manuscript.md" in alt
    abstract = alt.split("## Alternative abstract", 1)[1].split("## Contributions", 1)[0]
    words = [word for word in abstract.split() if word.strip()]
    assert 200 <= len(words) <= 300
