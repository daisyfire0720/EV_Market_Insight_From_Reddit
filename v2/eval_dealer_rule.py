#!/usr/bin/env python3
"""Score the hand-written dealer sentences and write a before/after table.

The sentences are invented. They are not Reddit posts. "Before" is the
original dealer label at threshold 0.5. "After" is that score or the
dealer-friction lexicon. A second column records an alternate dealer label
that was tried and not adopted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotate import (  # noqa: E402
    BARRIER_HYPOTHESIS,
    BARRIER_LABELS,
    DEFAULT_MODEL,
    _load,
    _one,
)
from ev_v2.dealer import decide_dealer

ALTERNATE_DEALER = "dealer markup or dealer pressure during the purchase"
FIXTURE = ROOT / "fixtures" / "dealer_sentences.csv"
OUT = ROOT / "outputs" / "dealer_rule_before_after.csv"


def main() -> None:
    fixture = pd.read_csv(FIXTURE)
    clf, revision = _load(DEFAULT_MODEL)
    alternate_labels = tuple(ALTERNATE_DEALER if label == "dealer or buying friction" else label for label in BARRIER_LABELS)
    rows = []
    for _, item in fixture.iterrows():
        text = str(item["text"])
        before = _one(clf, text, BARRIER_LABELS, BARRIER_HYPOTHESIS, multi_label=True)
        after_labels = _one(clf, text, alternate_labels, BARRIER_HYPOTHESIS, multi_label=True)
        dealer_score = before["dealer or buying friction"]
        price_score = before["vehicle price or total cost"]
        decision = decide_dealer(text, dealer_score, 0.5)
        alternate_score = after_labels[ALTERNATE_DEALER]
        rows.append(
            {
                "sentence_id": item["sentence_id"],
                "kind": item["kind"],
                "expect_dealer": item["expect_dealer"],
                "text": text,
                "before_dealer_score": dealer_score,
                "before_price_score": price_score,
                "before_on": dealer_score >= 0.5,
                "after_on": decision.on,
                "after_reason": decision.reason,
                "alternate_label_score": alternate_score,
                "alternate_label_on": alternate_score >= 0.5,
                "model": DEFAULT_MODEL,
                "revision": revision,
            }
        )
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    expect_on = frame["expect_dealer"].eq("yes")
    report = ROOT / "outputs" / "dealer_rule_report.md"
    lines = [
        "# Dealer-friction rule on hand-written sentences",
        "",
        "These sentences were written for the codebook. They are not Reddit posts. "
        f"Model `{DEFAULT_MODEL}` revision `{revision}`. Threshold 0.5.",
        "",
        "Before: the original label `dealer or buying friction` is on when its score is at least 0.5.",
        "After: that score, or the dealer-friction lexicon. Ordinary service and charging-at-a-dealer are forced off when the lexicon does not match.",
        f"Alternate label tried and not adopted: `{ALTERNATE_DEALER}`, same template `This text is about {{}}.`.",
        "",
        f"Sentences: {len(frame)}. Before matched the intended label on {int((frame.before_on == expect_on).sum())}. After matched on {int((frame.after_on == expect_on).sum())}. The alternate label matched on {int((frame.alternate_label_on == expect_on).sum())}.",
        "",
        f"Intended dealer sentences: {int(expect_on.sum())}. Before on: {int(frame.loc[expect_on, 'before_on'].sum())}. After on: {int(frame.loc[expect_on, 'after_on'].sum())}.",
        f"Intended non-dealer sentences: {int((~expect_on).sum())}. Before false positives: {int((~expect_on & frame.before_on).sum())}. After false positives: {int((~expect_on & frame.after_on).sum())}.",
        "",
        "| Id | Intended | Before | After | Reason | Dealer score | Price score | Alternate on |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['sentence_id']} | {row['expect_dealer']} | {row['before_on']} | {row['after_on']} | {row['after_reason']} | {row['before_dealer_score']:.3f} | {row['before_price_score']:.3f} | {row['alternate_label_on']} |"
        )
    lines.append("")
    report.write_text("\n".join(lines))
    print(frame[["sentence_id", "expect_dealer", "before_on", "after_on", "after_reason"]].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
