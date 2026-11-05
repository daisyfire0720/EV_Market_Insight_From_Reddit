"""Comments v2: balanced draw, parent title in the prompt, own-draw RDiT.

Mock rows only. No model call and no network.
"""

from datetime import date, datetime, timezone

import pandas as pd

from ev_v2.after_llm import run_tables
from ev_v2.comments_v2 import (
    assign_candidates,
    balanced_draw,
    is_usable_body,
    nearest_cutoff,
    ordered_sides,
    parent_fields,
)
from ev_v2.taxonomy import ASPECT_CODES
import llm_label


def _ts(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc).timestamp()


def test_nearest_cutoff_assigns_one_cutoff_and_ties_go_earlier():
    assert nearest_cutoff(date(2023, 1, 7)) == ("ira_credit_rules_2023", 6)
    assert nearest_cutoff(date(2023, 1, 8)) == ("tesla_price_cut_2023", -5)
    assert nearest_cutoff(date(2022, 8, 15)) == ("ira_signing_2022", -1)
    assert nearest_cutoff(date(2025, 10, 1)) == ("credit_end_2025", 1)
    assert nearest_cutoff(None) is None


def test_candidates_drop_cutoff_day_and_outside_window_and_use_parent_audience():
    meta = pd.DataFrame(
        [
            {"id": "on_day", "created_utc": _ts(2022, 8, 16), "subreddit": "cars", "link_id": "t3_x", "text_type": "comment"},
            {"id": "left", "created_utc": _ts(2022, 8, 10), "subreddit": "cars", "link_id": "t3_p1", "text_type": "comment"},
            {"id": "right", "created_utc": _ts(2022, 8, 20), "subreddit": "cars", "link_id": "t3_x", "text_type": "comment"},
            {"id": "far", "created_utc": _ts(2024, 6, 1), "subreddit": "cars", "link_id": "t3_x", "text_type": "comment"},
            {"id": "nosub", "created_utc": _ts(2022, 8, 20), "subreddit": "pics", "link_id": "t3_x", "text_type": "comment"},
        ]
    )
    out = assign_candidates(meta, {"p1": "enthusiast"}).set_index("id")
    assert set(out.index) == {"left", "right"}
    assert out.loc["left", "audience"] == "enthusiast"
    assert out.loc["left", "audience_source"] == "parent_link"
    assert out.loc["right", "audience"] == "shopper"
    assert out.loc["left", "draw_side"] == "left"
    assert out.loc["right", "draw_side"] == "right"


def _many(n_left, n_right):
    rows = []
    for i in range(n_left):
        rows.append({"id": f"L{i}", "created_utc": _ts(2025, 9, 29) - i * 3600, "subreddit": "electricvehicles", "link_id": "", "text_type": "comment"})
    for i in range(n_right):
        rows.append({"id": f"R{i}", "created_utc": _ts(2025, 10, 1) + i * 3600, "subreddit": "electricvehicles", "link_id": "", "text_type": "comment"})
    return assign_candidates(pd.DataFrame(rows), {})


def test_balanced_draw_is_equal_per_side_and_trimmed_to_the_short_side():
    order = ordered_sides(_many(40, 7), seed=1)
    rows, counts = balanced_draw(order, None, cap=10)
    drawn = pd.DataFrame(rows, columns=["id", "cutoff", "audience", "side"])
    assert (drawn["side"] == "left").sum() == 7
    assert (drawn["side"] == "right").sum() == 7
    row = counts[(counts["cutoff_id"] == "credit_end_2025") & (counts["audience"] == "enthusiast")].iloc[0]
    assert int(row["n_drawn_per_side"]) == 7
    assert bool(row["trimmed_to_short_side"])


def test_draw_is_seeded_random_not_closest_first_and_respects_usable_bodies():
    cands = _many(100, 100)
    order_a = ordered_sides(cands, seed=5)
    order_b = ordered_sides(cands, seed=5)
    assert order_a == order_b
    rows, _ = balanced_draw(order_a, None, cap=5)
    left = {r[0] for r in rows if r[3] == "left"}
    assert left != {f"L{i}" for i in range(5)}
    usable = {f"L{i}" for i in range(3)} | {f"R{i}" for i in range(50)}
    rows, _ = balanced_draw(order_a, usable, cap=5)
    assert sum(r[3] == "left" for r in rows) == 3
    assert sum(r[3] == "right" for r in rows) == 3
    assert all(r[0] in usable for r in rows)


def test_usable_body_and_parent_fields():
    assert not is_usable_body("[removed]")
    assert not is_usable_body("")
    assert not is_usable_body("Thanks! I am a bot, and this action was performed automatically.")
    assert is_usable_body("Got my Ioniq 5 last week")
    title, excerpt = parent_fields("Should I buy?", "x" * 500)
    assert title == "Should I buy?" and len(excerpt) == 300
    assert parent_fields(float("nan"), "[deleted]") == ("", "")


def test_comment_prompt_has_parent_context_and_its_own_hash():
    text = llm_label.comment_v2_text("cars", "Leasing a Model 3?", "Thinking about it", "Do it, the cut made it cheap")
    assert text.startswith("[comment]\nSubreddit: r/cars\nParent title: Leasing a Model 3?")
    assert "Comment:\nDo it" in text
    prompt = llm_label.codebook_prompt("", text, parent_context=True)
    assert "You code one Reddit comment" in prompt
    assert llm_label.PARENT_CONTEXT_NOTE in prompt
    empty = llm_label.comment_v2_text("cars", float("nan"), None, "ok")
    assert "Parent title: \n" in empty
    # The submission prompt and its cache hash are unchanged by the comment v2 work.
    assert llm_label.prompt_hash() == "a4ea383c018fc1b9"
    assert llm_label.PARENT_CONTEXT_NOTE not in llm_label.codebook_prompt("t", "b")
    assert llm_label.prompt_hash(llm_label.PROMPT_VERSION_COMMENTS_V2, parent_context=True) != llm_label.prompt_hash()


def test_performance_stays_out_of_the_codebook():
    assert "performance" not in ASPECT_CODES


class _Capture:
    name = "capture"
    version = "capture-0"

    def __init__(self):
        self.prompts = []

    def complete(self, prompts):
        self.prompts.extend(prompts)
        return ['{"stage": "unclear", "families": [], "aspects": [], "stance": "no_stance", "evidence": ""}' for _ in prompts]


def test_label_posts_uses_parent_prompt_and_new_version(tmp_path):
    frame = pd.DataFrame(
        [{"id": "c1", "text_type": "comment", "subreddit": "Rivian", "parent_title": "R1S delivery day", "parent_excerpt": "", "body": "Congrats!"}]
    )
    backend = _Capture()
    out = llm_label.label_posts(frame, backend, tmp_path / "cache.jsonl")
    assert "Parent title: R1S delivery day" in backend.prompts[0]
    assert out.iloc[0]["prompt_version"] == llm_label.PROMPT_VERSION_COMMENTS_V2
    assert out.iloc[0]["prompt_hash"] == llm_label.prompt_hash(llm_label.PROMPT_VERSION_COMMENTS_V2, parent_context=True)


def test_own_draw_rdit_is_written_for_a_rebalanced_v2_sample(tmp_path):
    small = pd.DataFrame([{"id": "s1", "audience": "enthusiast", "subreddit": "electricvehicles", "month": "2025-09"}])
    authors = pd.DataFrame([{"id": "s1", "author_hash": "h", "created_utc": _ts(2025, 9, 1)}])
    labels = [{"id": "s1", "text_type": "submission", "stage": "awareness", "families": "", "aspects": "", "ok": True}]
    meta = []
    for i in range(60):
        side = -1 if i % 2 else 1
        cid = f"c{i}"
        labels.append({"id": cid, "text_type": "comment", "stage": "unclear", "families": "", "aspects": "incentives_policy" if i % 3 else "", "ok": True})
        meta.append({"id": cid, "created_utc": _ts(2025, 9, 30) + side * (1 + i) * 86400, "subreddit": "electricvehicles", "link_id": "", "draw_cutoff_id": "credit_end_2025", "draw_side": "left" if side < 0 else "right"})
    paths = {name: tmp_path / f"{name}.csv" for name in ("llm", "small", "authors", "meta")}
    pd.DataFrame(labels).to_csv(paths["llm"], index=False)
    small.to_csv(paths["small"], index=False)
    authors.to_csv(paths["authors"], index=False)
    pd.DataFrame(meta).to_csv(paths["meta"], index=False)
    out = tmp_path / "tables"
    run_tables(paths["llm"], paths["small"], paths["authors"], out, n_boot=0, comment_meta_path=paths["meta"], comments_rebalanced=True, comment_n_boot=0)
    own = pd.read_csv(out / "rdit_comments_own_draw.csv")
    assert set(own["cutoff_id"]) == {"credit_end_2025"}
    assert set(own["fit"]) <= {"main", "donut"}
    balance = pd.read_csv(out / "comment_window_balance.csv")
    assert "balanced draw" in balance["note"].iloc[0]
