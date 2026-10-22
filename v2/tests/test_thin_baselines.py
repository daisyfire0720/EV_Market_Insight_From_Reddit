"""The thin-baseline fill stays inside its budget and does not steal old clusters."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ev_v2.blocks import FILL_MONTHS, assign_week_block, fill_window_for_day
from ev_v2.round6_report import moved_rows
from ev_v2.thin_baselines import FILL_LIMIT, LABEL_BUDGET_MINUTES, max_posts, planned_windows


def _stamp(year, month, day):
    return int(datetime(year, month, day, 12, tzinfo=timezone.utc).timestamp())


def test_budget_is_the_four_flagged_months_and_under_a_few_hours():
    assert FILL_MONTHS == ("2021-08", "2022-06", "2022-12", "2024-09")
    windows = planned_windows()
    # 7 subreddits × 4 months × 3 weeks.
    assert len(windows) == 84
    assert max_posts() == 84 * FILL_LIMIT
    assert max_posts() == 3360
    assert LABEL_BUDGET_MINUTES == 90
    months = {month for _, month, _, _ in windows}
    assert months == set(FILL_MONTHS)


def test_fill_week_is_used_only_when_the_quarter_window_does_not_cover_the_day():
    # 3 June 2022 is inside the first fill week and is not an event week.
    stamp = _stamp(2022, 6, 3)
    block = assign_week_block("electricvehicles", stamp)
    window = fill_window_for_day(datetime(2022, 6, 3, tzinfo=timezone.utc).date())
    assert window == (datetime(2022, 6, 1).date(), datetime(2022, 6, 8).date())
    # Either the quarterly sample already covers this day, or the fill block is used.
    if "|fill|" in block:
        assert block == "electricvehicles|fill|2022-06-01"
    else:
        assert "2022Q2" in block
    # A day between the fill weeks stays on the month fallback unless the quarter window hits it.
    gap = assign_week_block("electricvehicles", _stamp(2022, 6, 9))
    assert "|fill|" not in gap
    # An event week still wins over a fill week. None of the fill months is an event
    # month, so check the IRA week directly.
    event = assign_week_block("electricvehicles", _stamp(2022, 8, 16))
    assert event.startswith("electricvehicles|ira_2022|")


def test_quarter_window_beats_a_fill_window_on_the_same_day():
    from datetime import date

    from ev_v2.blocks import _quarter_window

    as_of = date(2026, 10, 16)
    found = False
    for subreddit in ("electricvehicles", "teslamotors", "Rivian", "BoltEV", "Ioniq5", "KiaEV6", "evcharging"):
        for year, month in ((2021, 8), (2022, 6), (2022, 12), (2024, 9)):
            quarter = (month - 1) // 3 + 1
            window = _quarter_window(subreddit, year, quarter, as_of)
            if window is None:
                continue
            day = window[0]
            while day < window[1]:
                if f"{day.year:04d}-{day.month:02d}" in FILL_MONTHS and fill_window_for_day(day) is not None:
                    block = assign_week_block(subreddit, _stamp(day.year, day.month, day.day))
                    assert f"{year}Q{quarter}" in block
                    assert "|fill|" not in block
                    found = True
                    break
                day = date(day.year, day.month, day.day).fromordinal(day.toordinal() + 1)
            if found:
                break
        if found:
            break
    assert found


def test_manuscript_marks_unvalidated_results_and_missing_citations(tmp_path):
    from ev_v2.manuscript import markdown_to_tex, render_manuscript

    text = render_manuscript(Path("/workspace/v2/outputs"))
    assert "unvalidated" in text
    assert "[CITATION NEEDED]" in text
    assert "annotation_sheet.csv" in text
    tex = markdown_to_tex(text)
    assert "\\documentclass" in tex
    assert "\\section{Abstract}" in tex
    (tmp_path / "manuscript.md").write_text(text)


def test_moved_rows_keeps_only_a_changed_zero_decision():
    before = pd.DataFrame(
        [
            {
                "event_id": "ira_2022",
                "barrier": "purchase incentives or tax credits",
                "audience": "enthusiast",
                "baseline": "prior_3_months",
                "block": "week_block",
                "primary": True,
                "reported": True,
                "diff": 0.05,
                "boot_lo": 0.01,
                "boot_hi": 0.09,
                "n_baseline": 20,
            },
            {
                "event_id": "nacs_2023",
                "barrier": "charging access",
                "audience": "enthusiast",
                "baseline": "prior_3_months",
                "block": "week_block",
                "primary": True,
                "reported": True,
                "diff": -0.02,
                "boot_lo": -0.10,
                "boot_hi": 0.04,
                "n_baseline": 40,
            },
        ]
    )
    after = before.copy()
    after.loc[0, "boot_lo"] = -0.01
    after.loc[0, "n_baseline"] = 80
    moved = moved_rows(before, after)
    assert len(moved) == 1
    assert moved.iloc[0]["event_id"] == "ira_2022"
    assert bool(moved.iloc[0]["excluded_before"]) is True
    assert bool(moved.iloc[0]["excluded_after"]) is False
