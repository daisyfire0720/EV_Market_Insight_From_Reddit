import pandas as pd

from ev_v2.events import EVENTS
from run_analysis import event_volume_table


def test_event_sources_are_nonempty_and_ids_unique():
    ids = [e.event_id for e in EVENTS]
    assert len(ids) == len(set(ids))
    for event in EVENTS:
        assert event.source.startswith("http") or "http" in event.source
        assert event.month_start <= event.month_end


def test_volume_ratio_uses_census_months():
    # Flat 100 posts every month, then 200 in August 2022.
    months = pd.period_range("2021-01", "2023-12", freq="M").astype(str)
    rows = []
    for month in months:
        n = 200 if month == "2022-08" else 100
        rows.append({"subreddit": "electricvehicles", "month": month, "n_posts": n, "count_status": "ok"})
    table = event_volume_table(pd.DataFrame(rows)).set_index("event_id")
    ira = table.loc["ira_2022"]
    assert ira["mean_posts_event_months"] == 200
    assert ira["mean_posts_prior_3_months"] == 100
    assert ira["ratio_vs_prior_3"] == 2
    assert ira["mean_posts_same_months_prior_year"] == 100
    assert ira["ratio_vs_year_ago"] == 2
