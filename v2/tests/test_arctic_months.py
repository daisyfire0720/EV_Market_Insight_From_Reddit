from ev_v2.arctic import bucket_month


def test_aggregate_buckets_use_amsterdam_month_starts():
    # Observed 16 Oct 2026 from BoltEV monthly aggregates.
    assert bucket_month("2023-12-31T23:00:00.000Z") == "2024-01"
    assert bucket_month("2024-01-31T23:00:00.000Z") == "2024-02"
    assert bucket_month("2024-05-31T22:00:00.000Z") == "2024-06"
    assert bucket_month("2024-06-30T22:00:00.000Z") == "2024-07"
