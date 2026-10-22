"""Policy and market dates used as descriptive windows.

Dates are taken from the sources named in ``source``. Windows are for
description (volume and language around the date). They are not an
identification strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PolicyEvent:
    event_id: str
    label: str
    event_date: date
    # Inclusive calendar-month window used for the monthly volume comparison.
    month_start: str  # YYYY-MM
    month_end: str
    source: str
    note: str


# month_start/month_end are inclusive YYYY-MM strings.
EVENTS: tuple[PolicyEvent, ...] = (
    PolicyEvent(
        event_id="ira_2022",
        label="Inflation Reduction Act enacted",
        event_date=date(2022, 8, 16),
        month_start="2022-08",
        month_end="2022-08",
        source="Pub. L. 117-169, 16 Aug 2022, as cited in the amendment notes to 26 U.S.C. § 30D (https://uscode.house.gov/us/usc/119-118/t26/s30D/).",
        note="The Act rewrote the section 30D clean-vehicle credit. August 2022 is the enactment month, not a claim that discussion started on the 16th.",
    ),
    PolicyEvent(
        event_id="tesla_price_cut_2023",
        label="Tesla US and global price cuts",
        event_date=date(2023, 1, 13),
        month_start="2023-01",
        month_end="2023-01",
        source="Reuters, 13 Jan 2023, https://www.reuters.com/business/autos-transportation/tesla-cuts-prices-electric-vehicles-us-market-2023-01-13/.",
        note="Reuters reported price cuts of as much as 20 percent. January is the event month.",
    ),
    PolicyEvent(
        event_id="nacs_2023",
        label="Ford and GM adopt Tesla's NACS connector",
        event_date=date(2023, 5, 25),
        month_start="2023-05",
        month_end="2023-06",
        source=(
            "Reuters, 25 May 2023, https://www.reuters.com/business/autos-transportation/tesla-ford-ceos-talk-evs-twitter-forum-2023-05-25/; "
            "General Motors news release, 8 Jun 2023, https://investor.gm.com/news-releases/news-release-details/general-motors-doubles-down-commitment-unified-charging-standard/."
        ),
        note="Ford's 25 May 2023 agreement and GM's 8 June 2023 NACS announcement are one window covering May and June 2023.",
    ),
    PolicyEvent(
        event_id="credit_law_2025",
        label="Law ending the clean-vehicle credit enacted",
        event_date=date(2025, 7, 4),
        month_start="2025-07",
        month_end="2025-07",
        source="Pub. L. 119-21, 4 Jul 2025, 139 Stat. 72, amendment notes to 26 U.S.C. § 30D; IRS FAQ https://www.irs.gov/newsroom/faqs-for-modification-of-sections-25c-25d-25e-30c-30d-45l-45w-and-179d-under-public-law-119-21-139-stat-72-july-4-2025-commonly-known-as-the-one-big-beautiful-bill-obbb.",
        note="The statute replaced a 31 Dec 2032 placed-in-service sunset with an acquired-after 30 Sep 2025 cutoff. July 2025 is the enactment month.",
    ),
    PolicyEvent(
        event_id="credit_deadline_2025",
        label="Federal clean-vehicle credit acquisition deadline",
        event_date=date(2025, 9, 30),
        month_start="2025-09",
        month_end="2025-10",
        source="26 U.S.C. § 30D(h) as amended; IRS, https://www.irs.gov/clean-vehicle-tax-credits.",
        note="IRS: the new, previously-owned, and commercial clean-vehicle credits are not available for vehicles acquired after 30 Sep 2025. September-October covers the deadline and the following month. October 2026 is unrelated; this window is 2025.",
    ),
)


def event_months(event: PolicyEvent) -> list[str]:
    start_y, start_m = (int(x) for x in event.month_start.split("-"))
    end_y, end_m = (int(x) for x in event.month_end.split("-"))
    months = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months
