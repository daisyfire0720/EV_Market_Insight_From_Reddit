"""Framing A summaries: enthusiast sample versus shopper dump.

Labels are model output. Shares, interrupted time series, and event-study
contrasts describe those labels. They are not effects of the policy, and
they are not validated against human codes.

The enthusiast rows are the text sample from the seven EV subreddits.
The shopper rows are the keyword posts extracted from the dumps. A month
with no sample posts is left out. It is not filled in as a zero.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from ev_v2.arctic import CORE_SUBREDDITS, GENERAL_SUBREDDITS
from ev_v2.events import EVENTS, event_months

ENTHUSIAST = "enthusiast"
SHOPPER = "shopper"
AUDIENCES = (ENTHUSIAST, SHOPPER)

# The four shocks Framing A is built around. July 2025 (the statute month)
# stays in the volume table and is not a fifth interruption here.
FRAMING_EVENT_IDS = (
    "ira_2022",
    "tesla_price_cut_2023",
    "nacs_2023",
    "credit_deadline_2025",
)

# Pre-specified barrier for each shock. The tables also report the other barriers.
PRIMARY_BARRIER = {
    "ira_2022": "purchase incentives or tax credits",
    "tesla_price_cut_2023": "vehicle price or total cost",
    "nacs_2023": "charging access",
    "credit_deadline_2025": "purchase incentives or tax credits",
}

ITS_PRE_MONTHS = 12
ITS_POST_MONTHS = 6
EVENT_STUDY_PRE = 6
EVENT_STUDY_POST = 6
Z_95 = 1.96


def month_index(month: str) -> int:
    year, mon = (int(part) for part in str(month)[:7].split("-"))
    return year * 12 + mon - 1


def index_to_month(index: int) -> str:
    year, mon = divmod(int(index), 12)
    return f"{year:04d}-{mon + 1:02d}"


def shift_month(month: str, delta: int) -> str:
    return index_to_month(month_index(month) + delta)


def barrier_column(label: str) -> str:
    return "barrier_" + label.replace(" ", "_") + "_on"


def build_corpus(core: pd.DataFrame, dump: pd.DataFrame) -> pd.DataFrame:
    """Enthusiast text sample plus shopper dump posts, one row per id.

    Shopper posts that also appear in the enthusiast file are kept as
    enthusiast. That overlap is not expected. Shopper posts that also
    appear in the old mainstream API sample are kept from the dump, which
    is the fuller extract.
    """
    enth = core[core["subreddit"].isin(CORE_SUBREDDITS)].copy()
    enth["audience"] = ENTHUSIAST
    shop = dump[dump["subreddit"].isin(GENERAL_SUBREDDITS)].copy()
    shop["audience"] = SHOPPER
    both = pd.concat([enth, shop], ignore_index=True)
    both["id"] = both["id"].astype(str)
    both = both.drop_duplicates("id", keep="first")
    both["month"] = both["month"].astype(str).str.slice(0, 7)
    return both.reset_index(drop=True)


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Share and a Wilson 95% interval. Empty months return missing values."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    k = int(k)
    n = int(n)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def proportion_difference(k1: int, n1: int, k2: int, n2: int, z: float = Z_95) -> dict:
    """Difference p1 - p2 with a normal interval. p1 is the event share."""
    if n1 <= 0 or n2 <= 0:
        return {"diff": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan"), "p1": float("nan"), "p2": float("nan")}
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2
    se = sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    return {"diff": diff, "se": se, "lo": diff - z * se, "hi": diff + z * se, "p1": p1, "p2": p2}


def _ols_hc1(x_matrix: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC1 standard errors."""
    n, k = x_matrix.shape
    beta, *_ = np.linalg.lstsq(x_matrix, y, rcond=None)
    resid = y - x_matrix @ beta
    bread = np.linalg.inv(x_matrix.T @ x_matrix)
    scale = n / (n - k) if n > k else float("nan")
    meat = x_matrix.T @ ((resid ** 2)[:, None] * x_matrix) * scale
    var = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(var), 0, None))
    return beta, se


def fit_interrupted_series(
    monthly: pd.DataFrame,
    event_month: str,
    pre: int = ITS_PRE_MONTHS,
    post: int = ITS_POST_MONTHS,
    min_n: int = 5,
) -> dict:
    """WLS of monthly shares on time, a post indicator, and time since the event.

    ``monthly`` needs ``month``, ``n``, and ``k`` (count of posts with the
    barrier on). Months with fewer than ``min_n`` posts are left out.
    Weights are the monthly counts. The post indicator turns on at
    ``event_month``, including that month. Time is calendar months, so a
    missing month does not renumber the months that remain.
    """
    empty = {
        "n_months": 0,
        "n_posts": 0,
        "n_months_post": 0,
        "status": "too_few_months",
        "intercept": float("nan"),
        "time": float("nan"),
        "post": float("nan"),
        "time_after": float("nan"),
        "post_se": float("nan"),
        "post_lo": float("nan"),
        "post_hi": float("nan"),
    }
    start = shift_month(event_month, -pre)
    end = shift_month(event_month, post)
    work = monthly.copy()
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    work = work[(work["month"] >= start) & (work["month"] <= end) & (work["n"] >= min_n)]
    if len(work) < 8:
        empty["n_months"] = int(len(work))
        empty["n_posts"] = int(work["n"].sum()) if len(work) else 0
        return empty
    origin = month_index(start)
    event_at = month_index(event_month)
    indexes = work["month"].map(month_index).to_numpy()
    time = (indexes - origin).astype(float)
    after = (indexes >= event_at).astype(float)
    time_after = np.maximum(indexes - event_at, 0).astype(float)
    share = (work["k"] / work["n"]).to_numpy(dtype=float)
    x_matrix = np.column_stack([np.ones(len(work)), time, after, time_after])
    if np.linalg.matrix_rank(x_matrix) < x_matrix.shape[1]:
        empty["n_months"] = int(len(work))
        empty["n_posts"] = int(work["n"].sum())
        empty["n_months_post"] = int(after.sum())
        empty["status"] = "not_identified"
        return empty
    weights = np.sqrt(work["n"].to_numpy(dtype=float))
    beta, se = _ols_hc1(x_matrix * weights[:, None], share * weights)
    return {
        "n_months": int(len(work)),
        "n_posts": int(work["n"].sum()),
        "n_months_post": int(after.sum()),
        "status": "ok",
        "intercept": float(beta[0]),
        "time": float(beta[1]),
        "post": float(beta[2]),
        "time_after": float(beta[3]),
        "post_se": float(se[2]),
        "post_lo": float(beta[2] - Z_95 * se[2]),
        "post_hi": float(beta[2] + Z_95 * se[2]),
        "window_start": start,
        "window_end": end,
    }


def monthly_counts(frame: pd.DataFrame, on_col: str) -> pd.DataFrame:
    """Posts and 'on' counts by audience and month."""
    work = frame.copy()
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    work["_on"] = work[on_col].fillna(False).astype(bool)
    grouped = (
        work.groupby(["audience", "month"], as_index=False)
        .agg(n=("id", "size"), k=("_on", "sum"))
    )
    shares = grouped.apply(lambda row: wilson_interval(int(row["k"]), int(row["n"])), axis=1, result_type="expand")
    shares.columns = ["share", "lo", "hi"]
    return pd.concat([grouped, shares], axis=1)


def _posts_for(frame: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    return frame[frame["month"].isin(months)]


def contrast_row(frame: pd.DataFrame, on_col: str, event_id: str, barrier: str, audience: str, months: list[str], baseline_months: list[str], baseline_name: str) -> dict:
    group = frame[frame["audience"] == audience]
    event = _posts_for(group, months)
    base = _posts_for(group, baseline_months)
    k1 = int(event[on_col].fillna(False).sum()) if len(event) else 0
    k2 = int(base[on_col].fillna(False).sum()) if len(base) else 0
    diff = proportion_difference(k1, len(event), k2, len(base))
    return {
        "event_id": event_id,
        "barrier": barrier,
        "audience": audience,
        "baseline": baseline_name,
        "primary": PRIMARY_BARRIER.get(event_id) == barrier,
        "n_event": int(len(event)),
        "k_event": k1,
        "n_event_months": int(event["month"].nunique()) if len(event) else 0,
        "n_baseline": int(len(base)),
        "k_baseline": k2,
        "n_baseline_months": int(base["month"].nunique()) if len(base) else 0,
        "share_event": diff["p1"],
        "share_baseline": diff["p2"],
        "diff": diff["diff"],
        "se": diff["se"],
        "lo": diff["lo"],
        "hi": diff["hi"],
    }


def event_contrasts(frame: pd.DataFrame, barriers: list[str]) -> pd.DataFrame:
    """Event-window share minus the prior three months, and minus a year earlier."""
    rows = []
    catalog = {event.event_id: event for event in EVENTS}
    for event_id in FRAMING_EVENT_IDS:
        event = catalog[event_id]
        months = event_months(event)
        prior = [shift_month(months[0], -step) for step in (3, 2, 1)]
        year_ago = [shift_month(month, -12) for month in months]
        for barrier in barriers:
            column = barrier_column(barrier)
            for audience in AUDIENCES:
                rows.append(contrast_row(frame, column, event_id, barrier, audience, months, prior, "prior_3_months"))
                rows.append(contrast_row(frame, column, event_id, barrier, audience, months, year_ago, "year_ago"))
    return pd.DataFrame(rows)


def interrupted_series_table(frame: pd.DataFrame, barriers: list[str]) -> pd.DataFrame:
    rows = []
    catalog = {event.event_id: event for event in EVENTS}
    for event_id in FRAMING_EVENT_IDS:
        first = event_months(catalog[event_id])[0]
        for barrier in barriers:
            column = barrier_column(barrier)
            counts = monthly_counts(frame, column)
            for audience in AUDIENCES:
                fit = fit_interrupted_series(counts[counts["audience"] == audience], first)
                rows.append(
                    {
                        "event_id": event_id,
                        "event_month": first,
                        "barrier": barrier,
                        "audience": audience,
                        "primary": PRIMARY_BARRIER.get(event_id) == barrier,
                        **fit,
                    }
                )
    return pd.DataFrame(rows)


def event_study(frame: pd.DataFrame, barriers: list[str], pre: int = EVENT_STUDY_PRE, post: int = EVENT_STUDY_POST) -> pd.DataFrame:
    """Share at each relative month minus the share in the month before the event.

    Relative month 0 is the first event month. The omitted month is -1.
    A two-month event occupies 0 and 1. The interval is the normal
    two-proportion interval. Posts are treated as independent, which is a
    rough interval for the enthusiast week-sample.
    """
    rows = []
    catalog = {event.event_id: event for event in EVENTS}
    for event_id in FRAMING_EVENT_IDS:
        first = event_months(catalog[event_id])[0]
        origin = month_index(first)
        for barrier in barriers:
            column = barrier_column(barrier)
            for audience in AUDIENCES:
                group = frame[frame["audience"] == audience].copy()
                group["rel"] = group["month"].map(month_index) - origin
                reference = group[group["rel"] == -1]
                k_ref = int(reference[column].fillna(False).sum()) if len(reference) else 0
                n_ref = int(len(reference))
                for rel in range(-pre, post + 1):
                    if rel == -1:
                        continue
                    block = group[group["rel"] == rel]
                    k = int(block[column].fillna(False).sum()) if len(block) else 0
                    n = int(len(block))
                    diff = proportion_difference(k, n, k_ref, n_ref)
                    rows.append(
                        {
                            "event_id": event_id,
                            "barrier": barrier,
                            "audience": audience,
                            "primary": PRIMARY_BARRIER.get(event_id) == barrier,
                            "rel_month": rel,
                            "n": n,
                            "k": k,
                            "n_ref": n_ref,
                            "share": diff["p1"],
                            "share_ref": diff["p2"],
                            "diff": diff["diff"],
                            "se": diff["se"],
                            "lo": diff["lo"],
                            "hi": diff["hi"],
                        }
                    )
    return pd.DataFrame(rows)


def overall_shares(frame: pd.DataFrame, barriers: list[str]) -> pd.DataFrame:
    rows = []
    for barrier in barriers:
        column = barrier_column(barrier)
        for audience in AUDIENCES:
            group = frame[frame["audience"] == audience]
            k = int(group[column].fillna(False).sum())
            n = int(len(group))
            share, lo, hi = wilson_interval(k, n)
            rows.append({"measure": barrier, "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
    for audience in AUDIENCES:
        group = frame[frame["audience"] == audience]
        k = int((group["stage_label"] == "owning an electric vehicle").sum()) if "stage_label" in group.columns else 0
        n = int(len(group))
        share, lo, hi = wilson_interval(k, n)
        rows.append({"measure": "owning an electric vehicle", "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
        if "stance_label" in group.columns:
            k = int((group["stance_label"] == "support for electric vehicles").sum())
            share, lo, hi = wilson_interval(k, n)
            rows.append({"measure": "support for electric vehicles", "audience": audience, "n": n, "k": k, "share": share, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def audience_gap(overall: pd.DataFrame, measure: str) -> dict:
    """Shopper share minus enthusiast share."""
    shop = overall[(overall["measure"] == measure) & (overall["audience"] == SHOPPER)]
    enth = overall[(overall["measure"] == measure) & (overall["audience"] == ENTHUSIAST)]
    if shop.empty or enth.empty:
        return proportion_difference(0, 0, 0, 0)
    return proportion_difference(int(shop.iloc[0]["k"]), int(shop.iloc[0]["n"]), int(enth.iloc[0]["k"]), int(enth.iloc[0]["n"]))


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def render_framing_markdown(out_dir) -> str:
    """Prose for FINDINGS.md, read from the tables this module writes."""
    from pathlib import Path

    out_dir = Path(out_dir)
    lines = [
        "## Framing A labels (unvalidated)",
        "",
        "This section is model output. No human has coded these posts. The intervals in the table below treat posts as independent, which is too narrow for a file built from sampled weeks. The week and month block intervals are in the round 5 section under this one. A difference here is not an effect of the statute or the price cut.",
        "",
        "Enthusiast rows are the text sample from the seven EV subreddits. Shopper rows are the dump extract from r/cars, r/Autos, and r/carbuying, including the January–September 2026 monthly files when that extract has been merged. A probable bot, or a post under five tokens, is dropped before labeling, which is why these counts are smaller than the raw extract.",
        "",
    ]
    overall_path = out_dir / "framing_overall_shares.csv"
    contrast_path = out_dir / "framing_event_contrasts.csv"
    its_path = out_dir / "framing_its.csv"
    if not overall_path.exists():
        lines.append("The Framing A labeling run has not been written yet.")
        lines.append("")
        return "\n".join(lines)
    overall = pd.read_csv(overall_path)
    lines.append("### Shares in the whole labeled file")
    lines.append("")
    lines.append("| Measure | Enthusiast share | n | Shopper share | n | Shopper minus enthusiast | 95% interval |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    measures = list(dict.fromkeys(overall["measure"]))
    for measure in measures:
        enth = overall[(overall["measure"] == measure) & (overall["audience"] == ENTHUSIAST)]
        shop = overall[(overall["measure"] == measure) & (overall["audience"] == SHOPPER)]
        if enth.empty or shop.empty:
            continue
        gap = audience_gap(overall, measure)
        lines.append(
            f"| {measure} | {_fmt(enth.iloc[0]['share'])} | {int(enth.iloc[0]['n'])} | {_fmt(shop.iloc[0]['share'])} | {int(shop.iloc[0]['n'])} | {_fmt(gap['diff'])} | {_fmt(gap['lo'])} to {_fmt(gap['hi'])} |"
        )
    lines.append("")
    lines.append("### Event windows")
    lines.append("")
    lines.append(
        "Each contrast is the share in the event month or months minus the share in the prior three months, and minus the same months a year earlier. "
        "The pre-specified barrier is incentives for August 2022 and for September–October 2025, price for January 2023, and charging for May–June 2023. "
        "The other barriers are reported in `framing_event_contrasts.csv`."
    )
    lines.append("")
    if contrast_path.exists():
        contrasts = pd.read_csv(contrast_path)
        primary = contrasts[contrasts["primary"] == True]  # noqa: E712
        lines.append("| Event | Audience | Baseline | Event share | Baseline share | Difference | 95% interval | Event n | Baseline n |")
        lines.append("|---|---|---|---:|---:|---:|---|---:|---:|")
        for _, row in primary.sort_values(["event_id", "audience", "baseline"]).iterrows():
            lines.append(
                f"| {row['event_id']} / {row['barrier']} | {row['audience']} | {row['baseline']} | {_fmt(row['share_event'])} | {_fmt(row['share_baseline'])} | {_fmt(row['diff'])} | {_fmt(row['lo'])} to {_fmt(row['hi'])} | {int(row['n_event'])} | {int(row['n_baseline'])} |"
            )
        lines.append("")
    if its_path.exists():
        its = pd.read_csv(its_path)
        primary = its[its["primary"] == True]  # noqa: E712
        lines.append("### Interrupted series")
        lines.append("")
        lines.append(
            "Monthly shares from 12 months before the first event month through 6 months after, weighted by the number of labeled posts. "
            "The post coefficient is the level shift at the event month, after a linear pre-trend and a change in slope. "
            "Months with fewer than 5 labeled posts are left out. A missing month is skipped, not entered as zero."
        )
        lines.append("")
        lines.append("| Event | Audience | Level shift | 95% interval | Months | Posts | Post months | Status |")
        lines.append("|---|---|---:|---|---:|---:|---:|---|")
        for _, row in primary.sort_values(["event_id", "audience"]).iterrows():
            lines.append(
                f"| {row['event_id']} | {row['audience']} | {_fmt(row['post'])} | {_fmt(row['post_lo'])} to {_fmt(row['post_hi'])} | {int(row['n_months'])} | {int(row['n_posts'])} | {int(row['n_months_post'])} | {row['status']} |"
            )
        lines.append("")
    lines.append(
        "Figures: `framing_barrier_shares_monthly.png`, `framing_event_study.png`, `framing_its_level.png`. "
        "Event-study points are the share in that relative month minus the share in the month before the event (`framing_event_study.csv`)."
    )
    lines.append("")
    return "\n".join(lines)


def upsert_framing_findings(findings_path, out_dir) -> None:
    from pathlib import Path

    findings_path = Path(findings_path)
    start = "<!-- framing-a-start -->"
    end = "<!-- framing-a-end -->"
    section = render_framing_markdown(out_dir)
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    existing = findings_path.read_text() if findings_path.exists() else ""
    if start in existing and end in existing:
        head, rest = existing.split(start, 1)
        _, tail = rest.split(end, 1)
        findings_path.write_text(head.rstrip() + "\n\n" + block + tail.lstrip("\n"))
        return
    findings_path.write_text(existing.rstrip() + "\n\n" + block)
