"""Block bootstrap for Framing A contrasts.

Enthusiast posts are resampled by sampled week. Shopper posts are resampled
by calendar month, which is the dump's time block. A one-month window has a
single month block, so that interval is flagged and a subreddit-by-month
resample is reported beside it.

The point estimate is unchanged. Only the interval moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_v2.events import EVENTS, event_months
from ev_v2.framing import (
    AUDIENCES,
    ENTHUSIAST,
    FRAMING_EVENT_IDS,
    ITS_POST_MONTHS,
    ITS_PRE_MONTHS,
    PRIMARY_BARRIER,
    SHOPPER,
    barrier_column,
    fit_interrupted_series,
    monthly_counts,
    proportion_difference,
    shift_month,
)

N_BOOT = 2000
SEED = 20261019


def cluster_arrays(frame: pd.DataFrame, on_col: str, block_col: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([]), np.array([])
    work = pd.DataFrame(
        {
            "block": frame[block_col].astype(str).to_numpy(),
            "k": frame[on_col].fillna(False).astype(bool).astype(int).to_numpy(),
        }
    )
    grouped = work.groupby("block", sort=False)["k"].agg(k="sum", n="size")
    return grouped["k"].to_numpy(dtype=float), grouped["n"].to_numpy(dtype=float)


def bootstrap_difference(k1: np.ndarray, n1: np.ndarray, k2: np.ndarray, n2: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Resample each side's clusters with replacement and return p1 - p2."""
    draws = np.full(n_boot, np.nan)
    if len(k1) == 0 or len(k2) == 0:
        return draws
    e_take = rng.integers(0, len(k1), size=(n_boot, len(k1)))
    b_take = rng.integers(0, len(k2), size=(n_boot, len(k2)))
    ke = k1[e_take].sum(axis=1)
    ne = n1[e_take].sum(axis=1)
    kb = k2[b_take].sum(axis=1)
    nb = n2[b_take].sum(axis=1)
    ok = (ne > 0) & (nb > 0)
    draws[ok] = ke[ok] / ne[ok] - kb[ok] / nb[ok]
    return draws


def percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    clean = draws[np.isfinite(draws)]
    if len(clean) < 50:
        return float("nan"), float("nan")
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def _months_for(event_id: str) -> tuple[list[str], list[str], list[str]]:
    catalog = {event.event_id: event for event in EVENTS}
    months = event_months(catalog[event_id])
    prior = [shift_month(months[0], -step) for step in (3, 2, 1)]
    year_ago = [shift_month(month, -12) for month in months]
    return months, prior, year_ago


def contrast_bootstrap_row(
    frame: pd.DataFrame,
    on_col: str,
    event_id: str,
    barrier: str,
    audience: str,
    event_m: list[str],
    base_m: list[str],
    baseline_name: str,
    block_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    group = frame[frame["audience"] == audience]
    event = group[group["month"].isin(event_m)]
    base = group[group["month"].isin(base_m)]
    k1 = int(event[on_col].fillna(False).sum()) if len(event) else 0
    k2 = int(base[on_col].fillna(False).sum()) if len(base) else 0
    indep = proportion_difference(k1, len(event), k2, len(base))
    ek, en = cluster_arrays(event, on_col, block_col)
    bk, bn = cluster_arrays(base, on_col, block_col)
    draws = bootstrap_difference(ek, en, bk, bn, rng, n_boot)
    lo, hi = percentile_interval(draws)
    n_event_m = int(event["month"].nunique()) if len(event) else 0
    n_base_m = int(base["month"].nunique()) if len(base) else 0
    thin = n_event_m < len(event_m) or n_base_m < len(base_m)
    few_blocks = len(ek) < 2 or len(bk) < 2
    return {
        "event_id": event_id,
        "barrier": barrier,
        "audience": audience,
        "baseline": baseline_name,
        "primary": PRIMARY_BARRIER.get(event_id) == barrier,
        "block": block_col,
        "diff": indep["diff"],
        "indep_lo": indep["lo"],
        "indep_hi": indep["hi"],
        "boot_lo": lo,
        "boot_hi": hi,
        "indep_width": indep["hi"] - indep["lo"] if np.isfinite(indep["hi"]) else float("nan"),
        "boot_width": hi - lo if np.isfinite(hi) else float("nan"),
        "indep_excludes_zero": bool(np.isfinite(indep["lo"]) and (indep["lo"] > 0 or indep["hi"] < 0)),
        "boot_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
        "n_event": int(len(event)),
        "n_baseline": int(len(base)),
        "n_blocks_event": int(len(ek)),
        "n_blocks_baseline": int(len(bk)),
        "months_event_observed": n_event_m,
        "months_event_intended": int(len(event_m)),
        "months_baseline_observed": n_base_m,
        "months_baseline_intended": int(len(base_m)),
        "thin_window": bool(thin),
        "few_blocks": bool(few_blocks),
        "n_boot": int(np.isfinite(draws).sum()),
    }


def block_bootstrap_contrasts(
    frame: pd.DataFrame,
    barriers: list[str],
    enthusiast_block: str = "week_block",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Primary and other barriers, both baselines, both block definitions for shoppers."""
    rng = np.random.default_rng(seed)
    rows = []
    work = frame.copy()
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    if "subreddit_month" not in work.columns:
        work["subreddit_month"] = work["subreddit"].astype(str) + "|" + work["month"]
    if enthusiast_block not in work.columns:
        work[enthusiast_block] = work["subreddit_month"]
    for event_id in FRAMING_EVENT_IDS:
        months, prior, year_ago = _months_for(event_id)
        for barrier in barriers:
            column = barrier_column(barrier)
            for audience in AUDIENCES:
                blocks = [enthusiast_block] if audience == ENTHUSIAST else ["month", "subreddit_month"]
                # Shopper month column is named month, which is also the calendar field.
                # Cluster on a dedicated copy so groupby does not collide with the filter.
                if audience == SHOPPER and "month_block" not in work.columns:
                    work["month_block"] = work["month"]
                if audience == SHOPPER:
                    blocks = ["month_block", "subreddit_month"]
                for baseline_name, base_m in (("prior_3_months", prior), ("year_ago", year_ago)):
                    for block in blocks:
                        rows.append(
                            contrast_bootstrap_row(
                                work, column, event_id, barrier, audience, months, base_m, baseline_name, block, rng, n_boot
                            )
                        )
    out = pd.DataFrame(rows)
    out["reported"] = False
    for keys, block in out.groupby(["event_id", "barrier", "audience", "baseline"], sort=False):
        index = block.index
        month_rows = block[block["block"] == "month_block"]
        if len(month_rows) and not bool(month_rows.iloc[0]["few_blocks"]):
            chosen = month_rows.index[0]
        else:
            # Prefer the week block for enthusiasts, and subreddit-month when a
            # calendar month cannot be resampled (one-month windows).
            week = block[block["block"] == enthusiast_block]
            sub = block[block["block"] == "subreddit_month"]
            if len(week):
                chosen = week.index[0]
            elif len(sub):
                chosen = sub.index[0]
            else:
                chosen = index[0]
        out.loc[chosen, "reported"] = True
    return out


def bootstrap_interrupted_series(
    frame: pd.DataFrame,
    barriers: list[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    min_n: int = 5,
) -> pd.DataFrame:
    """Resample the months that enter the regression, then refit the level shift."""
    rng = np.random.default_rng(seed)
    rows = []
    catalog = {event.event_id: event for event in EVENTS}
    for event_id in FRAMING_EVENT_IDS:
        first = event_months(catalog[event_id])[0]
        for barrier in barriers:
            column = barrier_column(barrier)
            counts = monthly_counts(frame, column)
            for audience in AUDIENCES:
                monthly = counts[counts["audience"] == audience]
                fit = fit_interrupted_series(monthly, first, min_n=min_n)
                draws = _resample_its(monthly, first, rng, n_boot, min_n)
                lo, hi = percentile_interval(draws)
                rows.append(
                    {
                        "event_id": event_id,
                        "barrier": barrier,
                        "audience": audience,
                        "primary": PRIMARY_BARRIER.get(event_id) == barrier,
                        "post": fit["post"],
                        "hc1_lo": fit["post_lo"],
                        "hc1_hi": fit["post_hi"],
                        "boot_lo": lo,
                        "boot_hi": hi,
                        "hc1_excludes_zero": bool(np.isfinite(fit["post_lo"]) and (fit["post_lo"] > 0 or fit["post_hi"] < 0)),
                        "boot_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                        "n_months": fit["n_months"],
                        "n_posts": fit["n_posts"],
                        "n_months_post": fit["n_months_post"],
                        "status": fit["status"],
                        "n_boot": int(np.isfinite(draws).sum()),
                        "window_start": fit.get("window_start", shift_month(first, -ITS_PRE_MONTHS)),
                        "window_end": fit.get("window_end", shift_month(first, ITS_POST_MONTHS)),
                    }
                )
    return pd.DataFrame(rows)


def _resample_its(monthly: pd.DataFrame, event_month: str, rng: np.random.Generator, n_boot: int, min_n: int) -> np.ndarray:
    start = shift_month(event_month, -ITS_PRE_MONTHS)
    end = shift_month(event_month, ITS_POST_MONTHS)
    work = monthly.copy()
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    work = work[(work["month"] >= start) & (work["month"] <= end) & (work["n"] >= min_n)].reset_index(drop=True)
    draws = np.full(n_boot, np.nan)
    if len(work) < 8:
        return draws
    n = len(work)
    for i in range(n_boot):
        take = rng.integers(0, n, size=n)
        sample = work.iloc[take]
        # A month drawn twice keeps both rows, so that month has double weight.
        fit = fit_interrupted_series(sample, event_month, min_n=min_n)
        if fit["status"] == "ok":
            draws[i] = fit["post"]
    return draws


def window_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Which event and baseline windows have fewer sampled months than the design asks for."""
    rows = []
    work = frame.copy()
    work["month"] = work["month"].astype(str).str.slice(0, 7)
    for event_id in FRAMING_EVENT_IDS:
        months, prior, year_ago = _months_for(event_id)
        for audience in AUDIENCES:
            group = work[work["audience"] == audience]
            for name, intended in (("event", months), ("prior_3_months", prior), ("year_ago", year_ago)):
                observed = sorted(set(group.loc[group["month"].isin(intended), "month"]))
                rows.append(
                    {
                        "event_id": event_id,
                        "audience": audience,
                        "window": name,
                        "intended_months": len(intended),
                        "observed_months": len(observed),
                        "missing_months": ",".join(month for month in intended if month not in observed),
                        "thin_window": len(observed) < len(intended),
                        "n_posts": int(group["month"].isin(intended).sum()),
                    }
                )
    return pd.DataFrame(rows)
