"""Regression discontinuity in time, as scaffolding on the current labels.

The outcome is a barrier indicator. The running variable is days from a
cutoff. A local linear fit on each side, with a triangular kernel, estimates
the jump. Bandwidth is chosen by even/odd day cross-validation on a small
grid. A donut drops the days next to the cutoff. Placebo cutoffs use the
same bandwidth and outcome. None of this is an effect. The labels are the
small model until a human-coded file replaces them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

BANDWIDTHS = (30, 60, 90, 180)
DONUT_DAYS = 7
PLACEBO_SHIFTS = (-180, 180)
MIN_SIDE_POSTS = 30
MIN_SIDE_DAYS = 5

# Cutoffs Daisy named for the discontinuity scaffold.
# IRA credit rules: vehicles placed in service after 31 Dec 2022 under the
# credit as rewritten by Pub. L. 117-169 (26 U.S.C. § 30D). The other three
# dates are the ones already sourced in research_design.md.
CUTOFFS = (
    {
        "cutoff_id": "ira_signing_2022",
        "cutoff": date(2022, 8, 16),
        "outcome": "barrier_purchase_incentives_or_tax_credits_on",
        "outcome_label": "purchase incentives or tax credits",
    },
    {
        "cutoff_id": "ira_credit_rules_2023",
        "cutoff": date(2023, 1, 1),
        "outcome": "barrier_purchase_incentives_or_tax_credits_on",
        "outcome_label": "purchase incentives or tax credits",
    },
    {
        "cutoff_id": "tesla_price_cut_2023",
        "cutoff": date(2023, 1, 13),
        "outcome": "barrier_vehicle_price_or_total_cost_on",
        "outcome_label": "vehicle price or total cost",
    },
    {
        "cutoff_id": "credit_end_2025",
        "cutoff": date(2025, 9, 30),
        "outcome": "barrier_purchase_incentives_or_tax_credits_on",
        "outcome_label": "purchase incentives or tax credits",
    },
)


def _as_day(value) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(stamp):
        return None
    if stamp > 10_000_000_000:
        stamp = stamp / 1000.0
    return datetime.fromtimestamp(stamp, tz=timezone.utc).date()


def _jump(day: np.ndarray, y: np.ndarray, weight: np.ndarray, bandwidth: int, donut: int) -> dict:
    """Local linear jump. ``day`` is days from the cutoff."""
    left = (day < -donut) & (day >= -bandwidth)
    right = (day > donut) & (day <= bandwidth)

    def _side(mask: np.ndarray) -> tuple[float, int, int]:
        if int(mask.sum()) < 2:
            return float("nan"), int(weight[mask].sum()) if mask.any() else 0, int(mask.sum())
        x = day[mask].astype(float)
        outcome = y[mask].astype(float)
        w = weight[mask].astype(float)
        scale = max(bandwidth, 1)
        kernel = np.maximum(0.0, 1.0 - np.abs(x) / scale)
        w = w * kernel
        if float(w.sum()) <= 0:
            return float("nan"), 0, int(mask.sum())
        design = np.column_stack([np.ones(len(x)), x])
        sw = np.sqrt(w)
        beta, _, _, _ = np.linalg.lstsq(design * sw[:, None], outcome * sw, rcond=None)
        return float(beta[0]), int(round(float(weight[mask].sum()))), int(mask.sum())

    left_level, n_left, days_left = _side(left)
    right_level, n_right, days_right = _side(right)
    tau = right_level - left_level
    thin = n_left < MIN_SIDE_POSTS or n_right < MIN_SIDE_POSTS or days_left < MIN_SIDE_DAYS or days_right < MIN_SIDE_DAYS
    return {
        "tau": tau,
        "left_level": left_level,
        "right_level": right_level,
        "n_left": n_left,
        "n_right": n_right,
        "days_left": days_left,
        "days_right": days_right,
        "thin": bool(thin) or not np.isfinite(tau),
    }


def _daily(frame: pd.DataFrame, outcome: str, cutoff: date) -> pd.DataFrame:
    work = frame.dropna(subset=["created_day"]).copy()
    work["day"] = work["created_day"].map(lambda item: (item - cutoff).days)
    work["y"] = work[outcome].fillna(False).astype(bool).astype(int)
    if work.empty:
        return pd.DataFrame(columns=["day", "y", "n"])
    grouped = work.groupby("day", sort=True)["y"].agg(k="sum", n="size")
    grouped["y"] = grouped["k"] / grouped["n"]
    return grouped.reset_index()


def choose_bandwidth(daily: pd.DataFrame, grid: tuple[int, ...] = BANDWIDTHS) -> int:
    """Even/odd day cross-validation. Falls back to the middle of the grid."""
    if daily.empty:
        return 90
    best_h = 90
    best_score = float("inf")
    day = daily["day"].to_numpy()
    y = daily["y"].to_numpy()
    n = daily["n"].to_numpy(dtype=float)
    for bandwidth in grid:
        even = (day % 2) == 0
        odd = ~even
        if int(even.sum()) < 4 or int(odd.sum()) < 4:
            continue
        fit = _jump(day[even], y[even], n[even], bandwidth, donut=0)
        if not np.isfinite(fit["tau"]):
            continue
        # Predict the odd days from the two boundary levels, without a slope,
        # which is enough to punish a bandwidth that is only noise.
        x = day[odd]
        pred = np.where(x >= 0, fit["right_level"], fit["left_level"])
        if not np.isfinite(pred).all():
            continue
        resid = y[odd] - pred
        score = float(np.average(resid**2, weights=n[odd]))
        if score < best_score:
            best_score = score
            best_h = int(bandwidth)
    return best_h


def _week_interval(work: pd.DataFrame, outcome: str, cutoff: date, bandwidth: int, donut: int, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile interval for one jump. Weeks are resampled; the bandwidth is not re-chosen."""
    posts = work.dropna(subset=["created_day"]).copy()
    if posts.empty or n_boot <= 0:
        return float("nan"), float("nan")
    posts["day"] = posts["created_day"].map(lambda item: (item - cutoff).days)
    posts["y"] = posts[outcome].fillna(False).astype(bool).astype(int)
    posts = posts[(posts["day"] >= -bandwidth) & (posts["day"] <= bandwidth) & (posts["day"].abs() > donut)]
    if posts.empty:
        return float("nan"), float("nan")
    posts["week"] = (posts["day"] // 7).astype(int)
    weeks = posts["week"].unique()
    if len(weeks) < 4:
        return float("nan"), float("nan")
    grouped = {int(week): block for week, block in posts.groupby("week")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        take = rng.choice(weeks, size=len(weeks), replace=True)
        sample = pd.concat([grouped[int(week)] for week in take], ignore_index=True)
        daily = sample.groupby("day")["y"].agg(k="sum", n="size").reset_index()
        daily["y"] = daily["k"] / daily["n"]
        fit = _jump(daily["day"].to_numpy(), daily["y"].to_numpy(), daily["n"].to_numpy(dtype=float), bandwidth, donut)
        if np.isfinite(fit["tau"]):
            draws.append(fit["tau"])
    if len(draws) < max(20, n_boot // 5):
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def rdit_table(frame: pd.DataFrame, audience: str | None = None, n_boot: int = 0, seed: int = 20261021) -> pd.DataFrame:
    """One row per cutoff, for the main fit, the bandwidth grid, the donut, and the placebos.

    ``n_boot`` resamples week clusters for the main fit and the donut only.
    Zero leaves the interval empty. The bandwidth chosen on the original
    sample is held fixed inside the resample.
    """
    work = frame.copy()
    if audience is not None and "audience" in work.columns:
        work = work[work["audience"] == audience]
    if "created_day" not in work.columns:
        work["created_day"] = work["created_utc"].map(_as_day) if "created_utc" in work.columns else None
    rows = []
    for spec in CUTOFFS:
        outcome = spec["outcome"]
        if outcome not in work.columns:
            continue
        daily = _daily(work, outcome, spec["cutoff"])
        bandwidth = choose_bandwidth(daily)
        specs = [("main", spec["cutoff"], 0), ("donut", spec["cutoff"], DONUT_DAYS)]
        for shift in PLACEBO_SHIFTS:
            specs.append((f"placebo_{shift:+d}d", spec["cutoff"] + timedelta(days=shift), 0))
        for kind, cutoff, donut in specs:
            used = daily if cutoff == spec["cutoff"] else _daily(work, outcome, cutoff)
            if used.empty:
                fit = {"tau": float("nan"), "left_level": float("nan"), "right_level": float("nan"), "n_left": 0, "n_right": 0, "days_left": 0, "days_right": 0, "thin": True}
            else:
                fit = _jump(used["day"].to_numpy(), used["y"].to_numpy(), used["n"].to_numpy(dtype=float), bandwidth, donut)
            lo, hi = (float("nan"), float("nan"))
            if n_boot > 0 and kind in {"main", "donut"} and not fit["thin"]:
                lo, hi = _week_interval(
                    work,
                    outcome,
                    cutoff,
                    bandwidth,
                    donut,
                    n_boot,
                    seed + cutoff.toordinal() + (0 if kind == "main" else 17),
                )
            rows.append(
                {
                    "cutoff_id": spec["cutoff_id"],
                    "cutoff": cutoff.isoformat(),
                    "audience": audience or "all",
                    "outcome": spec["outcome_label"],
                    "fit": kind,
                    "bandwidth_days": bandwidth,
                    "donut_days": donut,
                    "tau_lo": lo,
                    "tau_hi": hi,
                    "validated": False,
                    **fit,
                }
            )
        if not daily.empty:
            for alt in BANDWIDTHS:
                alt_fit = _jump(daily["day"].to_numpy(), daily["y"].to_numpy(), daily["n"].to_numpy(dtype=float), alt, 0)
                rows.append(
                    {
                        "cutoff_id": spec["cutoff_id"],
                        "cutoff": spec["cutoff"].isoformat(),
                        "audience": audience or "all",
                        "outcome": spec["outcome_label"],
                        "fit": f"bandwidth_{alt}",
                        "bandwidth_days": alt,
                        "donut_days": 0,
                        "tau_lo": float("nan"),
                        "tau_hi": float("nan"),
                        "validated": False,
                        **alt_fit,
                    }
                )
    return pd.DataFrame(rows)
