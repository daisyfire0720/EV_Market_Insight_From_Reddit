"""FINDINGS section for the thin-baseline fill.

Numbers are read from the before/after tables. The section says the labels
are still unvalidated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ev_v2.round5_report import upsert_marked
from ev_v2.thin_baselines import FILL_LIMIT, LABEL_BUDGET_MINUTES, max_posts


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def _excludes(lo, hi) -> bool:
    if lo is None or hi is None:
        return False
    if isinstance(lo, float) and np.isnan(lo):
        return False
    if isinstance(hi, float) and np.isnan(hi):
        return False
    return bool(lo > 0 or hi < 0)


def moved_rows(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Primary reported rows whose bootstrap interval changed its relation to zero."""
    keys = [
        key
        for key in ("event_id", "barrier", "audience", "baseline", "block")
        if key in before.columns and key in after.columns
    ]
    left = before.copy()
    right = after.copy()
    if "reported" in left.columns:
        left = left[left["reported"] == True]  # noqa: E712
    if "reported" in right.columns:
        right = right[right["reported"] == True]  # noqa: E712
    if "primary" in left.columns:
        left = left[left["primary"] == True]  # noqa: E712
    if "primary" in right.columns:
        right = right[right["primary"] == True]  # noqa: E712
    merged = left.merge(right, on=keys, suffixes=("_before", "_after"))
    if merged.empty:
        return merged
    before_ex = [_excludes(lo, hi) for lo, hi in zip(merged["boot_lo_before"], merged["boot_hi_before"])]
    after_ex = [_excludes(lo, hi) for lo, hi in zip(merged["boot_lo_after"], merged["boot_hi_after"])]
    merged = merged.assign(excluded_before=before_ex, excluded_after=after_ex)
    return merged[merged["excluded_before"] != merged["excluded_after"]].reset_index(drop=True)


def render_round6_markdown(out_dir) -> str:
    out_dir = Path(out_dir)
    cap = max_posts()
    lines = [
        "## Round 6 thin baselines (unvalidated)",
        "",
        "The labels are still the small open model. No person has coded them. "
        f"This round filled only the enthusiast months Round 5 flagged, under a cap of "
        f"**{cap}** posts ({FILL_LIMIT} per subreddit-window) and **{LABEL_BUDGET_MINUTES}** minutes of CPU labeling. "
        "The 282,348-post dense pull was not started. Differences are still not effects.",
        "",
    ]
    manifest_path = out_dir / "thin_baseline_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        lines.append(
            f"Pulled {int(manifest.get('n_downloaded', 0))} submissions, kept {int(manifest.get('n_kept', 0))} after the bot and length filter, "
            f"and appended {int(manifest.get('n_appended', 0))} new rows "
            f"({int(manifest.get('n_labels_before', 0))} to {int(manifest.get('n_labels_after', 0))}). "
            f"Labeling took {manifest.get('label_minutes', '')} minutes. "
            f"Windows requested: {int(manifest.get('n_windows', 0))}."
        )
        lines.append("")
        by_month = manifest.get("kept_by_month") or {}
        if by_month:
            lines.append("| Month | New labeled posts |")
            lines.append("|---|---:|")
            for month in sorted(by_month):
                lines.append(f"| {month} | {int(by_month[month])} |")
            lines.append("")
    before = out_dir / "framing_block_bootstrap_before_fill.csv"
    after = out_dir / "framing_block_bootstrap.csv"
    if before.exists() and after.exists():
        lines.append("### What moved in the block-bootstrap contrasts")
        lines.append("")
        lines.extend(_move_prose(pd.read_csv(before), pd.read_csv(after), "diff"))
        lines.append("")
        lines.extend(_still_excludes(pd.read_csv(after)))
        lines.append("")
        lines.extend(_large_point_moves(pd.read_csv(before), pd.read_csv(after)))
        lines.append("")
    its_before = out_dir / "framing_its_bootstrap_before_fill.csv"
    its_after = out_dir / "framing_its_bootstrap.csv"
    if its_before.exists() and its_after.exists():
        lines.append("### What moved in the interrupted series")
        lines.append("")
        lines.extend(_move_prose(pd.read_csv(its_before), pd.read_csv(its_after), "post", its=True))
        lines.append("")
    flags = out_dir / "framing_window_flags.csv"
    flags_before = out_dir / "framing_window_flags_before_fill.csv"
    if flags.exists():
        thin = pd.read_csv(flags)
        enth = thin[thin["audience"] == "enthusiast"].copy()
        lines.append("### Enthusiast window coverage")
        lines.append("")
        if flags_before.exists():
            old = pd.read_csv(flags_before)
            old = old[old["audience"] == "enthusiast"]
            merged = enth.merge(
                old,
                on=["event_id", "window"],
                suffixes=("", "_before"),
            )
            lines.append("| Event | Window | Posts before | Posts after | Missing after |")
            lines.append("|---|---|---:|---:|---|")
            for _, row in merged.iterrows():
                missing = row["missing_months"] if isinstance(row["missing_months"], str) else ""
                if pd.isna(row["missing_months"]):
                    missing = ""
                lines.append(
                    f"| {row['event_id']} | {row['window']} | {int(row['n_posts_before'])} | {int(row['n_posts'])} | {missing} |"
                )
        else:
            lines.append("| Event | Window | Posts | Missing |")
            lines.append("|---|---|---:|---|")
            for _, row in enth.iterrows():
                missing = "" if pd.isna(row["missing_months"]) else row["missing_months"]
                lines.append(f"| {row['event_id']} | {row['window']} | {int(row['n_posts'])} | {missing} |")
        still = enth[enth["thin_window"] == True]  # noqa: E712
        lines.append("")
        if still.empty:
            lines.append("No enthusiast event, prior-three, or year-ago window is missing a designed month after the fill.")
        else:
            lines.append("Windows that are still missing a designed month:")
            lines.append("")
            for _, row in still.iterrows():
                lines.append(f"- {row['event_id']} {row['window']}: {row['missing_months']} ({int(row['n_posts'])} posts).")
        lines.append("")
    lines.append(
        "The manuscript draft that uses these tables is `v2/paper/manuscript.md`. "
        "Every share in it is preliminary until the annotation sheet is coded."
    )
    lines.append("")
    return "\n".join(lines)


def _still_excludes(after: pd.DataFrame) -> list[str]:
    rows = after[(after["primary"] == True) & (after["reported"] == True) & (after["boot_excludes_zero"] == True)]  # noqa: E712
    if rows.empty:
        return ["No primary reported block-bootstrap interval excludes zero after the fill."]
    lines = ["Primary reported intervals that still exclude zero after the fill:"]
    lines.append("")
    for _, row in rows.iterrows():
        lines.append(
            f"- {row['event_id']} / {row['audience']} / {row['baseline']}: {_fmt(row['diff'])} "
            f"({_fmt(row['boot_lo'])} to {_fmt(row['boot_hi'])}, baseline n={int(row['n_baseline'])})."
        )
    return lines


def _large_point_moves(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    """Point estimates that moved by more than 0.02 even if the zero decision did not."""
    keys = [key for key in ("event_id", "barrier", "audience", "baseline", "block") if key in before.columns and key in after.columns]
    left = before[(before["primary"] == True) & (before["reported"] == True)]  # noqa: E712
    right = after[(after["primary"] == True) & (after["reported"] == True)]  # noqa: E712
    merged = left.merge(right, on=keys, suffixes=("_before", "_after"))
    lines = ["Point estimates that moved by more than 0.02 and did not change their relation to zero:"]
    lines.append("")
    any_row = False
    for _, row in merged.iterrows():
        delta = float(row["diff_after"]) - float(row["diff_before"])
        if abs(delta) <= 0.02:
            continue
        if bool(row["boot_excludes_zero_before"]) != bool(row["boot_excludes_zero_after"]):
            continue
        any_row = True
        lines.append(
            f"- {row['event_id']} / {row['audience']} / {row['baseline']}: {_fmt(row['diff_before'])} to {_fmt(row['diff_after'])} "
            f"(baseline n {int(row['n_baseline_before'])} to {int(row['n_baseline_after'])})."
        )
    if not any_row:
        return ["No primary point estimate moved by more than 0.02 without also changing whether zero is inside the interval."]
    return lines


def _move_prose(before: pd.DataFrame, after: pd.DataFrame, point_col: str, its: bool = False) -> list[str]:
    moved = moved_rows(before, after)
    lines = []
    if moved.empty:
        lines.append("No primary reported interval changed whether it includes zero.")
        return lines
    lines.append(
        f"{len(moved)} primary reported interval"
        f"{'s' if len(moved) != 1 else ''} changed whether zero is inside the bootstrap interval."
    )
    lines.append("")
    if its:
        lines.append("| Event | Audience | Before | After | Posts before | Posts after | Zero before | Zero after |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
    else:
        lines.append("| Event | Audience | Baseline | Before | After | Baseline n before | Baseline n after | Zero before | Zero after |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
    for _, row in moved.iterrows():
        zero_before = "excludes 0" if row["excluded_before"] else "includes 0"
        zero_after = "excludes 0" if row["excluded_after"] else "includes 0"
        if its:
            lines.append(
                f"| {row['event_id']} | {row['audience']} | {_fmt(row[point_col + '_before'])} "
                f"({_fmt(row['boot_lo_before'])} to {_fmt(row['boot_hi_before'])}) | "
                f"{_fmt(row[point_col + '_after'])} "
                f"({_fmt(row['boot_lo_after'])} to {_fmt(row['boot_hi_after'])}) | "
                f"{int(row['n_posts_before'])} | {int(row['n_posts_after'])} | {zero_before} | {zero_after} |"
            )
        else:
            lines.append(
                f"| {row['event_id']} | {row['audience']} | {row['baseline']} | {_fmt(row[point_col + '_before'])} "
                f"({_fmt(row['boot_lo_before'])} to {_fmt(row['boot_hi_before'])}) | "
                f"{_fmt(row[point_col + '_after'])} "
                f"({_fmt(row['boot_lo_after'])} to {_fmt(row['boot_hi_after'])}) | "
                f"{int(row['n_baseline_before'])} | {int(row['n_baseline_after'])} | {zero_before} | {zero_after} |"
            )
    return lines


def upsert_round6_findings(findings_path, out_dir) -> None:
    upsert_marked(
        Path(findings_path),
        "<!-- round6-start -->",
        "<!-- round6-end -->",
        render_round6_markdown(out_dir),
    )
