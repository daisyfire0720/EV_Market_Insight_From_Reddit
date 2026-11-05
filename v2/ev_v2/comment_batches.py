"""Choose which comments to label first so the GPU run stays inside 30 hours.

The planning rates are the slow ends of the bands in ``RUN_ON_MY_GPU.md``:
28 hours for 22,827 submissions on 14B, and 14 hours on 7B. Slots are the
extra comments that still finish at or under 30 hours at that rate. No text
is invented. Bodies stay out of the committed files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ev_v2.comment_coverage import WINDOW_DAYS, _cutoff_utc, authors_on_both_sides
from ev_v2.rdit import CUTOFFS

N_SUBMISSIONS = 22827
BUDGET_HOURS = 30.0
PLAN_HOURS_14B = 28.0
PLAN_HOURS_7B = 14.0


def comment_slots(plan_hours: float, n_submissions: int = N_SUBMISSIONS, budget_hours: float = BUDGET_HOURS) -> int:
    """How many comments fit with the submissions inside the budget, rounding down."""
    if plan_hours <= 0 or n_submissions <= 0:
        return 0
    total = int(budget_hours * n_submissions // plan_hours)
    return max(0, total - n_submissions)


def hours_at(n_items: int, plan_hours: float, n_submissions: int = N_SUBMISSIONS) -> float:
    if n_submissions <= 0:
        return 0.0
    return plan_hours * n_items / n_submissions


def _crossed_cutoffs(events: pd.DataFrame) -> dict[str, set[str]]:
    """author_hash -> cutoff ids where that author has an item on both sides."""
    found: dict[str, set[str]] = {}
    for spec in CUTOFFS:
        both = authors_on_both_sides(events, _cutoff_utc(spec["cutoff"]), WINDOW_DAYS)
        for digest in both:
            found.setdefault(digest, set()).add(spec["cutoff_id"])
    return found


def _distance_days(created: float, cutoff_ids: set[str]) -> float:
    if not cutoff_ids:
        specs = CUTOFFS
    else:
        specs = [spec for spec in CUTOFFS if spec["cutoff_id"] in cutoff_ids]
    created_days = created / 86400.0
    best = None
    for spec in specs:
        cutoff_days = _cutoff_utc(spec["cutoff"]) / 86400.0
        distance = abs(created_days - cutoff_days)
        if best is None or distance < best:
            best = distance
    return float(best if best is not None else 10_000.0)


def rank_comments(comments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Cutoff-crossing authors first, then comments closest to a cutoff they cross."""
    frame = comments.copy()
    if frame.empty:
        frame["cutoff_crossing"] = []
        frame["distance_days"] = []
        frame["priority_rank"] = []
        return frame
    frame["author_hash"] = frame["author_hash"].fillna("").astype(str)
    frame["created_utc"] = pd.to_numeric(frame["created_utc"], errors="coerce")
    crossed = _crossed_cutoffs(events)
    frame["cutoff_crossing"] = frame["author_hash"].map(lambda digest: int(digest in crossed and digest != ""))
    frame["distance_days"] = [
        _distance_days(float(created), crossed.get(digest, set()))
        if created == created
        else 10_000.0
        for digest, created in zip(frame["author_hash"], frame["created_utc"])
    ]
    frame = frame.sort_values(
        ["cutoff_crossing", "distance_days", "id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["priority_rank"] = frame.index + 1
    return frame


def assign_batches(comments: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Batch 1 fits in 30 hours. 14B when the priority set fits; otherwise 7B."""
    ranked = rank_comments(comments, events)
    slots_14b = comment_slots(PLAN_HOURS_14B)
    slots_7b = comment_slots(PLAN_HOURS_7B)
    n_priority = int(ranked["cutoff_crossing"].sum()) if len(ranked) and "cutoff_crossing" in ranked.columns else 0
    if n_priority <= slots_14b:
        plan = "14b"
        cap = slots_14b
        plan_hours = PLAN_HOURS_14B
    else:
        plan = "7b"
        cap = slots_7b
        plan_hours = PLAN_HOURS_7B
    cap = min(cap, len(ranked))
    batch1 = ranked.iloc[:cap].copy()
    batch2 = ranked.iloc[cap:].copy()
    batch1["batch"] = 1
    batch2["batch"] = 2
    batch1["model_plan"] = plan
    batch2["model_plan"] = plan
    n1 = len(batch1)
    info = {
        "model_plan": plan,
        "n_submissions": N_SUBMISSIONS,
        "n_comments": int(len(ranked)),
        "n_priority": n_priority,
        "slots_14b": slots_14b,
        "slots_7b": slots_7b,
        "n_batch1": n1,
        "n_batch2": int(len(batch2)),
        "plan_hours_submissions": plan_hours,
        "hours_submissions_plus_batch1": round(hours_at(N_SUBMISSIONS + n1, plan_hours), 2),
        "hours_14b_slow_batch1": round(hours_at(N_SUBMISSIONS + n1, PLAN_HOURS_14B), 2),
        "hours_7b_slow_batch1": round(hours_at(N_SUBMISSIONS + n1, PLAN_HOURS_7B), 2),
        "hours_14b_slow_all_comments": round(hours_at(N_SUBMISSIONS + len(ranked), PLAN_HOURS_14B), 2),
        "validated": False,
    }
    return batch1, batch2, info


META_COLUMNS = [
    "id",
    "author_hash",
    "created_utc",
    "subreddit",
    "link_id",
    "score",
    "text_type",
    "cutoff_crossing",
    "distance_days",
    "priority_rank",
    "batch",
    "model_plan",
]


def write_batches(comments: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch1, batch2, info = assign_batches(comments, events)
    keep = [column for column in META_COLUMNS if column in batch1.columns or column in ("batch", "model_plan")]
    for frame, name in ((batch1, "comment_label_batch1.csv"), (batch2, "comment_label_batch2.csv")):
        columns = [column for column in META_COLUMNS if column in frame.columns]
        frame.loc[:, columns].to_csv(out_dir / name, index=False)
    pd.DataFrame([info]).to_csv(out_dir / "comment_label_plan.csv", index=False)
    return info


def materialize_label_input(batch_csv: Path, cache_jsonl: Path, out_csv: Path) -> int:
    """Join committed ids to the gitignored body cache. The output is local only."""
    import json

    batch = pd.read_csv(batch_csv)
    bodies: dict[str, str] = {}
    path = Path(cache_jsonl)
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            bodies[str(row.get("id") or "")] = str(row.get("body") or "")
    batch["id"] = batch["id"].astype(str)
    batch["body"] = batch["id"].map(lambda doc_id: bodies.get(doc_id, ""))
    batch["title"] = ""
    if "text_type" not in batch.columns:
        batch["text_type"] = "comment"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(out_csv, index=False)
    return int(len(batch))
