"""Retry, split, and checkpoint rules for Arctic Shift downloads.

The API uses HTTP 422 both for "slow down" (a fast response) and for a real
query timeout (a response after the database has worked). Those two cases
need different waits. A window that still times out is split. A checkpoint
lets a later process skip windows that already succeeded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


class FetchError(RuntimeError):
    def __init__(self, message: str, kind: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


def classify_and_backoff(
    attempt: int,
    *,
    status: int | None,
    elapsed_s: float,
    reset_header: str | None,
    error: str | None,
) -> tuple[str, float]:
    """Return ``(kind, seconds_to_sleep)``.

    ``kind`` is ``rate_limit``, ``timeout``, or ``other``. A 422 that comes
    back in under 1.5 seconds is treated as a rate limit, because that is too
    fast to be a database timeout. The reset header is then honored, capped
    at 90 seconds. A slower 422 keeps a short backoff so the caller can split
    the window instead of sleeping out the rate-limit header.
    """
    reset: float | None = None
    if reset_header:
        try:
            reset = float(reset_header)
        except ValueError:
            reset = None
    message = (error or "").lower()
    fast_reject = elapsed_s < 1.5 and (status in {422, 429} or "slow down" in message)
    if status == 429 or fast_reject:
        wait = 5.0 if reset is None else reset + 0.5
        return "rate_limit", min(wait, 90.0)
    if status == 422 or "timeout" in message:
        return "timeout", min(1.5 * (attempt + 1), 8.0)
    return "other", min(1.5 * (attempt + 1), 8.0)


def split_span(
    start: datetime, end: datetime, min_span: timedelta
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]] | None:
    """Split ``[start, end)`` in half. Return None when the span is already small."""
    if end <= start:
        return None
    if (end - start) <= min_span:
        return None
    mid = start + (end - start) / 2
    if mid <= start or mid >= end:
        return None
    return (start, mid), (mid, end)


def window_key(kind: str, subreddit: str, start: datetime, end: datetime, query: str) -> str:
    return "|".join(
        [
            kind,
            subreddit,
            start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            query,
        ]
    )


class Checkpoint:
    """JSON map of window key to status. Writes are atomic."""

    def __init__(self, path: Path):
        self.path = path
        self.windows: dict[str, dict] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            if isinstance(raw, dict) and isinstance(raw.get("windows"), dict):
                self.windows = raw["windows"]

    def get(self, key: str) -> dict | None:
        return self.windows.get(key)

    def mark(
        self,
        key: str,
        status: str,
        n: int | None = None,
        error: str | None = None,
    ) -> None:
        self.windows[key] = {
            "status": status,
            "n": n,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "windows": self.windows}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def failed_records(self) -> list[dict]:
        rows = []
        for key, rec in sorted(self.windows.items()):
            if rec.get("status") == "failed":
                rows.append({"key": key, **rec})
        return rows


@dataclass
class WindowOutcome:
    n: int | None
    status: str


def resolve_window(
    *,
    kind: str,
    subreddit: str,
    query: str,
    start: datetime,
    end: datetime,
    fetch,
    checkpoint: Checkpoint,
    min_span: timedelta,
    retry_failed: bool = False,
) -> WindowOutcome:
    """Fetch one window, splitting on query timeout.

    ``fetch(start, end)`` returns an integer count, or raises ``FetchError``.
    Completed checkpoint rows are not requested again. ``retry_failed`` recomputes
    failed and partial windows and still skips windows marked ok.
    """
    key = window_key(kind, subreddit, start, end, query)
    existing = checkpoint.get(key)
    if existing is not None:
        status = existing.get("status")
        redo = retry_failed and status in {"failed", "partial"}
        if status in {"ok", "ok_empty", "partial", "failed"} and not redo:
            return WindowOutcome(existing.get("n"), status)

    try:
        n = int(fetch(start, end))
    except FetchError as exc:
        if exc.kind == "rate_limit":
            # Leave the window uncheckpointed so a later run retries it.
            return WindowOutcome(None, "deferred")
        pieces = split_span(start, end, min_span) if exc.kind == "timeout" else None
        if pieces is None:
            checkpoint.mark(key, "failed", error=f"{exc.kind}: {exc}")
            return WindowOutcome(None, "failed")
        (left_s, left_e), (right_s, right_e) = pieces
        left = resolve_window(
            kind=kind,
            subreddit=subreddit,
            query=query,
            start=left_s,
            end=left_e,
            fetch=fetch,
            checkpoint=checkpoint,
            min_span=min_span,
            retry_failed=retry_failed,
        )
        right = resolve_window(
            kind=kind,
            subreddit=subreddit,
            query=query,
            start=right_s,
            end=right_e,
            fetch=fetch,
            checkpoint=checkpoint,
            min_span=min_span,
            retry_failed=retry_failed,
        )
        return _combine(key, left, right, checkpoint)

    status = "ok_empty" if n == 0 else "ok"
    checkpoint.mark(key, status, n=n)
    return WindowOutcome(n, status)


def _known(part: WindowOutcome) -> int | None:
    if part.status in {"ok", "ok_empty", "partial"} and part.n is not None:
        return int(part.n)
    return None


def _combine(key: str, left: WindowOutcome, right: WindowOutcome, checkpoint: Checkpoint) -> WindowOutcome:
    left_n = _known(left)
    right_n = _known(right)
    if left.status in {"ok", "ok_empty"} and right.status in {"ok", "ok_empty"}:
        total = (left_n or 0) + (right_n or 0)
        status = "ok_empty" if total == 0 else "ok"
        checkpoint.mark(key, status, n=total)
        return WindowOutcome(total, status)
    if left_n is None and right_n is None:
        checkpoint.mark(key, "failed", error="both splits failed")
        return WindowOutcome(None, "failed")
    total = (left_n or 0) + (right_n or 0)
    checkpoint.mark(key, "partial", n=total, error="at least one split failed")
    return WindowOutcome(total, "partial")
