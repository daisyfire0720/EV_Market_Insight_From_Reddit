#!/usr/bin/env python3
"""Round-5 tables: annotation draw, block bootstrap, shopper slices.

Labels stay unvalidated. The script reads ``v2/outputs/framing_labels.csv``.
It fetches post text and enthusiast timestamps from Arctic Shift when those
caches are missing. It does not invent human codes.

From the repository root::

    python v2/run_round5.py
    python v2/run_round5.py --skip-fetch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ev_v2.annotation_sample import (  # noqa: E402
    blank_sheet,
    draw_annotation_sample,
    model_key,
    prepare_label_frame,
    sample_composition,
)
from ev_v2.blocks import assign_week_block  # noqa: E402
from ev_v2.bootstrap import block_bootstrap_contrasts, bootstrap_interrupted_series, window_flags  # noqa: E402
from ev_v2.cleaning import submission_text  # noqa: E402
from ev_v2.framing import PRIMARY_BARRIER, upsert_framing_findings  # noqa: E402
from ev_v2.round5_report import upsert_round5_card, upsert_round5_findings  # noqa: E402
from ev_v2.sensitivity import slice_contrasts, slice_shares  # noqa: E402

LABELS = ROOT / "outputs" / "framing_labels.csv"
OUT = ROOT / "outputs"
CARD = ROOT / "data_card"
RAW = ROOT / "data" / "raw"
ANNOT = ROOT / "annotation"
FINDINGS = OUT / "FINDINGS.md"
IDS_URL = "https://arctic-shift.photon-reddit.com/api/posts/ids"
USER_AGENT = "ev-reddit-research/0.2"
EXCERPT_CHARS = 500

BARRIERS = list(PRIMARY_BARRIER.values())
# The bootstrap tables cover every barrier, not only the pre-specified one.
ALL_BARRIERS = [
    "charging access",
    "driving range",
    "vehicle price or total cost",
    "purchase incentives or tax credits",
    "battery reliability",
    "dealer or buying friction",
    "environmental impact",
]


def _load_labels() -> pd.DataFrame:
    frame = pd.read_csv(LABELS)
    frame["id"] = frame["id"].astype(str)
    frame["month"] = frame["month"].astype(str).str.slice(0, 7)
    return frame


def _fetch_ids(ids: list[str], fields: str) -> list[dict]:
    rows: list[dict] = []
    ids = [str(item) for item in ids]
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        url = IDS_URL + "?ids=" + ",".join(chunk) + "&fields=" + fields
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        last_error = "no response"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                time.sleep(2 + attempt * 3)
        else:
            print(f"  ids fetch failed ({last_error})", flush=True)
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.extend(data.values())
        print(f"  ids {start + len(chunk)}/{len(ids)} got {len(rows)}", flush=True)
        time.sleep(0.3)
    return rows


_USER_MENTION = __import__("re").compile(r"/?u/[A-Za-z0-9_-]+")


def _redact(text: str) -> str:
    return _USER_MENTION.sub("u/[redacted]", text or "")


def _excerpt(title: str, selftext: str) -> str:
    text = _redact(submission_text(title, selftext))
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rstrip() + "…"


def _ensure_times(frame: pd.DataFrame, skip_fetch: bool) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "enthusiast_times.csv"
    if path.exists():
        times = pd.read_csv(path)
    elif skip_fetch:
        times = pd.DataFrame(columns=["id", "created_utc"])
    else:
        enth_ids = frame.loc[frame["audience"] == "enthusiast", "id"].tolist()
        print(f"fetching timestamps for {len(enth_ids)} enthusiast posts", flush=True)
        got = _fetch_ids(enth_ids, "id,created_utc")
        times = pd.DataFrame(got)
        if "id" not in times.columns:
            times = pd.DataFrame(columns=["id", "created_utc"])
        times["id"] = times["id"].astype(str)
        times = times.drop_duplicates("id")
        times.to_csv(path, index=False)
        print(f"wrote {path} rows={len(times)}", flush=True)
    times["id"] = times["id"].astype(str)
    merged = frame.merge(times[["id", "created_utc"]], on="id", how="left")
    blocks = []
    fallback = 0
    for audience, subreddit, created in zip(merged["audience"], merged["subreddit"], merged["created_utc"]):
        if audience != "enthusiast":
            blocks.append("")
            continue
        block = assign_week_block(subreddit, created)
        if "|month|" in block or block.endswith("|unknown"):
            fallback += 1
        blocks.append(block)
    merged["week_block"] = blocks
    merged["subreddit_month"] = merged["subreddit"].astype(str) + "|" + merged["month"].astype(str)
    merged.attrs["week_fallback"] = fallback
    return merged


def _text_for(ids: list[str], skip_fetch: bool) -> dict[str, dict]:
    path = RAW / "annotation_text.jsonl"
    have: dict[str, dict] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    have[str(row["id"])] = row
    missing = [doc_id for doc_id in ids if doc_id not in have]
    if missing and not skip_fetch:
        print(f"fetching text for {len(missing)} annotation posts", flush=True)
        got = _fetch_ids(missing, "id,title,selftext")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for row in got:
                doc_id = str(row.get("id", ""))
                if not doc_id:
                    continue
                stored = {
                    "id": doc_id,
                "title": _redact(row.get("title") or ""),
                "text_excerpt": _excerpt(row.get("title") or "", row.get("selftext") or ""),
                }
                have[doc_id] = stored
                handle.write(json.dumps(stored) + "\n")
    return have


def _exclude_preview() -> set[str]:
    path = ANNOT / "annotation_preview.csv"
    if not path.exists():
        return set()
    preview = pd.read_csv(path)
    return set(preview["id"].astype(str))


def _write_annotation(labels: pd.DataFrame, prepared: pd.DataFrame, skip_fetch: bool) -> None:
    draw = draw_annotation_sample(prepared, exclude_ids=_exclude_preview())
    text = _text_for(draw["id"].tolist(), skip_fetch)
    for row in text.values():
        row["title"] = _redact(row.get("title") or "")
        row["text_excerpt"] = _redact(row.get("text_excerpt") or "")
    sheet = blank_sheet(prepared, draw, text)
    key = model_key(prepared, draw)
    manifest = draw.merge(
        prepared[["id", "audience", "subreddit", "month", "year_band"]],
        on="id",
        how="left",
    )
    ANNOT.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(ANNOT / "annotation_sheet.csv", index=False)
    key.to_csv(ANNOT / "annotation_model_key.csv", index=False)
    manifest.to_csv(ANNOT / "sample_manifest.csv", index=False)
    composition = sample_composition(draw, prepared)
    composition.to_csv(OUT / "annotation_sample_summary.csv", index=False)
    meta = pd.DataFrame(
        [
            {
                "n": int(len(draw)),
                "n_enthusiast": int((manifest["audience"] == "enthusiast").sum()),
                "n_shopper": int((manifest["audience"] == "shopper").sum()),
                "n_with_text": int(sheet["text_excerpt"].fillna("").astype(str).str.len().gt(0).sum()),
                "n_barrier_layer": int((draw["layer"] == "barrier").sum()),
                "n_event_layer": int((draw["layer"] == "event").sum()),
                "n_prevalence_layer": int((draw["layer"] == "prevalence").sum()),
                "n_fill_layer": int((draw["layer"] == "fill").sum()),
            }
        ]
    )
    meta.to_csv(OUT / "annotation_sample_meta.csv", index=False)
    print(meta.to_dict(orient="records")[0], flush=True)


def _dense_probe(skip_fetch: bool) -> None:
    """Record why the full event-window enthusiast pull was not labeled."""
    counts = pd.read_csv(CARD / "monthly_post_counts.csv")
    subs = {
        "electricvehicles",
        "teslamotors",
        "BoltEV",
        "Rivian",
        "Ioniq5",
        "KiaEV6",
        "evcharging",
    }

    def in_window(month: str) -> bool:
        return ("2021-08" <= month <= "2023-11") or ("2024-09" <= month <= "2026-03")

    counts["month"] = counts["month"].astype(str).str.slice(0, 7)
    window = counts[counts["subreddit"].isin(subs) & counts["month"].map(in_window)]
    posts = int(pd.to_numeric(window["n_posts"], errors="coerce").fillna(0).sum())
    cells = int(len(window))
    probe = {
        "union_subreddit_months": cells,
        "union_posts": posts,
        "pages_at_limit_100": int((posts + 99) // 100),
    }
    if not skip_fetch:
        import urllib.parse

        url = "https://arctic-shift.photon-reddit.com/api/posts/search?" + urllib.parse.urlencode(
            {
                "subreddit": "evcharging",
                "after": "2021-08-01",
                "before": "2021-09-01",
                "limit": "100",
                "sort": "asc",
                "fields": "id,created_utc",
            }
        )
        started = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            n = len(payload.get("data") or [])
            probe["evcharging_2021_08_posts"] = n
            probe["evcharging_2021_08_seconds"] = round(elapsed, 2)
            probe["evcharging_2021_08_complete"] = n < 100
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            probe["evcharging_2021_08_error"] = str(exc)
    pages = probe["pages_at_limit_100"]
    seconds = probe.get("evcharging_2021_08_seconds", 1.5)
    rate_per_min = 140
    label_hours = posts / rate_per_min / 60
    api_hours = pages * seconds / 3600
    probe["prose"] = (
        f"The full enthusiast window is 12 months before each shock through 6 months after. "
        f"The union of those windows is {cells} subreddit-months and {posts:,} submissions in the monthly census. "
        f"At the search page size of 100, that is about {pages:,} requests. "
        f"A timed request for r/evcharging in August 2021 returned {probe.get('evcharging_2021_08_posts', '—')} posts "
        f"in {probe.get('evcharging_2021_08_seconds', '—')} seconds"
        + (" and was the whole month." if probe.get("evcharging_2021_08_complete") else ".")
        + f" At that pace the transfer is on the order of {api_hours:.1f} hours. "
        f"Labeling {posts:,} posts at the earlier CPU rate of about {rate_per_min} posts a minute is about {label_hours:.0f} hours. "
        "The full window was not downloaded and was not labeled. "
        "The IRA year-ago cell in the week sample is still the 23-post cell in the Framing A table. "
        "What was checked this round is the census total and one complete small month (r/evcharging, August 2021)."
    )
    (CARD / "dense_pull_probe.json").write_text(json.dumps(probe, indent=2))
    print(probe["prose"], flush=True)


def _shopper_2026_note() -> None:
    """Write the feasibility note from the extract, not from a raw line count.

    Submission bodies contain newlines, so counting lines in the CSV overstates
    the number of posts.
    """
    path = RAW / "mainstream_2026.csv"
    kept = None
    labeled = None
    if path.exists():
        frame = pd.read_csv(path, usecols=["subreddit", "month"])
        kept = int(len(frame))
        frame["month"] = frame["month"].astype(str).str.slice(0, 7)
        counts = (
            frame.groupby(["subreddit", "month"], as_index=False)
            .size()
            .rename(columns={"size": "n_posts"})
            .sort_values(["subreddit", "month"])
        )
        counts.to_csv(CARD / "dump_2026_monthly_counts.csv", index=False)
    labels_path = OUT / "framing_labels.csv"
    if labels_path.exists() and kept is not None:
        labels = pd.read_csv(labels_path, usecols=["audience", "month"])
        labels["month"] = labels["month"].astype(str).str.slice(0, 7)
        labeled = int(((labels["audience"] == "shopper") & (labels["month"] >= "2026-01") & (labels["month"] <= "2026-09")).sum())
    its_sentence = ""
    its_path = OUT / "framing_its.csv"
    if its_path.exists():
        its = pd.read_csv(its_path)
        row = its[
            (its["event_id"] == "credit_deadline_2025")
            & (its["audience"] == "shopper")
            & (its["primary"] == True)  # noqa: E712
        ]
        if len(row):
            item = row.iloc[0]
            its_sentence = (
                f" The shopper interrupted series for that deadline now uses {int(item['n_months'])} months "
                f"and {int(item['n_months_post'])} post months. The level shift is {float(item['post']):.3f}, "
                f"interval {float(item['post_lo']):.3f} to {float(item['post_hi']):.3f}."
            )
    prose = (
        "Checked on 19–20 October 2026. The September 2026 torrent lists "
        "`submissions/RS_2026-09.zst` at 22,407,703,827 bytes and `comments/RC_2026-09.zst` at 51,948,958,640 bytes. "
        "The HTTP seed at seed.pullpush.io returned 403. Peers served the files. "
        "January’s submissions file is 19,601,894,565 bytes and arrived at 125 MiB/s; the other eight months are about 18–24 GB and arrived near 100 MiB/s. "
        "Each file was scanned in about 16 minutes (about 41–47 million lines) and deleted before the next download. "
        "Comments were not downloaded. One submissions file at a time fits in the free disk; nine comments files would not. "
        "The word-boundary filter kept "
        + (f"{kept} keyword submissions" if kept is not None else "the keyword submissions")
        + " from r/cars, r/Autos, and r/carbuying for January–September 2026. "
        "The API keyword census for those months is 485. The gap is the same kind of word-boundary difference as the 2015–2025 extract, and the filter was not retuned. "
        + (f"Eight posts were under five tokens, so {labeled} were labeled and appended to the shopper file. " if labeled is not None else "")
        + "September and October 2025, the credit-deadline window, were already in the per-subreddit extract. "
        "Adding 2026 does not change that window’s shares. It extends the months after the window."
        + its_sentence
        + " Month counts are in `dump_2026_monthly_counts.csv`."
    )
    (CARD / "shopper_2026_feasibility.json").write_text(json.dumps({"kept_posts": kept, "labeled_posts": labeled, "prose": prose}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-5 annotation draw, block bootstrap, and sensitivity tables.")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    labels = _load_labels()
    labeled = _ensure_times(labels, args.skip_fetch)
    print(f"week fallback rows {labeled.attrs.get('week_fallback', 0)}", flush=True)
    blocks = labeled.loc[labeled["audience"] == "enthusiast", ["id", "created_utc", "week_block"]]
    blocks.to_csv(OUT / "enthusiast_week_blocks.csv", index=False)
    prepared = prepare_label_frame(labeled)
    _write_annotation(labels, prepared, args.skip_fetch)
    print("block bootstrap", flush=True)
    boot = block_bootstrap_contrasts(labeled, ALL_BARRIERS, n_boot=args.n_boot)
    boot.to_csv(OUT / "framing_block_bootstrap.csv", index=False)
    flags = window_flags(labeled)
    flags.to_csv(OUT / "framing_window_flags.csv", index=False)
    print("ITS bootstrap", flush=True)
    its = bootstrap_interrupted_series(labeled, ALL_BARRIERS, n_boot=args.n_boot)
    its.to_csv(OUT / "framing_its_bootstrap.csv", index=False)
    print("sensitivity", flush=True)
    shares = slice_shares(labeled, ALL_BARRIERS)
    shares.to_csv(OUT / "framing_sensitivity_shares.csv", index=False)
    contrasts = slice_contrasts(labeled, ALL_BARRIERS)
    contrasts.to_csv(OUT / "framing_sensitivity_contrasts.csv", index=False)
    _dense_probe(args.skip_fetch)
    _shopper_2026_note()
    if FINDINGS.exists():
        upsert_framing_findings(FINDINGS, OUT)
        upsert_round5_findings(FINDINGS, OUT)
    card = CARD / "DATA_CARD.md"
    if card.exists():
        upsert_round5_card(card)
    print("round 5 tables written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
