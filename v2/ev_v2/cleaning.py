"""Light cleaning for Reddit submissions before sentiment and topic models."""

from __future__ import annotations

import re

import pandas as pd

from ev_v2.bots import contains_bot_phrase, is_probable_bot_username

_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"\s+")
_DELETED = {"[deleted]", "[removed]", "deleted", "removed", ""}


def clean_text(text: str | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    blob = str(text)
    blob = blob.replace("\u200b", " ")
    blob = _MD_LINK.sub(r"\1", blob)
    blob = _URL.sub(" ", blob)
    blob = blob.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    blob = _WS.sub(" ", blob).strip()
    return blob


def submission_text(title: str | None, selftext: str | None) -> str:
    title_c = clean_text(title)
    body_c = clean_text(selftext)
    if body_c.lower() in _DELETED:
        body_c = ""
    if title_c and body_c:
        return f"{title_c}\n{body_c}"
    return title_c or body_c


def prepare_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add ``text`` and drop deleted, empty, and probable-bot rows.

    Returns the kept frame and a count of rows removed at each step.
    """
    required = {"title", "selftext", "author"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"posts frame missing columns: {sorted(missing)}")
    counts = {"input_rows": int(len(df))}
    work = df.copy()
    work["text"] = [
        submission_text(t, b) for t, b in zip(work["title"], work["selftext"])
    ]
    work["author_is_bot"] = work["author"].map(is_probable_bot_username)
    work["text_is_bot"] = work["text"].map(contains_bot_phrase)
    work["n_tokens"] = work["text"].str.split().map(len)
    bot_mask = work["author_is_bot"] | work["text_is_bot"]
    empty_mask = work["n_tokens"] < 5
    counts["removed_bot"] = int(bot_mask.sum())
    counts["removed_short_or_empty"] = int((~bot_mask & empty_mask).sum())
    kept = work.loc[~bot_mask & ~empty_mask].copy()
    counts["kept_rows"] = int(len(kept))
    return kept, counts
