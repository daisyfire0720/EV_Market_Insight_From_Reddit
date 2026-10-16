"""Bot-username filter that does not swallow ordinary names.

The v1 regex ``(?:auto\\w*|\\w*bot\\w*)`` drops any username that contains
``bot`` or starts with ``auto`` (``abbott``, ``talbot``, ``autoenthusiast``).
This filter keeps those names and drops accounts that follow Reddit's bot
naming conventions or a short known-bot list.
"""

from __future__ import annotations

import re

# Usernames that are bots even when they do not use a ``_bot`` token.
KNOWN_BOT_USERNAMES = frozenset(
    {
        "automoderator",
        "auto_moderator",
        "autotldr",
        "autowikibot",
        "remindmebot",
        "video_descriptionbot",
        "generic_reddit_bot",
        "reddit-user-analyzer",
    }
)

# Human names the v1 regex falsely removed. Checked before the suffix rule.
HUMAN_NAME_EXCEPTIONS = frozenset(
    {
        "abbott",
        "abbot",
        "talbot",
        "autoenthusiast",
        "autolycus",
        "botham",
        "sabot",
        "saboteur",
        "bothwell",
    }
)

# ``bot`` as its own token, delimited by the start/end of the name or by
# ``_``, ``-``, or a digit. Does not match ``bot`` inside ``abbott``.
_BOT_TOKEN = re.compile(r"(?:^|[_\-\d])bot(?:$|[_\-\d])", flags=re.IGNORECASE)

# ``remindmebot``-style suffix. Length >= 8 avoids short surnames such as
# ``talbot`` (6). A separator also qualifies (``haiku_bot``).
_BOT_SUFFIX = re.compile(r"bot\d*$", flags=re.IGNORECASE)

BOT_PHRASES = (
    "i am a bot",
    "i'm a bot",
    "this action was performed automatically",
    "please contact the moderators",
)


def is_probable_bot_username(author: str | None) -> bool:
    if author is None:
        return False
    name = str(author).strip().lower()
    if not name or name in {"[deleted]", "[removed]"}:
        return False
    if name in HUMAN_NAME_EXCEPTIONS:
        return False
    if name in KNOWN_BOT_USERNAMES:
        return True
    if _BOT_TOKEN.search(name):
        return True
    if _BOT_SUFFIX.search(name) and ("_" in name or "-" in name or len(name) >= 8):
        # ``talbot`` ends in bot but is shorter than 8 and has no separator.
        # Guard the exception list again in case a longer surname is added later.
        if name not in HUMAN_NAME_EXCEPTIONS:
            # ``robot`` (5) is not dropped by the length rule. ``cooldownbot`` is.
            if len(name) >= 8 or "_" in name or "-" in name:
                return True
    if name.startswith("auto") and "mod" in name:
        return True
    return False


def contains_bot_phrase(text: str | None) -> bool:
    if not text:
        return False
    blob = str(text).lower()
    return any(phrase in blob for phrase in BOT_PHRASES)
