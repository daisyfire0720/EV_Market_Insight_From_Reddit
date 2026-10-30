"""v2 measurement pipeline for EV Reddit posts.

The original packages under ``src/`` are left unchanged. This package
reimplements the pieces that the March 2026 review found were broken:
word-boundary keyword matching, topic-table column names, default paths,
the bot-username filter, and per-document topic probabilities.
"""

__version__ = "0.2.0"
