"""VADER sentiment, the dictionary baseline used by Ruan and Lv."""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_texts(texts: list[str]) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = [analyzer.polarity_scores(t if isinstance(t, str) else "") for t in texts]
    out = pd.DataFrame(rows)
    out = out.rename(
        columns={
            "neg": "vader_neg",
            "neu": "vader_neu",
            "pos": "vader_pos",
            "compound": "vader_compound",
        }
    )
    return out
