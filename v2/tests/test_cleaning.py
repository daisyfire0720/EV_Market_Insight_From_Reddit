import pandas as pd

from ev_v2.cleaning import prepare_posts


def test_prepare_drops_bots_and_keeps_human_names():
    df = pd.DataFrame(
        {
            "title": [
                "Home charging in an apartment",
                "AutoModerator notice about the megathread",
                "Hi",
                "Talbot bought a Bolt",
            ],
            "selftext": [
                "The level 2 charger needs a 240 volt circuit in the garage.",
                "I am a bot and this action was performed automatically.",
                "ok",
                "It was a long search but the tax credit made the payment work.",
            ],
            "author": ["abbott", "AutoModerator", "someone", "talbot"],
        }
    )
    kept, counts = prepare_posts(df)
    assert counts["removed_bot"] == 1
    assert set(kept["author"]) == {"abbott", "talbot"}
