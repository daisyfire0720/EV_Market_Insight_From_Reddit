from ev_v2.bots import contains_bot_phrase, is_probable_bot_username


def test_ordinary_names_are_kept():
    for name in ["abbott", "Abbott", "talbot", "autoenthusiast", "bothwell", "sarah"]:
        assert is_probable_bot_username(name) is False


def test_known_and_conventional_bots_are_dropped():
    for name in ["AutoModerator", "remindmebot", "x_bot", "haiku-bot", "CoolDownBot", "auto_moderator"]:
        assert is_probable_bot_username(name) is True


def test_deleted_authors_are_not_classified_as_bots():
    assert is_probable_bot_username("[deleted]") is False
    assert is_probable_bot_username(None) is False


def test_bot_phrase_in_body():
    assert contains_bot_phrase("I am a bot, and this action was performed automatically") is True
    assert contains_bot_phrase("I love my Bolt") is False
