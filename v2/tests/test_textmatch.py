from ev_v2.textmatch import contains_keyword, find_keywords


def test_ev_does_not_match_inside_longer_words():
    assert contains_keyword("every level revs", "ev") is False
    assert contains_keyword("I bought an EV", "ev") is True
    assert contains_keyword("EVs are common", "ev") is False
    assert contains_keyword("the ev.", "ev") is True


def test_vs_does_not_match_inside_evs():
    assert contains_keyword("evs are cheap", "vs") is False
    assert contains_keyword("model 3 vs model y", "vs") is True


def test_multiword_phrase_uses_boundaries():
    assert contains_keyword("the tax credit ends", "tax credit") is True
    assert contains_keyword("tax credits", "tax credit") is False
    hits = find_keywords("Is the tax credit worth it vs a hybrid?", ["tax credit", "vs", "ev"])
    assert hits == ["tax credit", "vs"]
