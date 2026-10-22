from ev_v2.dealer import decide_dealer, friction_lexicon


def test_markup_scored_as_price_is_still_dealer_friction():
    text = "The dealer added a $5,000 market adjustment over MSRP."
    # The CPU model scored a sentence like this at about 0.13 dealer / 0.90 price.
    decision = decide_dealer(text, dealer_score=0.126, threshold=0.5)
    assert decision.on is True
    assert decision.reason == "lexicon"
    assert friction_lexicon(text)


def test_plain_price_service_and_charging_location_stay_off():
    assert decide_dealer("The car is too expensive at $45,000.", 0.02).on is False
    service = decide_dealer("My dealer did the 10,000-mile service and rotated the tires.", 0.89)
    assert service.on is False
    assert service.reason == "suppressed_service_or_charging_location"
    assert decide_dealer("My dealer did the 10,000-mile service and rotated the tires.", 0.05).reason == "off"
    charging = decide_dealer("Charging at the dealer was free while I waited for the shuttle.", 0.54)
    assert charging.on is False
    assert charging.reason == "suppressed_service_or_charging_location"
    assert decide_dealer("The dealer quoted me $42,000, which seems high for a used one.", 0.32).on is False
    # A markup mentioned next to service is still dealer friction.
    assert decide_dealer("The dealer service desk added a dealer markup.", 0.10).on is True


def test_high_nli_score_turns_dealer_on_without_a_keyword():
    text = "The finance manager pressured me to buy nitrogen and VIN etch."
    assert friction_lexicon(text) is False
    decision = decide_dealer(text, dealer_score=0.80)
    assert decision.on is True
    assert decision.reason == "nli"


def test_inventory_wait_needs_a_dealer_mention():
    assert friction_lexicon("Waiting six months because the dealer has no inventory.")
    assert friction_lexicon("The dealership put me on a wait list with no delivery date.")
    assert not friction_lexicon("No inventory at the grocery store.")


def test_word_boundary_does_not_match_inside_another_token():
    assert not friction_lexicon("The marked-upholstery color is dealer-only.")
    assert friction_lexicon("They added a dealer markup overnight.")
