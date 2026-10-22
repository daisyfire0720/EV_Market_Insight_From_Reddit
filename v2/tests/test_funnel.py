from ev_v2.funnel import assign_stage


def test_substring_false_friend_does_not_survive_word_boundaries():
    text = "every revs the engine and evs are everywhere"
    sub_stage, sub_hits = assign_stage(text, mode="substring")
    bound_stage, bound_hits = assign_stage(text, mode="boundary")
    assert "ev" in sub_hits or "vs" in sub_hits
    assert sub_stage == "Consideration"
    assert "ev" not in bound_hits and "vs" not in bound_hits
    assert bound_stage == "Unclear"


def test_real_comparison_still_lands_in_consideration():
    stage, hits = assign_stage("model 3 vs model y reliability", mode="boundary")
    assert stage == "Consideration"
    assert "vs" in hits and "reliability" in hits
