import numpy as np

from ev_v2.topics import fit_lda


def test_every_document_has_a_topic_probability():
    texts = [
        "home charger installation apartment breaker panel level 2",
        "home charging overnight garage outlet slow",
        "public fast charger broken station queue",
        "dc fast charging road trip plug",
        "tax credit incentive dealer markup price",
        "lease payment apr monthly cost expensive",
        "battery degradation range winter cold",
        "range anxiety highway miles estimate",
        "dealer inventory wait list delivery",
        "bought the car msrp out the door",
        "service visit repair warranty claim",
        "owned it for a year maintenance",
    ] * 4
    result = fit_lda(texts, n_topics=3, random_state=0, min_df=2, max_features=500)
    assert result.doc_topic.shape == (len(texts), 3)
    assert not np.isnan(result.topic_probability_max).any()
    sums = result.doc_topic.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert result.topic_probability_max.min() > 0
    frame = result.document_frame()
    assert frame["topic_probability_max"].notna().all()
    assert set(result.topic_words["topic"]) == {0, 1, 2}
