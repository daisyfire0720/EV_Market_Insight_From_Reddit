import pandas as pd

from ev_v2.schema import (
    DEFAULT_LLM_TOPIC_FILE,
    DEFAULT_REFINEMENT_DIR,
    attach_topic_info,
    normalize_topic_info,
)


def test_default_path_matches_refine_output_not_ev_refinement():
    assert DEFAULT_REFINEMENT_DIR == "output/topic_refinement"
    assert DEFAULT_REFINEMENT_DIR != "output/ev_refinement"
    assert DEFAULT_LLM_TOPIC_FILE == "all_subreddits_topic_labels_llm.csv"


def test_llm_columns_map_onto_eda_columns():
    llm = pd.DataFrame(
        {
            "topic_id": [0, -1],
            "doc_count": [12, 4],
            "topic_label_llm": ["Charging at home", "outlier"],
        }
    )
    norm = normalize_topic_info(llm)
    assert list(norm["Topic"]) == [0, -1]
    assert list(norm["Count"]) == [12, 4]


def test_attach_accepts_llm_schema():
    docs = pd.DataFrame({"topic": [0, -1], "text": ["a", "b"]})
    info = pd.DataFrame({"topic_id": [0, -1], "doc_count": [3, 1], "topic_label_llm": ["A", "out"]})
    merged = attach_topic_info(docs, info)
    assert merged.loc[0, "Count"] == 3
    assert merged.loc[0, "topic_label_llm"] == "A"
    assert merged["Topic"].notna().all()
