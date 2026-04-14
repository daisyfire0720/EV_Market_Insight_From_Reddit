from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_funnel_analyzer_aligned import MarketFunnelAnalyzer
from market_funnel_visualizer_aligned import MarketFunnelVisualizer


OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topic_id": 1,
                "topic_label_llm": "Should I buy an EV for commuting?",
                "topic_summary_llm": "I am new to EVs and wondering if an EV is worth considering for daily commuting.",
                "Count": 10,
            },
            {
                "topic_id": 2,
                "topic_label_llm": "Model Y vs Ioniq 5 for family car",
                "topic_summary_llm": "Comparing shortlisted EV SUVs for cargo, comfort, and commute.",
                "Count": 9,
            },
            {
                "topic_id": 3,
                "topic_label_llm": "Can I make an EV work if I live in an apartment and street park?",
                "topic_summary_llm": "Charging feasibility, tax credit, and insurance questions before buying.",
                "Count": 12,
            },
            {
                "topic_id": 4,
                "topic_label_llm": "Dealer added markup and delivery keeps slipping",
                "topic_summary_llm": "Dealership markup, inventory, and delivery timing during purchase.",
                "Count": 7,
            },
            {
                "topic_id": 5,
                "topic_label_llm": "I just bought a Bolt - how do I set up charging and the app?",
                "topic_summary_llm": "New owner with first charging setup and app confusion questions.",
                "Count": 6,
            },
            {
                "topic_id": 6,
                "topic_label_llm": "After two winters my EV range seems worse",
                "topic_summary_llm": "Battery degradation and service concerns after owning the EV for years.",
                "Count": 5,
            },
        ]
    )


def main() -> None:
    df = build_sample_df()
    analyzer = MarketFunnelAnalyzer(topic_weight_col="Count")
    outputs = analyzer.run_full_analysis(df=df)

    print("\n[1] Stage mapping preview")
    print(outputs["topic_stage_mapping"][["topic_id", "topic_label_llm", "funnel_stage", "topic_family", "stage_confidence"]])

    expected_stage_map = {
        1: "Awareness / Need Formation",
        2: "Consideration / Shortlisting",
        3: "Evaluation / Practical Fit",
        4: "Purchase / Transaction",
        5: "Onboarding / Early Ownership",
        6: "Long-Term Ownership / Retention",
    }

    got = outputs["topic_stage_mapping"].set_index("topic_id")["funnel_stage"].to_dict()
    for topic_id, expected in expected_stage_map.items():
        actual = got.get(topic_id)
        if actual != expected:
            raise AssertionError(f"topic_id={topic_id}: expected {expected}, got {actual}")

    xlsx_path = OUTPUT_DIR / "market_funnel_analysis_test.xlsx"
    analyzer.export_results(str(xlsx_path))
    print(f"\n[2] Excel exported to: {xlsx_path.resolve()}")

    viz = MarketFunnelVisualizer(output_dir=str(OUTPUT_DIR / "figures"))
    fig1 = viz.plot_funnel_stage_distribution(outputs["stage_summary"])
    fig2 = viz.plot_stage_confidence(outputs["stage_summary"])
    fig3 = viz.plot_stage_family_heatmap(outputs["topic_stage_mapping"])

    print("\n[3] Generated figure files")
    print(fig1)
    print(fig2)
    print(fig3)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
