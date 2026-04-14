from market_funnel_analyzer_aligned import MarketFunnelAnalyzer

analyzer = MarketFunnelAnalyzer(topic_weight_col="Count")
outputs = analyzer.run_full_analysis(csv_path="test_market_funnel_topics.csv")

print(outputs["topic_stage_mapping"][[
    "topic_id",
    "topic_label_llm",
    "funnel_stage",
    "topic_family",
    "stage_confidence",
    "topic_family_confidence"
]])

analyzer.export_results("test_market_funnel_outputs.xlsx")

from market_funnel_visualizer_aligned import MarketFunnelVisualizer

viz = MarketFunnelVisualizer(output_dir="test_figures")
viz.plot_funnel_stage_distribution(outputs["stage_summary"])
viz.plot_stage_confidence(outputs["stage_summary"])
viz.plot_stage_family_heatmap(outputs["topic_stage_mapping"])