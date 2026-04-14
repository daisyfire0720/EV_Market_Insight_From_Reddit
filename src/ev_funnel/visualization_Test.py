from market_funnel_visualizer_aligned import MarketFunnelVisualizer

viz = MarketFunnelVisualizer(output_dir="test_figures")
viz.plot_funnel_stage_distribution(outputs["stage_summary"])
viz.plot_stage_confidence(outputs["stage_summary"])
viz.plot_stage_family_heatmap(outputs["topic_stage_mapping"])