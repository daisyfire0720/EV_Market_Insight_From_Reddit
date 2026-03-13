from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


@dataclass
class PlotStyle:
    figsize: tuple = (10, 6)
    dpi: int = 300
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    legend_size: int = 9
    table_font_size: int = 9


class MarketFunnelVisualizer:
    """
    Create paper-ready charts and tables for topic-level market funnel analysis.

    Works with outputs from MarketFunnelAnalyzer, but can also operate on plain
    DataFrames if the required columns are present.

    Main use cases:
    1. Funnel-stage distribution charts
    2. Pain-point opportunity charts
    3. Stage x pain-point heatmaps
    4. Sankey charts for stage -> pain point or any other categorical flow
    5. Topic summary / stage summary tables exported as PNG and CSV
    6. Topic-over-time and stage-over-time charts from doc-level data
    """

    def __init__(self, output_dir: str = "market_funnel_figures", style: Optional[PlotStyle] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style or PlotStyle()
        self._set_base_style()

    def _set_base_style(self) -> None:
        plt.rcParams.update(
            {
                "figure.dpi": self.style.dpi,
                "savefig.dpi": self.style.dpi,
                "axes.titlesize": self.style.title_size,
                "axes.labelsize": self.style.label_size,
                "xtick.labelsize": self.style.tick_size,
                "ytick.labelsize": self.style.tick_size,
                "legend.fontsize": self.style.legend_size,
                "figure.titlesize": self.style.title_size,
                "font.size": self.style.label_size,
            }
        )

    @staticmethod
    def _save(fig: plt.Figure, path_base: Path) -> List[str]:
        png_path = str(path_base.with_suffix(".png"))
        pdf_path = str(path_base.with_suffix(".pdf"))
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        return [png_path, pdf_path]

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    @staticmethod
    def _wrap_text(text: str, width: int = 28) -> str:
        words = str(text).split()
        if not words:
            return ""
        lines = []
        cur = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + len(cur) > width:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += len(w)
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

    def plot_funnel_stage_distribution(
        self,
        stage_summary_df: pd.DataFrame,
        weight_col: str = "weighted_topics",
        stage_col: str = "funnel_stage",
        title: str = "Market Funnel Stage Distribution",
        filename: str = "funnel_stage_distribution",
        annotate_share: bool = True,
    ) -> List[str]:
        self._validate_columns(stage_summary_df, [stage_col, weight_col], "stage_summary_df")
        df = stage_summary_df.copy().sort_values(weight_col, ascending=True)

        fig, ax = plt.subplots(figsize=self.style.figsize)
        bars = ax.barh(df[stage_col].astype(str), df[weight_col].astype(float))
        ax.set_title(title)
        ax.set_xlabel(weight_col.replace("_", " ").title())
        ax.set_ylabel("Funnel Stage")

        if annotate_share and "weighted_share" in df.columns:
            for bar, share in zip(bars, df["weighted_share"].fillna(0.0)):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f" {share:.1%}",
                    va="center",
                )

        return self._save(fig, self.output_dir / filename)

    def plot_stage_confidence(
        self,
        stage_summary_df: pd.DataFrame,
        title: str = "Average Funnel Assignment Confidence by Stage",
        filename: str = "stage_confidence",
    ) -> List[str]:
        self._validate_columns(stage_summary_df, ["funnel_stage", "avg_confidence"], "stage_summary_df")
        df = stage_summary_df.copy().sort_values("avg_confidence", ascending=False)

        fig, ax = plt.subplots(figsize=self.style.figsize)
        bars = ax.bar(df["funnel_stage"].astype(str), df["avg_confidence"].astype(float))
        ax.set_title(title)
        ax.set_ylabel("Average Confidence")
        ax.set_xlabel("Funnel Stage")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)

        for bar, val in zip(bars, df["avg_confidence"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center")

        return self._save(fig, self.output_dir / filename)

    def plot_pain_point_opportunity(
        self,
        pain_point_summary_df: pd.DataFrame,
        top_n: int = 12,
        title: str = "Top Pain-Point Opportunities",
        filename: str = "pain_point_opportunity",
    ) -> List[str]:
        self._validate_columns(
            pain_point_summary_df,
            ["pain_point_label", "opportunity_score"],
            "pain_point_summary_df",
        )
        df = pain_point_summary_df.copy().sort_values("opportunity_score", ascending=False).head(top_n)
        df = df.iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(df) + 2)))
        bars = ax.barh(df["pain_point_label"].astype(str), df["opportunity_score"].astype(float))
        ax.set_title(title)
        ax.set_xlabel("Opportunity Score")
        ax.set_ylabel("Pain Point")

        if "weighted_topics" in df.columns:
            for bar, wt in zip(bars, df["weighted_topics"]):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {wt:.1f}", va="center")

        return self._save(fig, self.output_dir / filename)

    def plot_stage_pain_point_heatmap(
        self,
        pain_point_table_df: pd.DataFrame,
        stage_col: str = "funnel_stage",
        pain_col: str = "pain_point_label",
        value_col: str = "topic_weight",
        top_n_pain_points: int = 10,
        title: str = "Stage × Pain-Point Heatmap",
        filename: str = "stage_pain_point_heatmap",
    ) -> List[str]:
        self._validate_columns(pain_point_table_df, [stage_col, pain_col, value_col], "pain_point_table_df")
        df = pain_point_table_df.copy()

        top_pains = (
            df.groupby(pain_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_pain_points)
            .index.tolist()
        )
        df = df[df[pain_col].isin(top_pains)]

        pivot = pd.pivot_table(
            df,
            index=stage_col,
            columns=pain_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(8, 0.9 * pivot.shape[1] + 3), max(4.5, 0.75 * pivot.shape[0] + 2)))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_xticklabels([self._wrap_text(x, 18) for x in pivot.columns], rotation=30, ha="right")
        ax.set_yticklabels(pivot.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(value_col.replace("_", " ").title())

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center")

        return self._save(fig, self.output_dir / filename)

    def export_table_as_figure(
        self,
        df: pd.DataFrame,
        filename: str,
        title: Optional[str] = None,
        max_rows: int = 12,
        columns: Optional[Sequence[str]] = None,
        round_cols: Optional[Iterable[str]] = None,
    ) -> List[str]:
        table_df = df.copy()
        if columns is not None:
            table_df = table_df.loc[:, list(columns)]
        table_df = table_df.head(max_rows).copy()

        if round_cols is not None:
            for col in round_cols:
                if col in table_df.columns:
                    table_df[col] = pd.to_numeric(table_df[col], errors="ignore")
                    if pd.api.types.is_numeric_dtype(table_df[col]):
                        table_df[col] = table_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        for col in table_df.columns:
            table_df[col] = table_df[col].astype(str).map(lambda x: self._wrap_text(x, 24))

        n_rows, n_cols = table_df.shape
        fig_h = max(2.8, 0.5 * n_rows + 1.2)
        fig_w = max(8, 1.8 * n_cols)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        if title:
            ax.set_title(title, pad=12)

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.style.table_font_size)
        table.scale(1, 1.35)

        csv_path = str((self.output_dir / f"{filename}.csv"))
        df.to_csv(csv_path, index=False)
        saved = self._save(fig, self.output_dir / filename)
        saved.append(csv_path)
        return saved

    def build_topic_time_series(
        self,
        doc_df: pd.DataFrame,
        date_col: str,
        topic_col: str,
        topic_weight_col: Optional[str] = None,
        freq: str = "M",
        normalize_to_share: bool = False,
    ) -> pd.DataFrame:
        self._validate_columns(doc_df, [date_col, topic_col], "doc_df")
        df = doc_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, topic_col]).copy()
        df["period"] = df[date_col].dt.to_period(freq).dt.to_timestamp()
        df["_weight"] = df[topic_weight_col].astype(float) if topic_weight_col else 1.0

        ts = (
            df.groupby(["period", topic_col], as_index=False)["_weight"]
            .sum()
            .rename(columns={"_weight": "value", topic_col: "topic_label"})
        )

        if normalize_to_share:
            totals = ts.groupby("period")["value"].transform("sum")
            ts["value"] = np.where(totals > 0, ts["value"] / totals, 0.0)
            ts["metric"] = "share"
        else:
            ts["metric"] = "count"

        return ts.sort_values(["topic_label", "period"]).reset_index(drop=True)

    def plot_topic_over_time(
        self,
        time_series_df: pd.DataFrame,
        top_n_topics: int = 8,
        topic_col: str = "topic_label",
        date_col: str = "period",
        value_col: str = "value",
        title: str = "Topic Trends Over Time",
        filename: str = "topic_over_time",
    ) -> List[str]:
        self._validate_columns(time_series_df, [topic_col, date_col, value_col], "time_series_df")
        df = time_series_df.copy()
        top_topics = (
            df.groupby(topic_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_topics)
            .index.tolist()
        )
        df = df[df[topic_col].isin(top_topics)].copy()
        pivot = df.pivot(index=date_col, columns=topic_col, values=value_col).fillna(0)
        pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", linewidth=1.8, label=str(col))

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(value_col.replace("_", " ").title())
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        return self._save(fig, self.output_dir / filename)

    def build_stage_time_series(
        self,
        doc_df: pd.DataFrame,
        date_col: str,
        topic_col: str,
        topic_stage_map_df: pd.DataFrame,
        stage_col: str = "funnel_stage",
        topic_label_col_in_map: str = "topic_label_llm",
        topic_weight_col: Optional[str] = None,
        freq: str = "M",
        normalize_to_share: bool = False,
    ) -> pd.DataFrame:
        self._validate_columns(doc_df, [date_col, topic_col], "doc_df")
        self._validate_columns(topic_stage_map_df, [topic_label_col_in_map, stage_col], "topic_stage_map_df")

        merged = doc_df.merge(
            topic_stage_map_df[[topic_label_col_in_map, stage_col]].drop_duplicates(),
            left_on=topic_col,
            right_on=topic_label_col_in_map,
            how="left",
        )
        merged[stage_col] = merged[stage_col].fillna("Unmapped")

        ts = self.build_topic_time_series(
            merged,
            date_col=date_col,
            topic_col=stage_col,
            topic_weight_col=topic_weight_col,
            freq=freq,
            normalize_to_share=normalize_to_share,
        )
        return ts.rename(columns={"topic_label": "funnel_stage"})

    def plot_stage_over_time(
        self,
        stage_time_series_df: pd.DataFrame,
        title: str = "Funnel Stage Trends Over Time",
        filename: str = "stage_over_time",
    ) -> List[str]:
        self._validate_columns(stage_time_series_df, ["funnel_stage", "period", "value"], "stage_time_series_df")
        pivot = stage_time_series_df.pivot(index="period", columns="funnel_stage", values="value").fillna(0)
        pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", linewidth=2.0, label=str(col))

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        return self._save(fig, self.output_dir / filename)

    def build_sankey_flow_table(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        value_col: str = "topic_weight",
        min_value: float = 0.0,
        top_n_links: Optional[int] = None,
        source_order: Optional[Sequence[str]] = None,
        target_order: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self._validate_columns(df, [source_col, target_col, value_col], "df")
        flow = (
            df.groupby([source_col, target_col], as_index=False)[value_col]
            .sum()
            .rename(columns={source_col: "source", target_col: "target", value_col: "value"})
        )
        flow = flow[flow["value"] >= float(min_value)].copy()

        if source_order is not None:
            flow["_source_rank"] = flow["source"].map({v: i for i, v in enumerate(source_order)}).fillna(9999)
        else:
            flow["_source_rank"] = 9999

        if target_order is not None:
            flow["_target_rank"] = flow["target"].map({v: i for i, v in enumerate(target_order)}).fillna(9999)
        else:
            flow["_target_rank"] = 9999

        flow = flow.sort_values(["_source_rank", "source", "value", "_target_rank", "target"], ascending=[True, True, False, True, True])
        if top_n_links is not None:
            flow = flow.head(top_n_links).copy()

        return flow.drop(columns=["_source_rank", "_target_rank"]).reset_index(drop=True)

    @staticmethod
    def _default_plotly_node_colors(n_nodes: int) -> List[str]:
        palette = [
            "rgba(31,119,180,0.80)", "rgba(255,127,14,0.80)", "rgba(44,160,44,0.80)",
            "rgba(214,39,40,0.80)", "rgba(148,103,189,0.80)", "rgba(140,86,75,0.80)",
            "rgba(227,119,194,0.80)", "rgba(127,127,127,0.80)", "rgba(188,189,34,0.80)",
            "rgba(23,190,207,0.80)",
        ]
        return [palette[i % len(palette)] for i in range(n_nodes)]

    def plot_sankey(
        self,
        flow_df: pd.DataFrame,
        title: str = "Market Funnel Sankey",
        filename: str = "market_funnel_sankey",
        source_col: str = "source",
        target_col: str = "target",
        value_col: str = "value",
        node_order: Optional[Sequence[str]] = None,
        width: int = 1200,
        height: int = 700,
    ) -> List[str]:
        if go is None:
            raise ImportError("plotly is required for Sankey charts. Please install plotly.")

        self._validate_columns(flow_df, [source_col, target_col, value_col], "flow_df")
        df = flow_df.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[source_col, target_col, value_col])
        if df.empty:
            raise ValueError("flow_df is empty after cleaning; nothing to plot.")

        if node_order is None:
            left_nodes = df[source_col].astype(str).drop_duplicates().tolist()
            right_nodes = [x for x in df[target_col].astype(str).drop_duplicates().tolist() if x not in left_nodes]
            node_labels = left_nodes + right_nodes
        else:
            node_labels = [str(x) for x in node_order]
            missing = set(df[source_col].astype(str)).union(set(df[target_col].astype(str))) - set(node_labels)
            if missing:
                node_labels += sorted(missing)

        node_to_idx = {node: i for i, node in enumerate(node_labels)}

        link_source = df[source_col].astype(str).map(node_to_idx).tolist()
        link_target = df[target_col].astype(str).map(node_to_idx).tolist()
        link_value = df[value_col].astype(float).tolist()
        link_labels = [
            f"{src} → {tgt}: {val:.1f}"
            for src, tgt, val in zip(df[source_col].astype(str), df[target_col].astype(str), link_value)
        ]

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=18,
                        thickness=18,
                        line=dict(color="rgba(80,80,80,0.5)", width=0.5),
                        label=node_labels,
                        color=self._default_plotly_node_colors(len(node_labels)),
                    ),
                    link=dict(
                        source=link_source,
                        target=link_target,
                        value=link_value,
                        label=link_labels,
                    ),
                )
            ]
        )
        fig.update_layout(title_text=title, font_size=12, width=width, height=height)

        saved: List[str] = []
        html_path = str((self.output_dir / f"{filename}.html"))
        fig.write_html(html_path)
        saved.append(html_path)

        # Try to export static paper-ready outputs if kaleido is available.
        for ext in ["png", "pdf"]:
            out_path = str((self.output_dir / f"{filename}.{ext}"))
            try:
                fig.write_image(out_path)
                saved.append(out_path)
            except Exception:
                pass

        flow_csv = str((self.output_dir / f"{filename}_flows.csv"))
        df.to_csv(flow_csv, index=False)
        saved.append(flow_csv)
        return saved

    def plot_stage_pain_point_sankey(
        self,
        pain_point_table_df: pd.DataFrame,
        stage_col: str = "funnel_stage",
        pain_col: str = "pain_point_label",
        value_col: str = "topic_weight",
        min_value: float = 0.0,
        top_n_links: Optional[int] = 20,
        title: str = "Funnel Stage → Pain-Point Sankey",
        filename: str = "stage_pain_point_sankey",
    ) -> Tuple[pd.DataFrame, List[str]]:
        self._validate_columns(pain_point_table_df, [stage_col, pain_col, value_col], "pain_point_table_df")
        stage_order = [
            "Awareness",
            "Consideration",
            "Evaluation",
            "Purchase",
            "Ownership",
            "Advocacy",
            "Unmapped",
        ]
        flow_df = self.build_sankey_flow_table(
            pain_point_table_df,
            source_col=stage_col,
            target_col=pain_col,
            value_col=value_col,
            min_value=min_value,
            top_n_links=top_n_links,
            source_order=stage_order,
        )
        saved = self.plot_sankey(
            flow_df,
            title=title,
            filename=filename,
        )
        return flow_df, saved

    def plot_topic_stage_sankey(
        self,
        topic_stage_df: pd.DataFrame,
        topic_col: str = "topic_label_llm",
        stage_col: str = "funnel_stage",
        value_col: str = "topic_weight",
        min_value: float = 0.0,
        top_n_links: Optional[int] = 25,
        title: str = "Topic → Funnel Stage Sankey",
        filename: str = "topic_stage_sankey",
    ) -> Tuple[pd.DataFrame, List[str]]:
        self._validate_columns(topic_stage_df, [topic_col, stage_col, value_col], "topic_stage_df")
        stage_order = [
            "Awareness",
            "Consideration",
            "Evaluation",
            "Purchase",
            "Ownership",
            "Advocacy",
            "Unmapped",
        ]
        flow_df = self.build_sankey_flow_table(
            topic_stage_df,
            source_col=topic_col,
            target_col=stage_col,
            value_col=value_col,
            min_value=min_value,
            top_n_links=top_n_links,
            target_order=stage_order,
        )
        saved = self.plot_sankey(
            flow_df,
            title=title,
            filename=filename,
        )
        return flow_df, saved

    def create_full_figure_pack(
        self,
        stage_summary_df: Optional[pd.DataFrame] = None,
        pain_point_summary_df: Optional[pd.DataFrame] = None,
        pain_point_table_df: Optional[pd.DataFrame] = None,
        topic_deep_dive_df: Optional[pd.DataFrame] = None,
        include_sankey: bool = True,
    ) -> dict:
        outputs = {}

        if stage_summary_df is not None:
            outputs["funnel_stage_distribution"] = self.plot_funnel_stage_distribution(stage_summary_df)
            outputs["stage_confidence"] = self.plot_stage_confidence(stage_summary_df)
            outputs["stage_summary_table"] = self.export_table_as_figure(
                stage_summary_df,
                filename="stage_summary_table",
                title="Stage Summary Table",
                max_rows=12,
                round_cols=["avg_confidence", "avg_match_score", "weighted_share"],
            )

        if pain_point_summary_df is not None:
            outputs["pain_point_opportunity"] = self.plot_pain_point_opportunity(pain_point_summary_df)
            outputs["pain_point_summary_table"] = self.export_table_as_figure(
                pain_point_summary_df,
                filename="pain_point_summary_table",
                title="Pain-Point Summary Table",
                max_rows=15,
                round_cols=["opportunity_score", "weighted_topics"],
            )

        if pain_point_table_df is not None:
            outputs["stage_pain_point_heatmap"] = self.plot_stage_pain_point_heatmap(pain_point_table_df)
            if include_sankey:
                flow_df, saved = self.plot_stage_pain_point_sankey(pain_point_table_df)
                outputs["stage_pain_point_sankey"] = saved
                outputs["stage_pain_point_sankey_flows"] = flow_df

        if topic_deep_dive_df is not None:
            keep_cols = [
                c for c in [
                    "topic_id",
                    "topic_label_llm",
                    "funnel_stage",
                    "stage_confidence",
                    "customer_need",
                    "purchase_barrier",
                    "suggested_action",
                ] if c in topic_deep_dive_df.columns
            ]
            outputs["topic_deep_dive_table"] = self.export_table_as_figure(
                topic_deep_dive_df,
                filename="topic_deep_dive_table",
                title="Topic Deep-Dive Table",
                max_rows=12,
                columns=keep_cols,
                round_cols=["stage_confidence"],
            )
            if include_sankey:
                topic_col_candidates = ["topic_label_llm", "topic_label", "topic_name"]
                value_col_candidates = ["topic_weight", "Count", "doc_count", "topic_count"]
                topic_col = next((c for c in topic_col_candidates if c in topic_deep_dive_df.columns), None)
                value_col = next((c for c in value_col_candidates if c in topic_deep_dive_df.columns), None)
                if topic_col is not None and value_col is not None and "funnel_stage" in topic_deep_dive_df.columns:
                    flow_df, saved = self.plot_topic_stage_sankey(
                        topic_deep_dive_df,
                        topic_col=topic_col,
                        stage_col="funnel_stage",
                        value_col=value_col,
                    )
                    outputs["topic_stage_sankey"] = saved
                    outputs["topic_stage_sankey_flows"] = flow_df

        return outputs


if __name__ == "__main__":
    # Example usage template
    # stage_summary = pd.read_csv("stage_summary.csv")
    # pain_summary = pd.read_csv("pain_point_summary.csv")
    # pain_table = pd.read_csv("pain_point_table.csv")
    # topic_deep_dive = pd.read_csv("topic_deep_dive.csv")
    # viz = MarketFunnelVisualizer(output_dir="market_funnel_figures")
    # outputs = viz.create_full_figure_pack(
    #     stage_summary_df=stage_summary,
    #     pain_point_summary_df=pain_summary,
    #     pain_point_table_df=pain_table,
    #     topic_deep_dive_df=topic_deep_dive,
    #     include_sankey=True,
    # )
    # print(outputs)
    #
    # Or explicitly:
    # flow_df, sankey_files = viz.plot_stage_pain_point_sankey(
    #     pain_table,
    #     min_value=2,
    #     top_n_links=18,
    # )
    # print(flow_df.head())
    # print(sankey_files)
    pass
