from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


class MarketFunnelVisualizer:
    """Visualizer aligned to the 6-stage EV funnel analyzer."""

    STAGE_ORDER = [
        "Awareness / Need Formation",
        "Consideration / Shortlisting",
        "Evaluation / Practical Fit",
        "Purchase / Transaction",
        "Onboarding / Early Ownership",
        "Long-Term Ownership / Retention",
        "Unclear / Cross-Stage",
        "Unmapped",
    ]

    def __init__(self, output_dir: str = "market_funnel_figures", dpi: int = 300) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 11,
        })

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
    def _wrap_text(text: str, width: int = 24) -> str:
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

    def _sort_stage_df(self, df: pd.DataFrame, stage_col: str = "funnel_stage") -> pd.DataFrame:
        out = df.copy()
        out["_stage_order"] = out[stage_col].map({v: i for i, v in enumerate(self.STAGE_ORDER)}).fillna(999)
        return out.sort_values(["_stage_order", stage_col]).drop(columns="_stage_order")

    def plot_funnel_stage_distribution(
        self,
        stage_summary_df: pd.DataFrame,
        weight_col: str = "weighted_topics",
        stage_col: str = "funnel_stage",
        title: str = "EV Market Funnel Stage Distribution",
        filename: str = "funnel_stage_distribution",
    ) -> List[str]:
        self._validate_columns(stage_summary_df, [stage_col, weight_col], "stage_summary_df")
        df = self._sort_stage_df(stage_summary_df, stage_col=stage_col)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.barh(df[stage_col].astype(str), df[weight_col].astype(float))
        ax.set_title(title)
        ax.set_xlabel(weight_col.replace("_", " ").title())
        ax.set_ylabel("Funnel Stage")
        ax.invert_yaxis()

        if "weighted_share" in df.columns:
            for bar, share in zip(bars, df["weighted_share"].fillna(0.0)):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {share:.1%}", va="center")

        return self._save(fig, self.output_dir / filename)

    def plot_stage_confidence(
        self,
        stage_summary_df: pd.DataFrame,
        title: str = "Average Funnel Assignment Confidence by Stage",
        filename: str = "stage_confidence",
    ) -> List[str]:
        self._validate_columns(stage_summary_df, ["funnel_stage", "avg_confidence"], "stage_summary_df")
        df = self._sort_stage_df(stage_summary_df)

        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        bars = ax.bar(df["funnel_stage"].astype(str), df["avg_confidence"].astype(float))
        ax.set_title(title)
        ax.set_ylabel("Average Confidence")
        ax.set_xlabel("Funnel Stage")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=28)

        for bar, val in zip(bars, df["avg_confidence"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.2f}", ha="center")

        return self._save(fig, self.output_dir / filename)

    def plot_stage_family_heatmap(
        self,
        topic_stage_df: pd.DataFrame,
        stage_col: str = "funnel_stage",
        family_col: str = "topic_family",
        value_col: str = "topic_weight",
        title: str = "Stage × Topic-Family Heatmap",
        filename: str = "stage_family_heatmap",
    ) -> List[str]:
        self._validate_columns(topic_stage_df, [stage_col, family_col, value_col], "topic_stage_df")
        df = topic_stage_df.copy()
        pivot = pd.pivot_table(df, index=stage_col, columns=family_col, values=value_col, aggfunc="sum", fill_value=0)
        pivot = pivot.reindex([s for s in self.STAGE_ORDER if s in pivot.index])
        pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(10, 0.9 * pivot.shape[1] + 3), max(4.5, 0.7 * pivot.shape[0] + 2)))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_xticklabels([self._wrap_text(x, 18) for x in pivot.columns], rotation=28, ha="right")
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
            table_df[col] = table_df[col].astype(str).map(lambda x: self._wrap_text(x, 22))

        n_rows, n_cols = table_df.shape
        fig_h = max(2.8, 0.48 * n_rows + 1.3)
        fig_w = max(8.5, 1.8 * n_cols)
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
        table.set_fontsize(9)
        table.scale(1, 1.35)

        csv_path = str((self.output_dir / f"{filename}.csv"))
        df.to_csv(csv_path, index=False)
        saved = self._save(fig, self.output_dir / filename)
        saved.append(csv_path)
        return saved

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
        height: int = 720,
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
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=18, thickness=18, label=node_labels),
            link=dict(
                source=df[source_col].astype(str).map(node_to_idx).tolist(),
                target=df[target_col].astype(str).map(node_to_idx).tolist(),
                value=df[value_col].astype(float).tolist(),
            ),
        )])
        fig.update_layout(title_text=title, font_size=12, width=width, height=height)

        saved: List[str] = []
        html_path = str((self.output_dir / f"{filename}.html"))
        fig.write_html(html_path)
        saved.append(html_path)
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
        flow_df = self.build_sankey_flow_table(
            topic_stage_df,
            source_col=topic_col,
            target_col=stage_col,
            value_col=value_col,
            min_value=min_value,
            top_n_links=top_n_links,
            target_order=self.STAGE_ORDER,
        )
        saved = self.plot_sankey(flow_df, title=title, filename=filename)
        return flow_df, saved

    def plot_stage_pain_point_sankey(
        self,
        pain_point_table_df: pd.DataFrame,
        stage_col: str = "funnel_stage",
        pain_col: str = "pain_point_name",
        value_col: str = "topic_weight",
        min_value: float = 0.0,
        top_n_links: Optional[int] = 20,
        title: str = "Funnel Stage → Pain Point Sankey",
        filename: str = "stage_pain_point_sankey",
    ) -> Tuple[pd.DataFrame, List[str]]:
        self._validate_columns(pain_point_table_df, [stage_col, pain_col, value_col], "pain_point_table_df")
        flow_df = self.build_sankey_flow_table(
            pain_point_table_df,
            source_col=stage_col,
            target_col=pain_col,
            value_col=value_col,
            min_value=min_value,
            top_n_links=top_n_links,
            source_order=self.STAGE_ORDER,
        )
        saved = self.plot_sankey(flow_df, title=title, filename=filename)
        return flow_df, saved


if __name__ == "__main__":
    print("Use this visualizer with outputs from market_funnel_analyzer_aligned.py")
