"""Model performance page: metrics, curves, feature importance, confusion matrix."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipe_leak.dashboard.components.charts import (
    model_metrics_chart,
    feature_importance_chart,
    _apply_style,
)
from pipe_leak.ml.evaluate import compute_roc_curve, compute_pr_curve, compute_calibration_data


def render(
    metrics: dict | None,
    importance_df: pd.DataFrame | None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
):
    """Render the model performance page."""
    if metrics is None:
        st.info("Train the model to see performance metrics.")
        return

    # Top-level metric badges
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    metric_keys = [
        ("accuracy", "Accuracy", "blue"),
        ("precision", "Precision", "green"),
        ("recall", "Recall", "amber"),
        ("f1", "F1 Score", "red"),
        ("roc_auc", "ROC AUC", "purple"),
        ("pr_auc", "PR AUC", "cyan"),
    ]

    cols = st.columns(len([k for k, _, _ in metric_keys if k in metrics]))
    col_idx = 0
    for key, label, accent in metric_keys:
        if key in metrics:
            val = metrics[key]
            with cols[col_idx]:
                st.markdown(
                    f'<div class="kpi-card accent-{accent}">'
                    f'<div class="kpi-value">{val:.3f}</div>'
                    f'<div class="kpi-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            col_idx += 1

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Charts: metrics bar + feature importance
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(model_metrics_chart(metrics), use_container_width=True)
    with col2:
        st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

    # Model curves
    if y_true is not None and y_prob is not None and len(np.unique(y_true)) > 1:
        st.markdown('<div class="section-title">Diagnostic Curves</div>', unsafe_allow_html=True)

        col3, col4, col5 = st.columns(3)

        with col3:
            roc = compute_roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                name=f"Model (AUC={roc['auc']:.3f})",
                line=dict(color="#3b82f6", width=3),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name="Random",
                line=dict(dash="dash", color="#94a3b8", width=1.5),
                hoverinfo="skip",
            ))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.05),
            )
            st.plotly_chart(_apply_style(fig), use_container_width=True)

        with col4:
            pr = compute_pr_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pr["recall"], y=pr["precision"],
                name=f"Model (AUC={pr['pr_auc']:.3f})",
                line=dict(color="#22c55e", width=3),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.08)",
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            ))
            fig.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(x=0.05, y=0.05),
            )
            st.plotly_chart(_apply_style(fig), use_container_width=True)

        with col5:
            cal = compute_calibration_data(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cal["predicted"], y=cal["actual"],
                name="Model",
                mode="lines+markers",
                line=dict(color="#8b5cf6", width=3),
                marker=dict(size=8, color="#8b5cf6", line=dict(width=2, color="white")),
                hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name="Perfect Calibration",
                line=dict(dash="dash", color="#94a3b8", width=1.5),
                hoverinfo="skip",
            ))
            fig.update_layout(
                title="Calibration Plot",
                xaxis_title="Predicted Probability",
                yaxis_title="Observed Fraction",
                legend=dict(x=0.05, y=0.95),
            )
            st.plotly_chart(_apply_style(fig), use_container_width=True)

    # Confusion matrix as heatmap
    if metrics.get("confusion_matrix"):
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

        cm = np.array(metrics["confusion_matrix"])

        # Center it in a narrow column
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            fig = go.Figure(go.Heatmap(
                z=cm[::-1],
                x=["Predicted: No Leak", "Predicted: Leak"],
                y=["Actual: Leak", "Actual: No Leak"],
                text=cm[::-1],
                texttemplate="%{text:,}",
                textfont=dict(size=18, color="white"),
                colorscale=[[0, "#1e3a5f"], [1, "#3b82f6"]],
                showscale=False,
                hovertemplate="<b>%{y}</b> / <b>%{x}</b><br>Count: %{z:,}<extra></extra>",
            ))
            fig.update_layout(
                height=350,
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(_apply_style(fig), use_container_width=True)

    # Brier score
    if "brier_score" in metrics:
        st.caption(f"Brier Score: {metrics['brier_score']:.4f} (lower is better, 0 = perfect calibration)")
