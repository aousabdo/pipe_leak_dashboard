"""Model performance page: metrics, feature importance, curves."""

import streamlit as st
import pandas as pd
import numpy as np

from pipe_leak.dashboard.components.charts import (
    model_metrics_chart,
    feature_importance_chart,
)
from pipe_leak.ml.evaluate import compute_roc_curve, compute_pr_curve, compute_calibration_data
import plotly.graph_objects as go


def render(
    metrics: dict | None,
    importance_df: pd.DataFrame | None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
):
    """Render the model performance page."""
    st.markdown('<div class="section-header">Prediction Model</div>', unsafe_allow_html=True)

    if metrics is None:
        st.info("Train the model to see performance metrics.")
        return

    # Metrics and importance side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(model_metrics_chart(metrics), use_container_width=True)

    with col2:
        st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

    # Curves if we have probability data
    if y_true is not None and y_prob is not None and len(np.unique(y_true)) > 1:
        st.markdown("### Model Curves")
        col3, col4, col5 = st.columns(3)

        with col3:
            roc = compute_roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                name=f"ROC (AUC={roc['auc']:.3f})", line=dict(color="#3182ce"),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name="Random", line=dict(dash="dash", color="#999"),
            ))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            pr = compute_pr_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pr["recall"], y=pr["precision"],
                name=f"PR (AUC={pr['pr_auc']:.3f})", line=dict(color="#38a169"),
            ))
            fig.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col5:
            cal = compute_calibration_data(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cal["predicted"], y=cal["actual"],
                name="Model", mode="lines+markers", line=dict(color="#805ad5"),
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name="Perfect", line=dict(dash="dash", color="#999"),
            ))
            fig.update_layout(
                title="Calibration Plot",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Fraction Positive",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    if metrics.get("confusion_matrix"):
        with st.expander("Confusion Matrix"):
            cm = np.array(metrics["confusion_matrix"])
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual: No Leak", "Actual: Leak"],
                    columns=["Pred: No Leak", "Pred: Leak"],
                ),
                use_container_width=True,
            )
