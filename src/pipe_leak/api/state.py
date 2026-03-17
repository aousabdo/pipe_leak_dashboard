"""
Server-side state management.

Holds simulation data, trained model, and precomputed results in memory.
Provides methods to extract JSON-serializable data for the API.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pipe_leak.config import SimulationConfig, ML_CONFIG, PROCESSED_DIR, MODELS_DIR
from pipe_leak.simulation.network import build_pipe_network
from pipe_leak.simulation.events import generate_leak_events
from pipe_leak.ml.features import create_feature_dataset, get_feature_columns
from pipe_leak.ml.splits import temporal_train_test_split
from pipe_leak.ml.classifiers import LeakClassifier
from pipe_leak.ml.evaluate import (
    evaluate_predictions,
    compute_roc_curve,
    compute_pr_curve,
    compute_calibration_data,
)


class AppState:
    def __init__(self):
        self.pipes_gdf = None
        self.events_df = None
        self.model = None
        self.metrics = None
        self.importance = None
        self.y_test = None
        self.y_prob = None
        self.risk_scores = None
        self.sim_params = None

    @property
    def has_data(self) -> bool:
        return self.pipes_gdf is not None and self.events_df is not None

    @property
    def has_model(self) -> bool:
        return self.model is not None and self.metrics is not None

    def run_simulation(self, num_pipes: int, sim_years: int, seed: int):
        config = SimulationConfig(seed=seed, num_pipes=num_pipes, simulation_years=sim_years)
        self.pipes_gdf = build_pipe_network(config)
        self.events_df = generate_leak_events(self.pipes_gdf, config)
        self.sim_params = {"num_pipes": num_pipes, "sim_years": sim_years, "seed": seed}
        # Clear model when data changes
        self.model = None
        self.metrics = None
        self.importance = None
        self.risk_scores = None

    def train(self, model_type: str = "xgboost"):
        train_df, test_df = temporal_train_test_split(
            self.pipes_gdf, self.events_df,
            horizon_days=ML_CONFIG.prediction_horizon_days,
        )
        model = LeakClassifier(model_type=model_type)
        model.train(train_df, optimize=False)

        preds, probs = model.predict(test_df)
        y_test = test_df["target"].values
        metrics = evaluate_predictions(y_test, preds, probs)
        importance = model.get_feature_importance()

        self.model = model
        self.metrics = metrics
        self.importance = importance
        self.y_test = y_test
        self.y_prob = probs

        # Compute risk scores for all pipes
        if self.events_df is not None and not self.events_df.empty:
            latest_date = pd.to_datetime(self.events_df["date"]).max()
            pred_features = create_feature_dataset(
                self.pipes_gdf, self.events_df,
                latest_date, ML_CONFIG.prediction_horizon_days,
            )
            _, self.risk_scores = model.predict(pred_features)

    def get_status(self):
        return {
            "has_data": self.has_data,
            "has_model": self.has_model,
            "num_pipes": len(self.pipes_gdf) if self.pipes_gdf is not None else 0,
            "num_events": len(self.events_df) if self.events_df is not None and not self.events_df.empty else 0,
            "sim_params": self.sim_params,
            "model_type": self.model.model_type if self.model is not None else None,
        }

    def get_overview(self):
        pipes = self.pipes_gdf
        events = self.events_df

        n_pipes = len(pipes)
        n_events = len(events) if events is not None and not events.empty else 0
        total_cost = float(events["repair_cost"].sum()) if n_events > 0 else 0
        total_water_loss = float(events["water_loss_gallons"].sum()) if n_events > 0 else 0
        avg_age = float(pipes["age"].mean())
        high_risk_count = 0
        if self.risk_scores is not None:
            high_risk_count = int((self.risk_scores > 0.7).sum())

        # Severity breakdown
        severity_counts = {}
        if n_events > 0:
            severity_counts = events["severity"].value_counts().to_dict()

        # Monthly event trend
        monthly_trend = []
        if n_events > 0:
            ev = events.copy()
            ev["date"] = pd.to_datetime(ev["date"])
            ev["month"] = ev["date"].dt.to_period("M").astype(str)
            monthly = ev.groupby("month").agg(
                count=("pipe_id", "size"),
                cost=("repair_cost", "sum"),
                water_loss=("water_loss_gallons", "sum"),
            ).reset_index()
            monthly_trend = monthly.to_dict(orient="records")

        # Material distribution
        material_dist = pipes["material"].value_counts().to_dict()

        # Cost by severity
        cost_by_severity = {}
        if n_events > 0:
            cost_by_severity = events.groupby("severity")["repair_cost"].sum().to_dict()

        # Pipe age distribution (binned)
        age_bins = [0, 10, 20, 30, 40, 50, 60, 80, 120]
        age_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
        age_hist = pd.cut(pipes["age"], bins=age_bins, labels=age_labels, right=False)
        age_dist = age_hist.value_counts().sort_index().to_dict()

        return {
            "kpis": {
                "total_pipes": n_pipes,
                "total_events": n_events,
                "total_cost": round(total_cost, 2),
                "total_water_loss": round(total_water_loss, 0),
                "avg_pipe_age": round(avg_age, 1),
                "high_risk_pipes": high_risk_count,
                "avg_cost_per_event": round(total_cost / max(n_events, 1), 2),
                "events_per_pipe": round(n_events / max(n_pipes, 1), 2),
            },
            "severity_counts": severity_counts,
            "monthly_trend": monthly_trend,
            "material_distribution": material_dist,
            "cost_by_severity": {k: round(v, 2) for k, v in cost_by_severity.items()},
            "age_distribution": age_dist,
        }

    def get_pipes(self):
        pipes = self.pipes_gdf.copy()

        # Add risk scores
        if self.risk_scores is not None:
            pipes["risk_score"] = self.risk_scores
        else:
            pipes["risk_score"] = 0.0

        # Convert geometry to coordinates for deck.gl
        records = []
        for _, row in pipes.iterrows():
            geom = row["geometry"]
            coords = list(geom.coords) if hasattr(geom, "coords") else []
            records.append({
                "pipe_id": row["pipe_id"],
                "start_lon": coords[0][0] if len(coords) > 0 else row["mid_lon"],
                "start_lat": coords[0][1] if len(coords) > 0 else row["mid_lat"],
                "end_lon": coords[-1][0] if len(coords) > 1 else row["mid_lon"],
                "end_lat": coords[-1][1] if len(coords) > 1 else row["mid_lat"],
                "mid_lon": float(row["mid_lon"]),
                "mid_lat": float(row["mid_lat"]),
                "material": row.get("material", "Unknown"),
                "age": int(row.get("age", 0)),
                "diameter_m": float(row.get("diameter_m", 0.15)),
                "diameter_category": row.get("diameter_category", ""),
                "soil_type": row.get("soil_type", "Unknown"),
                "pressure_avg_m": float(row.get("pressure_avg_m", 30.0)),
                "installation_year": int(row.get("installation_year", 2000)),
                "prev_repairs": int(row.get("prev_repairs", 0)),
                "risk_score": float(row.get("risk_score", 0.0)),
                "length_m": float(row.get("length_m", 200)),
            })

        return {"pipes": records}

    def get_events(self, filters=None):
        ev = self.events_df.copy()
        ev["date"] = pd.to_datetime(ev["date"])

        if filters:
            if filters.severity:
                ev = ev[ev["severity"].isin(filters.severity)]
            if filters.material:
                ev = ev[ev["material"].isin(filters.material)]
            if filters.date_from:
                ev = ev[ev["date"] >= pd.to_datetime(filters.date_from)]
            if filters.date_to:
                ev = ev[ev["date"] <= pd.to_datetime(filters.date_to)]

        ev["date"] = ev["date"].dt.strftime("%Y-%m-%d")
        return {"events": ev.to_dict(orient="records"), "total": len(ev)}

    def get_analysis(self):
        events = self.events_df.copy()
        pipes = self.pipes_gdf
        events["date"] = pd.to_datetime(events["date"])

        # Monthly trend by severity
        events["month"] = events["date"].dt.to_period("M").astype(str)
        monthly_severity = events.groupby(["month", "severity"]).size().reset_index(name="count")
        monthly_severity_data = monthly_severity.to_dict(orient="records")

        # Material risk profile
        material_stats = events.groupby("material").agg(
            event_count=("pipe_id", "size"),
            avg_cost=("repair_cost", "mean"),
            total_cost=("repair_cost", "sum"),
            avg_water_loss=("water_loss_gallons", "mean"),
        ).reset_index()
        # Add pipe count per material
        pipe_material_counts = pipes["material"].value_counts().to_dict()
        material_stats["pipe_count"] = material_stats["material"].map(pipe_material_counts).fillna(0).astype(int)
        material_stats["events_per_pipe"] = (material_stats["event_count"] / material_stats["pipe_count"].clip(lower=1)).round(2)
        material_stats = material_stats.round(2)

        # Yearly trend
        events["year"] = events["date"].dt.year
        yearly = events.groupby("year").agg(
            count=("pipe_id", "size"),
            cost=("repair_cost", "sum"),
        ).reset_index()

        # Soil type analysis
        soil_stats = events.groupby("soil_type").agg(
            event_count=("pipe_id", "size"),
            avg_cost=("repair_cost", "mean"),
        ).reset_index().round(2)

        # Cost by severity and year
        cost_sev_year = events.groupby(["year", "severity"])["repair_cost"].sum().reset_index()
        cost_sev_year = cost_sev_year.round(2)

        # Peak month
        month_counts = events["date"].dt.month.value_counts()
        peak_month = int(month_counts.index[0]) if len(month_counts) > 0 else 1
        month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                       5: "May", 6: "June", 7: "July", 8: "August",
                       9: "September", 10: "October", 11: "November", 12: "December"}

        return {
            "monthly_severity": monthly_severity_data,
            "material_risk": material_stats.to_dict(orient="records"),
            "yearly_trend": yearly.to_dict(orient="records"),
            "soil_analysis": soil_stats.to_dict(orient="records"),
            "cost_by_severity_year": cost_sev_year.to_dict(orient="records"),
            "insights": {
                "peak_month": month_names.get(peak_month, "Unknown"),
                "most_affected_material": material_stats.sort_values("events_per_pipe", ascending=False).iloc[0]["material"] if len(material_stats) > 0 else "N/A",
                "total_cost": round(float(events["repair_cost"].sum()), 2),
                "avg_detection_hours": round(float(events["detection_hours"].mean()), 1),
            },
        }

    def get_model_performance(self):
        metrics = self.metrics.copy()
        importance = self.importance

        # ROC curve
        roc_data = {}
        pr_data = {}
        cal_data = {}
        if self.y_test is not None and self.y_prob is not None:
            n_pos = self.y_test.sum()
            n_neg = len(self.y_test) - n_pos
            if n_pos > 0 and n_neg > 0:
                roc_data = compute_roc_curve(self.y_test, self.y_prob)
                pr_data = compute_pr_curve(self.y_test, self.y_prob)
                cal_data = compute_calibration_data(self.y_test, self.y_prob)

        # Feature importance
        imp_data = []
        if importance is not None:
            imp_data = importance.head(15).to_dict(orient="records")

        # Clean confusion matrix for JSON
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

        return {
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in metrics.items()
                if k != "confusion_matrix"
            },
            "confusion_matrix": cm,
            "roc_curve": roc_data,
            "pr_curve": pr_data,
            "calibration": cal_data,
            "feature_importance": imp_data,
            "model_type": self.model.model_type if self.model else "xgboost",
            "optimal_threshold": self.model.optimal_threshold if self.model else 0.5,
        }

    def get_filter_options(self):
        events = self.events_df
        pipes = self.pipes_gdf
        dates = pd.to_datetime(events["date"])

        return {
            "severities": sorted(events["severity"].unique().tolist()),
            "materials": sorted(pipes["material"].unique().tolist()),
            "date_range": {
                "min": dates.min().strftime("%Y-%m-%d"),
                "max": dates.max().strftime("%Y-%m-%d"),
            },
        }
