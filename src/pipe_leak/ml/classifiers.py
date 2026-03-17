"""
ML classifiers for pipe leak prediction.

Supports multiple model types: XGBoost, Random Forest, Logistic Regression.
Uses scale_pos_weight / class_weight for class imbalance handling.
Finds optimal classification threshold based on F1 score.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

from pipe_leak.config import ML_CONFIG, MLConfig
from pipe_leak.ml.features import get_feature_columns


# Supported model types
MODEL_TYPES = ["xgboost", "random_forest", "logistic_regression", "gradient_boosting"]


class LeakClassifier:
    """Multi-model pipe leak classifier with optimal threshold selection."""

    def __init__(self, config: MLConfig | None = None, model_type: str = "xgboost"):
        self.config = config or ML_CONFIG
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.optimal_threshold: float = 0.5

    def train(
        self,
        train_df: pd.DataFrame,
        optimize: bool = False,
    ) -> dict:
        """
        Train the classifier on the provided feature dataset.

        Args:
            train_df: DataFrame with features and 'target' column.
            optimize: If True, run randomized hyperparameter search (XGBoost only).

        Returns:
            Dict of training info (class distribution, params used).
        """
        feature_cols = get_feature_columns(train_df)
        self.feature_names = feature_cols

        X = train_df[feature_cols].values
        y = train_df["target"].values

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if n_pos == 0:
            raise ValueError(
                f"No positive examples in training data. "
                f"All {n_neg} pipes had no leaks in the target window. "
                f"Try a longer prediction horizon or more simulation years."
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        scale_pos_weight = n_neg / max(n_pos, 1)

        if self.model_type == "xgboost":
            self.model = self._train_xgboost(X_scaled, y, scale_pos_weight, optimize)
        elif self.model_type == "random_forest":
            self.model = self._train_random_forest(X_scaled, y)
        elif self.model_type == "logistic_regression":
            self.model = self._train_logistic_regression(X_scaled, y)
        elif self.model_type == "gradient_boosting":
            self.model = self._train_gradient_boosting(X_scaled, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Find optimal threshold using training data
        train_probs = self.model.predict_proba(X_scaled)[:, 1]
        self.optimal_threshold = self._find_optimal_threshold(y, train_probs)

        return {
            "model_type": self.model_type,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "optimal_threshold": round(self.optimal_threshold, 4),
        }

    def _train_xgboost(self, X, y, scale_pos_weight, optimize):
        if optimize:
            param_dist = {
                "n_estimators": randint(100, 400),
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.2),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3),
            }
            base_model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state,
            )
            search = RandomizedSearchCV(
                base_model, param_dist, n_iter=30, cv=3,
                scoring="roc_auc", n_jobs=-1,
                random_state=self.config.random_state,
            )
            search.fit(X, y)
            return search.best_estimator_
        else:
            params = self.config.xgb_params.copy()
            model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                **params,
            )
            model.fit(X, y)
            return model

    def _train_random_forest(self, X, y):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_logistic_regression(self, X, y):
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=self.config.random_state,
        )
        model.fit(X, y)
        return model

    def _train_gradient_boosting(self, X, y):
        # Compute sample weights for imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        sample_weights = np.where(y == 1, n_neg / max(n_pos, 1), 1.0)
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            random_state=self.config.random_state,
        )
        model.fit(X, y, sample_weight=sample_weights)
        return model

    def _find_optimal_threshold(self, y_true, y_prob):
        """Find threshold that maximizes F1 score."""
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in np.arange(0.05, 0.95, 0.01):
            preds = (y_prob >= threshold).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return best_threshold

    def predict(self, features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the optimal threshold.

        Returns:
            (predictions, probabilities) where predictions use optimal threshold.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        return predictions, probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances sorted by importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Logistic regression: use absolute coefficient values
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
