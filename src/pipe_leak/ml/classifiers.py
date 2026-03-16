"""
XGBoost binary classifier for pipe leak prediction.

Clean implementation without SMOTE, artificial metric caps, or DummyModel
workarounds. Uses scale_pos_weight for class imbalance handling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

from pipe_leak.config import ML_CONFIG, MLConfig
from pipe_leak.ml.features import get_feature_columns


class LeakClassifier:
    """XGBoost-based pipe leak classifier."""

    def __init__(self, config: MLConfig | None = None):
        self.config = config or ML_CONFIG
        self.model: XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def train(
        self,
        train_df: pd.DataFrame,
        optimize: bool = False,
    ) -> dict:
        """
        Train the classifier on the provided feature dataset.

        Args:
            train_df: DataFrame with features and 'target' column.
            optimize: If True, run randomized hyperparameter search.

        Returns:
            Dict of training info (class distribution, params used).
        """
        feature_cols = get_feature_columns(train_df)
        self.feature_names = feature_cols

        X = train_df[feature_cols].values
        y = train_df["target"].values

        # Check class distribution
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if n_pos == 0:
            raise ValueError(
                f"No positive examples in training data. "
                f"All {n_neg} pipes had no leaks in the target window. "
                f"Try a longer prediction horizon or more simulation years."
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance via scale_pos_weight
        scale_pos_weight = n_neg / max(n_pos, 1)

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
                base_model,
                param_dist,
                n_iter=30,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=self.config.random_state,
            )
            search.fit(X_scaled, y)
            self.model = search.best_estimator_
            best_params = search.best_params_
        else:
            params = self.config.xgb_params.copy()
            self.model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                **params,
            )
            self.model.fit(X_scaled, y)
            best_params = params

        return {
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "scale_pos_weight": round(scale_pos_weight, 2),
            "params": best_params,
        }

    def predict(self, features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            features_df: DataFrame with the same feature columns as training.

        Returns:
            (predictions, probabilities) where predictions are 0/1 and
            probabilities are P(leak) in [0, 1].
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances sorted by importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
