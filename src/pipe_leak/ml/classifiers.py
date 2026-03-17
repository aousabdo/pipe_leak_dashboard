"""
ML classifiers for pipe leak prediction.

Supports individual models and advanced ensemble methods:
- Individual: XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression
- Ensembles: Stacking (meta-learner), Soft Voting, Blended Boosting

Uses scale_pos_weight / class_weight for class imbalance handling.
Finds optimal classification threshold based on F1 score.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from pipe_leak.config import ML_CONFIG, MLConfig
from pipe_leak.ml.features import get_feature_columns


# Supported model types
MODEL_TYPES = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "gradient_boosting",
    "logistic_regression",
    "stacking_ensemble",
    "voting_ensemble",
    "blended_boosting",
]

# Display labels for the frontend
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "stacking_ensemble": "Stacking Ensemble",
    "voting_ensemble": "Voting Ensemble",
    "blended_boosting": "Blended Boosting",
}


class LeakClassifier:
    """Multi-model pipe leak classifier with optimal threshold selection."""

    def __init__(self, config: MLConfig | None = None, model_type: str = "xgboost"):
        self.config = config or ML_CONFIG
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.optimal_threshold: float = 0.5
        self._base_models: dict = {}  # stores base models for ensembles

    def train(
        self,
        train_df: pd.DataFrame,
        optimize: bool = False,
    ) -> dict:
        """
        Train the classifier on the provided feature dataset.

        Returns:
            Dict of training info (class distribution, model details).
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

        # Apply SMOTE resampling for stacking/voting ensembles (better class balance)
        use_smote = self.model_type in ("stacking_ensemble", "voting_ensemble")
        if use_smote and n_pos >= 6:
            smote = SMOTE(
                sampling_strategy=0.75,  # minority becomes 75% of majority
                k_neighbors=min(5, n_pos - 1),
                random_state=self.config.random_state,
            )
            X_train, y_train = smote.fit_resample(X_scaled, y)
        else:
            X_train, y_train = X_scaled, y

        # Recompute scale_pos_weight after resampling
        n_pos_train = int(y_train.sum())
        n_neg_train = len(y_train) - n_pos_train
        spw_train = n_neg_train / max(n_pos_train, 1)

        # Build the model
        builders = {
            "xgboost": lambda: self._train_xgboost(X_train, y_train, spw_train, optimize),
            "lightgbm": lambda: self._train_lightgbm(X_train, y_train, spw_train),
            "random_forest": lambda: self._train_random_forest(X_train, y_train),
            "gradient_boosting": lambda: self._train_gradient_boosting(X_train, y_train),
            "logistic_regression": lambda: self._train_logistic_regression(X_train, y_train),
            "stacking_ensemble": lambda: self._train_stacking(X_train, y_train, spw_train),
            "voting_ensemble": lambda: self._train_voting(X_train, y_train, spw_train),
            "blended_boosting": lambda: self._train_blended_boosting(X_train, y_train, spw_train),
        }

        if self.model_type not in builders:
            raise ValueError(f"Unknown model type: {self.model_type}. Choose from: {MODEL_TYPES}")

        self.model = builders[self.model_type]()

        # Find optimal threshold on ORIGINAL data (not SMOTE'd) to avoid bias
        train_probs = self.model.predict_proba(X_scaled)[:, 1]
        self.optimal_threshold = self._find_optimal_threshold(y, train_probs)

        return {
            "model_type": self.model_type,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "optimal_threshold": round(self.optimal_threshold, 4),
        }

    # ── Individual models ────────────────────────────────────────────────────

    def _train_xgboost(self, X, y, scale_pos_weight, optimize):
        if optimize:
            param_dist = {
                "n_estimators": randint(100, 500),
                "max_depth": randint(3, 10),
                "learning_rate": uniform(0.01, 0.2),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_alpha": uniform(0, 1),
                "reg_lambda": uniform(0.5, 2),
            }
            base = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state,
            )
            search = RandomizedSearchCV(
                base, param_dist, n_iter=30, cv=3,
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

    def _train_lightgbm(self, X, y, scale_pos_weight):
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=10,
            random_state=self.config.random_state,
            verbose=-1,
        )
        model.fit(X, y)
        return model

    def _train_random_forest(self, X, y):
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=5,
            class_weight="balanced",
            max_features="sqrt",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_gradient_boosting(self, X, y):
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        sample_weights = np.where(y == 1, n_neg / max(n_pos, 1), 1.0)
        model = GradientBoostingClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=self.config.random_state,
        )
        model.fit(X, y, sample_weight=sample_weights)
        return model

    def _train_logistic_regression(self, X, y):
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            random_state=self.config.random_state,
        )
        model.fit(X, y)
        return model

    # ── Ensemble models ──────────────────────────────────────────────────────

    def _train_stacking(self, X, y, scale_pos_weight):
        """
        Stacking Ensemble: XGBoost + LightGBM + Random Forest + Extra Trees
        with a calibrated Logistic Regression meta-learner.

        Each base model is trained via cross-validation, then the meta-learner
        learns the optimal combination of their predictions.
        """
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)

        base_estimators = [
            ("xgb", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state,
            )),
            ("lgbm", LGBMClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=10, verbose=-1,
                random_state=self.config.random_state + 1,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=8,
                min_samples_leaf=3, class_weight="balanced",
                max_features="sqrt",
                random_state=self.config.random_state + 2,
                n_jobs=-1,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=300, max_depth=10,
                min_samples_leaf=3, class_weight="balanced",
                max_features="sqrt",
                random_state=self.config.random_state + 3,
                n_jobs=-1,
            )),
        ]

        meta_learner = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000,
            random_state=self.config.random_state,
        )

        model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv,
            stack_method="predict_proba",
            passthrough=False,  # meta-learner only sees base model outputs
            n_jobs=-1,
        )
        model.fit(X, y)

        # Store base model names for feature importance
        self._base_models = {name: est for name, est in base_estimators}

        return model

    def _train_voting(self, X, y, scale_pos_weight):
        """
        Soft Voting Ensemble: Calibrated combination of diverse models.

        Uses CalibratedClassifierCV to ensure well-calibrated probabilities
        before averaging, which improves the quality of the soft vote.
        """
        xgb = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state,
            ),
            cv=3, method="isotonic",
        )
        lgbm = CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=10, verbose=-1,
                random_state=self.config.random_state + 1,
            ),
            cv=3, method="isotonic",
        )
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=10,
            min_samples_leaf=3, class_weight="balanced",
            random_state=self.config.random_state + 2,
            n_jobs=-1,
        )
        gb = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.85,
                random_state=self.config.random_state + 3,
            ),
            cv=3, method="sigmoid",
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.config.random_state + 4,
        )

        model = VotingClassifier(
            estimators=[
                ("xgb", xgb),
                ("lgbm", lgbm),
                ("rf", rf),
                ("gb", gb),
                ("mlp", mlp),
            ],
            voting="soft",
            weights=[2, 2, 1.5, 1.5, 1],  # higher weight for gradient boosters
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_blended_boosting(self, X, y, scale_pos_weight):
        """
        Blended Boosting: Multiple boosting variants with different architectures
        and hyperparameters, combined via soft voting.

        Diversity comes from different algorithms, depths, learning rates, and seeds.
        """
        boosters = [
            ("xgb_deep", XGBClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=0.1, reg_lambda=1.5,
                eval_metric="logloss",
                random_state=self.config.random_state,
            )),
            ("xgb_shallow", XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.config.random_state + 10,
            )),
            ("lgbm_wide", LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=10, verbose=-1,
                random_state=self.config.random_state + 20,
            )),
            ("lgbm_deep", LGBMClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.03,
                num_leaves=63, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=5, verbose=-1,
                random_state=self.config.random_state + 30,
            )),
        ]

        model = VotingClassifier(
            estimators=boosters,
            voting="soft",
            weights=[1.5, 1, 1.5, 1],
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    # ── Threshold & Prediction ───────────────────────────────────────────────

    def _find_optimal_threshold(self, y_true, y_prob):
        """Find threshold that maximizes F1 score."""
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in np.arange(0.05, 0.95, 0.005):
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
        """Make predictions using the optimal threshold."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= self.optimal_threshold).astype(int)

        return predictions, probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances sorted by importance.

        For ensembles, aggregates importances across base models.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importances = self._extract_importances()

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _extract_importances(self) -> np.ndarray:
        """Extract feature importances, handling ensembles gracefully."""
        model = self.model

        # Direct model with feature_importances_
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_

        # Logistic regression coefficients
        if hasattr(model, "coef_"):
            return np.abs(model.coef_[0])

        # StackingClassifier — average base model importances
        if isinstance(model, StackingClassifier):
            return self._aggregate_estimator_importances(model.estimators_)

        # VotingClassifier — average base model importances
        if isinstance(model, VotingClassifier):
            estimators = []
            for est in model.estimators_:
                # Unwrap CalibratedClassifierCV
                if isinstance(est, CalibratedClassifierCV):
                    if hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
                        estimators.append(est.calibrated_classifiers_[0].estimator)
                    else:
                        estimators.append(est)
                else:
                    estimators.append(est)
            return self._aggregate_estimator_importances(estimators)

        return np.zeros(len(self.feature_names))

    def _aggregate_estimator_importances(self, estimators) -> np.ndarray:
        """Average feature importances across multiple estimators."""
        n_features = len(self.feature_names)
        all_importances = []

        for est in estimators:
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
                if len(imp) == n_features:
                    all_importances.append(imp / max(imp.sum(), 1e-8))
            elif hasattr(est, "coef_"):
                imp = np.abs(est.coef_[0])
                if len(imp) == n_features:
                    all_importances.append(imp / max(imp.sum(), 1e-8))

        if all_importances:
            return np.mean(all_importances, axis=0)
        return np.zeros(n_features)
