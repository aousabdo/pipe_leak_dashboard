#!/usr/bin/env python3
"""CLI script to train the prediction model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipe_leak.config import ML_CONFIG
from pipe_leak.utils.io import load_pipes, load_events
from pipe_leak.ml.splits import temporal_train_test_split
from pipe_leak.ml.classifiers import LeakClassifier
from pipe_leak.ml.evaluate import evaluate_predictions
from pipe_leak.ml.features import get_feature_columns
from pipe_leak.ml.registry import save_model


def main():
    print("=" * 60)
    print("Pipe Leak Model Training")
    print("=" * 60)

    # Load data
    pipes_gdf = load_pipes()
    events_df = load_events()

    if pipes_gdf is None or events_df is None:
        print("Error: No simulation data found. Run `python scripts/run_simulation.py` first.")
        sys.exit(1)

    print(f"Loaded {len(pipes_gdf)} pipes and {len(events_df)} events")

    # Split data
    train_df, test_df = temporal_train_test_split(
        pipes_gdf, events_df, horizon_days=ML_CONFIG.prediction_horizon_days
    )
    print(f"Train: {len(train_df)} samples ({train_df['target'].sum()} positive)")
    print(f"Test:  {len(test_df)} samples ({test_df['target'].sum()} positive)")

    # Train
    model = LeakClassifier(ML_CONFIG)
    train_info = model.train(train_df, optimize=False)
    print(f"\nTraining info: {train_info}")

    # Evaluate
    preds, probs = model.predict(test_df)
    y_test = test_df["target"].values
    metrics = evaluate_predictions(y_test, preds, probs)

    print("\n--- Test Metrics ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")

    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    # Save
    save_model(model, {**train_info, **metrics})
    print("\nDone.")


if __name__ == "__main__":
    main()
