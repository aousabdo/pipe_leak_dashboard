"""
Model save/load with metadata tracking.
"""

import json
import joblib
from pathlib import Path
from datetime import datetime

from pipe_leak.config import MODELS_DIR


def save_model(model, metadata: dict, name: str = "leak_classifier") -> Path:
    """
    Save model and metadata to the models directory.

    Args:
        model: The trained model object (LeakClassifier instance).
        metadata: Dict of training metadata (params, metrics, etc.).
        name: Base name for the saved files.

    Returns:
        Path to the saved model directory.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"{name}_{timestamp}"
    model_dir.mkdir(exist_ok=True)

    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    meta_path = model_dir / "metadata.json"
    # Make metadata JSON-serializable
    clean_meta = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
            clean_meta[k] = v
        except (TypeError, ValueError):
            clean_meta[k] = str(v)

    clean_meta["saved_at"] = timestamp
    with open(meta_path, "w") as f:
        json.dump(clean_meta, f, indent=2)

    print(f"Model saved to {model_dir}")
    return model_dir


def load_latest_model(name: str = "leak_classifier"):
    """
    Load the most recently saved model.

    Args:
        name: Base name to match.

    Returns:
        The loaded model object, or None if no model found.
    """
    if not MODELS_DIR.exists():
        return None

    # Find matching model directories, sorted by name (timestamp suffix)
    dirs = sorted(MODELS_DIR.glob(f"{name}_*"), reverse=True)
    if not dirs:
        return None

    model_path = dirs[0] / "model.joblib"
    if not model_path.exists():
        return None

    return joblib.load(model_path)
