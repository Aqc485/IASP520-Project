from pathlib import Path

import joblib

from models import get_models
from preprocess import create_train_test_split, load_dataset, scale_amount_and_time


MODEL_DIR = Path("outputs/models")


def prepare_training_data():
    """Load, split, and scale the fraud dataset."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    X_train_scaled, X_test_scaled, scaler = scale_amount_and_time(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_model_artifact(model_name: str, model, scaler, feature_columns: list[str]) -> Path:
    """Save a trained model with preprocessing metadata."""
    model_path = MODEL_DIR / f"{model_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        },
        model_path,
    )
    return model_path


def train_model(model_name: str, model, X_train, y_train, scaler) -> tuple:
    """Train one model and save its artifact."""
    model.fit(X_train, y_train)
    model_path = save_model_artifact(model_name, model, scaler, X_train.columns.tolist())
    return model, model_path


def train_all_models():
    """Train the baseline and improved fraud detection models."""
    X_train, X_test, y_train, y_test, scaler = prepare_training_data()
    trained_models = {}

    for model_name, model in get_models().items():
        trained_model, model_path = train_model(model_name, model, X_train, y_train, scaler)
        trained_models[model_name] = {
            "model": trained_model,
            "model_path": model_path,
        }

    return trained_models, X_test, y_test


def train_baseline_model():
    """Train only the Logistic Regression baseline for quick checks."""
    X_train, X_test, y_train, y_test, scaler = prepare_training_data()
    model = get_models()["logistic_regression"]
    trained_model, _ = train_model("logistic_regression", model, X_train, y_train, scaler)
    return trained_model, X_test, y_test


if __name__ == "__main__":
    models, _, _ = train_all_models()
    for name, metadata in models.items():
        print(f"Saved {name} model to {metadata['model_path']}")
