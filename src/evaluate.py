from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from plots import (
    save_class_distribution_plot,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_model_comparison_plot,
)
from preprocess import load_dataset, TARGET_COLUMN
from train import train_all_models


METRICS_PATH = Path("reports/model_metrics.csv")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.csv")


def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    """Evaluate classification performance with fraud-focused metrics."""
    predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def get_feature_importance(model_name: str, model, feature_columns: list[str]) -> pd.DataFrame:
    """Return model feature importances when the estimator supports them."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["model", "feature", "importance"])

    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def evaluate_all_models() -> pd.DataFrame:
    """Train all models, save evaluation metrics, and save confusion matrix plots."""
    trained_models, X_test, y_test = train_all_models()
    results = []
    feature_importance_frames = []

    for model_name, metadata in trained_models.items():
        model = metadata["model"]
        metrics = evaluate_model(model_name, model, X_test, y_test)
        metrics["model_path"] = str(metadata["model_path"])
        metrics["confusion_matrix_plot"] = str(save_confusion_matrix_plot(model_name, model, X_test, y_test))
        results.append(metrics)

        feature_importances = get_feature_importance(model_name, model, X_test.columns.tolist())
        if not feature_importances.empty:
            feature_importance_frames.append(feature_importances)
            save_feature_importance_plot(model_name, feature_importances)

    metrics_df = pd.DataFrame(results)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    save_model_comparison_plot(metrics_df)
    dataset = load_dataset()
    save_class_distribution_plot(dataset[TARGET_COLUMN].value_counts())

    if feature_importance_frames:
        feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
        feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return metrics_df


if __name__ == "__main__":
    metrics = evaluate_all_models()
    print(metrics.to_string(index=False))
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved feature importances to {FEATURE_IMPORTANCE_PATH}")
