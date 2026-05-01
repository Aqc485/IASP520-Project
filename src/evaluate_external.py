import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from plots import save_confusion_matrix_plot
from preprocess import TARGET_COLUMN


MODEL_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")


def load_external_dataset(path: Path) -> pd.DataFrame:
    """Load a labeled external dataset for model testing."""
    return pd.read_csv(path)


def align_external_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Align an external dataset to the feature columns saved with a trained model."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"External dataset must include target column '{TARGET_COLUMN}'.")

    aligned_df = df.copy()

    # The 2023 dataset uses id instead of Time. Use it only for compatibility with the
    # already-trained model and document the result as an external validation check.
    if "Time" in feature_columns and "Time" not in aligned_df.columns and "id" in aligned_df.columns:
        aligned_df["Time"] = aligned_df["id"]

    missing_columns = [column for column in feature_columns if column not in aligned_df.columns]
    if missing_columns:
        raise ValueError(f"External dataset is missing required feature columns: {missing_columns}")

    X = aligned_df[feature_columns]
    y = aligned_df[TARGET_COLUMN]
    return X, y


def evaluate_saved_model(model_path: Path, dataset_path: Path) -> dict:
    """Evaluate one saved model artifact on an external labeled dataset."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    df = load_external_dataset(dataset_path)
    X_external, y_external = align_external_features(df, feature_columns)
    predictions = model.predict(X_external)
    tn, fp, fn, tp = confusion_matrix(y_external, predictions).ravel()

    plot_path = save_confusion_matrix_plot(
        f"{model_path.stem}_external_2023",
        model,
        X_external,
        y_external,
    )

    return {
        "model": model_path.stem,
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "accuracy": accuracy_score(y_external, predictions),
        "precision": precision_score(y_external, predictions, zero_division=0),
        "recall": recall_score(y_external, predictions, zero_division=0),
        "f1": f1_score(y_external, predictions, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix_plot": str(plot_path),
    }


def evaluate_external_dataset(dataset_path: Path, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Evaluate all saved model artifacts on a separate labeled dataset."""
    model_paths = sorted(model_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model artifacts found in {model_dir}. Run src/evaluate.py first.")

    results = [evaluate_saved_model(model_path, dataset_path) for model_path in model_paths]
    results_df = pd.DataFrame(results)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / f"external_{dataset_path.stem}_metrics.csv"
    results_df.to_csv(output_path, index=False)
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models on an external labeled dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/creditcard_2023.csv"),
        help="Path to external labeled CSV dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_external_dataset(args.data)
    print(metrics.to_string(index=False))
    print(f"Saved external metrics to outputs/metrics/external_{args.data.stem}_metrics.csv")
