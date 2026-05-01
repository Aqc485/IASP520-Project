from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


PLOTS_DIR = Path("outputs/plots")


def save_confusion_matrix_plot(model_name: str, model, X_test, y_test) -> Path:
    """Save a confusion matrix plot for one model."""
    predictions = model.predict(X_test)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> Path:
    """Save a grouped bar chart comparing precision, recall, and F1."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "model_comparison_metrics.png"

    plot_df = metrics_df.set_index("model")[["precision", "recall", "f1"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def save_class_distribution_plot(class_counts: pd.Series) -> Path:
    """Save a class distribution plot showing the imbalance."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "class_distribution.png"

    labels = ["Legitimate", "Fraud"]
    ax = class_counts.sort_index().plot(kind="bar", color=["#4C78A8", "#F58518"], figsize=(7, 4))
    ax.set_title("Class Distribution")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def save_feature_importance_plot(model_name: str, feature_importances: pd.DataFrame, top_n: int = 10) -> Path:
    """Save a top-N feature importance bar chart."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{model_name}_feature_importance.png"

    top_features = feature_importances.head(top_n).sort_values("importance")
    ax = top_features.plot(
        kind="barh",
        x="feature",
        y="importance",
        legend=False,
        figsize=(8, 5),
        color="#54A24B",
    )
    ax.set_title(f"{model_name.replace('_', ' ').title()} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path
