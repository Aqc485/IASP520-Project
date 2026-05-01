from pathlib import Path

import pandas as pd

from evaluate import evaluate_model
from plots import save_confusion_matrix_plot


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


def test_evaluate_model_returns_expected_metric_keys():
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 0, 1, 1])
    model = FixedPredictionModel([0, 1, 1, 1])

    metrics = evaluate_model("fixed_model", model, X_test, y_test)

    assert metrics["model"] == "fixed_model"
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 2
    assert metrics["recall"] == 1.0


def test_confusion_matrix_plot_is_saved(tmp_path, monkeypatch):
    import plots

    monkeypatch.setattr(plots, "PLOTS_DIR", tmp_path)
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 0, 1, 1])
    model = FixedPredictionModel([0, 1, 1, 1])

    plot_path = save_confusion_matrix_plot("fixed_model", model, X_test, y_test)

    assert plot_path.exists()
    assert plot_path.suffix == ".png"
    assert Path(plot_path).stat().st_size > 0
