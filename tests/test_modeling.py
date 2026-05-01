import pandas as pd
from sklearn.linear_model import LogisticRegression

from models import get_models
from train import save_model_artifact, train_model


def test_model_registry_contains_baseline_and_random_forest():
    models = get_models()

    assert "dummy_majority" in models
    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert models["dummy_majority"].strategy == "most_frequent"
    assert models["logistic_regression"].class_weight == "balanced"
    assert models["decision_tree"].class_weight == "balanced"
    assert models["random_forest"].class_weight == "balanced_subsample"


def test_training_produces_model_artifact(tmp_path, monkeypatch):
    import train

    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    X_train = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_train = pd.Series([0, 0, 1, 1])
    model = LogisticRegression(class_weight="balanced")

    trained_model, model_path = train_model("test_model", model, X_train, y_train, scaler=None)

    assert hasattr(trained_model, "predict")
    assert model_path.exists()


def test_saved_artifact_includes_feature_columns(tmp_path, monkeypatch):
    import joblib
    import train

    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    model = LogisticRegression()
    feature_columns = ["id", "Amount"]

    model_path = save_model_artifact("artifact_test", model, scaler=None, feature_columns=feature_columns)
    artifact = joblib.load(model_path)

    assert artifact["feature_columns"] == feature_columns
    assert artifact["model"] is not None
