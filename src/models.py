from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> dict:
    """Return the models used in the fraud detection comparison."""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
