import pandas as pd

from preprocess import create_train_test_split, scale_amount_and_time, split_features_target


def sample_dataframe():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "V1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Amount": [10, 20, 30, 40, 50, 60],
            "Class": [0, 0, 0, 1, 1, 1],
        }
    )


def test_split_features_target_removes_target_column():
    X, y = split_features_target(sample_dataframe())

    assert "Class" not in X.columns
    assert y.name == "Class"
    assert len(X) == len(y)


def test_train_test_split_is_stratified():
    X_train, X_test, y_train, y_test = create_train_test_split(
        sample_dataframe(),
        test_size=0.5,
        random_state=42,
    )

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_scaling_preserves_feature_columns():
    df = sample_dataframe()
    X, _ = split_features_target(df)
    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    X_train_scaled, X_test_scaled, _ = scale_amount_and_time(X_train, X_test)

    assert list(X_train_scaled.columns) == list(X_train.columns)
    assert list(X_test_scaled.columns) == list(X_test.columns)


def test_scaler_fits_only_training_data():
    df = sample_dataframe()
    X, _ = split_features_target(df)
    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    X_train_scaled, X_test_scaled, _ = scale_amount_and_time(X_train, X_test)

    assert round(float(X_train_scaled["Amount"].mean()), 10) == 0
    assert float(X_test_scaled["Amount"].mean()) != 0
