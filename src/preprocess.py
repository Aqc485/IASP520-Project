from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/creditcard.csv")
TARGET_COLUMN = "Class"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """Split the dataframe into feature columns and target labels."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Create a stratified train/test split to preserve the fraud ratio."""
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_amount_and_time(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale Amount and time/id columns while leaving PCA-transformed columns unchanged."""
    scaler = StandardScaler()
    columns_to_scale = [column for column in ["Time", "id", "Amount"] if column in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if columns_to_scale:
        X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return basic dataset facts used for exploration and reporting."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df[TARGET_COLUMN].value_counts().to_dict(),
    }


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
    print(describe_dataset(dataset))
