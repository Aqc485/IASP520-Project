from pathlib import Path

import pandas as pd

from preprocess import DATA_PATH, TARGET_COLUMN, describe_dataset, load_dataset


def test_dataset_file_exists():
    assert DATA_PATH.exists()


def test_dataset_has_expected_schema():
    df = load_dataset()
    expected_columns = {"id", "Amount", TARGET_COLUMN}
    expected_columns.update({f"V{i}" for i in range(1, 29)})

    assert set(df.columns) == expected_columns
    assert df.shape[1] == 31


def test_dataset_has_no_missing_values():
    df = load_dataset()
    summary = describe_dataset(df)

    assert summary["missing_values"] == 0


def test_dataset_contains_both_classes():
    df = load_dataset()
    class_values = set(df[TARGET_COLUMN].unique())

    assert class_values == {0, 1}
    assert df[TARGET_COLUMN].value_counts()[1] > 0


def test_dataset_duplicate_rows_are_measured():
    df = load_dataset()
    summary = describe_dataset(df)

    assert summary["duplicate_rows"] == int(df.duplicated().sum())
    assert summary["duplicate_rows"] >= 0
