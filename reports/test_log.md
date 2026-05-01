# Test Log

| Test ID | Category | Input/Condition | Expected Outcome | Actual Outcome | Pass/Fail | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| T001 | Data validation | Dataset file exists at `data/creditcard.csv` | File exists and can be read by the project | File exists at the expected path | Pass | `tests/test_data_validation.py::test_dataset_file_exists` |
| T002 | Data validation | Dataset schema is checked | Dataset contains `Time`, `Amount`, `Class`, and `V1` through `V28` | All expected columns were present; dataset has 31 columns | Pass | `tests/test_data_validation.py::test_dataset_has_expected_schema` |
| T003 | Data validation | Missing values are measured | Missing value count is available and equals 0 | Dataset summary reported 0 missing values | Pass | `tests/test_data_validation.py::test_dataset_has_no_missing_values` |
| T004 | Data validation | Target class values are checked | Dataset contains both legitimate and fraud labels | `Class` contains both `0` and `1`; fraud rows are present | Pass | `tests/test_data_validation.py::test_dataset_contains_both_classes` |
| T005 | Data validation | Duplicate rows are measured | Duplicate count is calculated and documented | Duplicate count is measured instead of assumed clean | Pass | `tests/test_data_validation.py::test_dataset_duplicate_rows_are_measured` |
| T006 | Preprocessing | Split features and target | `Class` is removed from feature matrix and returned as target | Feature matrix excludes `Class`; target length matches features | Pass | `tests/test_preprocess.py::test_split_features_target_removes_target_column` |
| T007 | Preprocessing | Stratified train/test split | Both train and test labels preserve both classes | Train and test sets each contain labels `0` and `1` | Pass | `tests/test_preprocess.py::test_train_test_split_is_stratified` |
| T008 | Preprocessing | Scale `Time` and `Amount` | Scaling keeps the same feature columns | Scaled train/test outputs preserved feature column order | Pass | `tests/test_preprocess.py::test_scaling_preserves_feature_columns` |
| T009 | Preprocessing | Prevent train/test leakage during scaling | Scaler fits on training data only | Training `Amount` mean is centered; test mean differs | Pass | `tests/test_preprocess.py::test_scaler_fits_only_training_data` |
| T010 | Model test | Model registry is inspected | Baseline and improved models are available | Dummy Majority, Logistic Regression, Decision Tree, and Random Forest are registered | Pass | `tests/test_modeling.py::test_model_registry_contains_baseline_and_random_forest` |
| T011 | Model test | Train a small model fixture | Training returns a model and saves an artifact | Test model artifact was written to a temporary path | Pass | `tests/test_modeling.py::test_training_produces_model_artifact` |
| T012 | Model test | Save model artifact metadata | Artifact includes feature column names | Saved artifact included expected feature columns | Pass | `tests/test_modeling.py::test_saved_artifact_includes_feature_columns` |
| T013 | Evaluation test | Evaluate fixed predictions | Metrics include confusion-matrix counts and recall | Returned TN, FP, FN, TP, and recall values matched expected results | Pass | `tests/test_evaluation.py::test_evaluate_model_returns_expected_metric_keys` |
| T014 | Evaluation test | Save confusion matrix plot | Plot file is created and non-empty | PNG plot was written to a temporary path | Pass | `tests/test_evaluation.py::test_confusion_matrix_plot_is_saved` |

Final automated test command:

```powershell
& "C:\Users\Aaron\AppData\Local\Programs\Python\Python311\python.exe" -m pytest -q
```

Final result:

```text
14 passed in 5.53s
```
