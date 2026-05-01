# IASP520 Project: Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using a reproducible Python pipeline.

## Project Goal

The goal of this project is to classify credit card transactions as either legitimate or fraudulent. The target variable is `Class`, where:

- `0` = legitimate transaction
- `1` = fraudulent transaction

Because fraudulent transactions are rare, this project compares a simple majority-class baseline against machine learning models using fraud-focused metrics such as precision, recall, F1-score, and confusion matrix results.

## Repository Structure

```text
IASP520 Project/
  data/
    creditcard.csv
  src/
    preprocess.py
    models.py
    train.py
    evaluate.py
    plots.py
  tests/
    conftest.py
    test_data_validation.py
    test_preprocess.py
    test_modeling.py
    test_evaluation.py
  outputs/
    metrics/
    plots/
    models/
  README.md
```

## Dataset

The main dataset is the Kaggle Credit Card Fraud Detection dataset.

Place the main dataset here:

```text
data/creditcard.csv
```

The raw datasets are ignored by Git because they are large. They must be downloaded locally before running the pipeline.

## Dependencies

Install dependencies with:

```powershell
python -m pip install pandas scikit-learn joblib matplotlib pytest
```

If VS Code is using a specific Python 3.11 interpreter, use:

```powershell
py -3.11 -m pip install pandas scikit-learn joblib matplotlib pytest
```

## How to Run

Inspect and summarize the dataset:

```powershell
python src/preprocess.py
```

Train all models:

```powershell
python src/train.py
```

Train models, evaluate them, save metrics, and save plots:

```powershell
python src/evaluate.py
```

If `python` is not recognized or uses the wrong environment, run the same commands with:

```powershell
py -3.11 src/evaluate.py
```

## Models

The pipeline compares:

- Dummy Classifier baseline: predicts the majority class
- Logistic Regression: simple machine learning baseline
- Decision Tree: interpretable nonlinear model
- Random Forest: improved ensemble model

Class imbalance is handled with class weights:

- Logistic Regression: `class_weight="balanced"`
- Decision Tree: `class_weight="balanced"`
- Random Forest: `class_weight="balanced_subsample"`

## Reproducing Results

1. Place `creditcard.csv` in the `data/` folder.
2. Install dependencies.
3. Run:

```powershell
python src/evaluate.py
```

4. Review generated metrics and plots in `outputs/`.

Expected main result pattern:

```text
Dummy Classifier F1:  0.000
Logistic Regression F1: 0.114
Decision Tree F1:       0.128
Random Forest F1:       0.829
```

Random Forest is the strongest model so far based on F1-score.

## Generated Outputs

Metrics:

```text
outputs/metrics/model_metrics.csv
outputs/metrics/feature_importance.csv
```

Plots:

```text
outputs/plots/class_distribution.png
outputs/plots/model_comparison_metrics.png
outputs/plots/dummy_majority_confusion_matrix.png
outputs/plots/logistic_regression_confusion_matrix.png
outputs/plots/decision_tree_confusion_matrix.png
outputs/plots/random_forest_confusion_matrix.png
outputs/plots/decision_tree_feature_importance.png
outputs/plots/random_forest_feature_importance.png
```

Model artifacts:

```text
outputs/models/dummy_majority.joblib
outputs/models/logistic_regression.joblib
outputs/models/decision_tree.joblib
outputs/models/random_forest.joblib
```

Note: generated outputs may be ignored by Git depending on submission size requirements. They can be reproduced by running `python src/evaluate.py`.

## Tests

Run the automated test suite:

```powershell
python -m pytest -q
```

Current test coverage includes:

- 5 data validation tests
- 4 preprocessing tests
- 3 model tests
- 2 evaluation tests

Latest passing result:

```text
14 passed
```
