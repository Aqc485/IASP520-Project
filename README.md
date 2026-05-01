# IASP520 Project

Machine learning project for detecting fraudulent credit card transactions.

## Project Structure

```text
IASP520Project/
  data/
    creditcard.csv
  src/
    preprocess.py
    models.py
    train.py
    evaluate.py
    plots.py
  tests/
  reports/
  README.md
```

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset. The raw CSV is kept local at:

```text
data/creditcard.csv
```

The dataset file is ignored by Git because it is large. Download it locally before running the code.

## Setup

```powershell
python -m pip install pandas scikit-learn joblib matplotlib pytest
```

## Run

Inspect and summarize the dataset:

```powershell
python src/preprocess.py
```

Train all models:

```powershell
python src/train.py
```

Evaluate the baseline model:
Evaluate all models and save metrics/plots:

```powershell
python src/evaluate.py
```

Generated outputs:

```text
outputs/metrics/model_metrics.csv
outputs/metrics/feature_importance.csv
outputs/plots/logistic_regression_confusion_matrix.png
outputs/plots/decision_tree_confusion_matrix.png
outputs/plots/random_forest_confusion_matrix.png
outputs/plots/model_comparison_metrics.png
outputs/plots/class_distribution.png
outputs/plots/decision_tree_feature_importance.png
outputs/plots/random_forest_feature_importance.png
outputs/models/logistic_regression.joblib
outputs/models/decision_tree.joblib
outputs/models/random_forest.joblib
```

Run automated tests:

```powershell
python -m pytest -q
```
