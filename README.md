# Task 2 - Predictive Modeling Using Machine Learning

## Loan Default Prediction

This project demonstrates a complete supervised-learning workflow for predicting loan default.

### Models
- Logistic Regression
- Decision Tree
- Random Forest

### Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC curves

### How to run

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Then run:

```bash
python predictive_modeling.py
```

The program automatically creates the reproducible synthetic dataset if
`loan_default_dataset.csv` is not present. It then creates:
- `loan_default_dataset.csv`
- `model_results.csv`
- `confusion_matrix.png`
- `roc_curves.png`
- `model_comparison.png`

The supplied experiment selected **Random Forest** as the best model by F1 Score (0.895).

See [Task_2_Report.md](Task_2_Report.md) for the project report.
