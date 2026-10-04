# TASK 2 - PREDICTIVE MODELING USING MACHINE LEARNING
## Loan Default Prediction

### 1. Introduction
The objective is to build a supervised machine-learning classification system that predicts whether a customer is likely to default on a loan.

### 2. Dataset
The project uses 600 synthetic customer records with:
- Age
- Income (thousands)
- Credit Score
- Employment status
- Loan Amount (thousands)
- Debt-to-Income ratio
- Previous Defaults
- Default (target: 0 = No Default, 1 = Default)

Missing values are included intentionally to demonstrate practical data cleaning.

### 3. Methodology
1. Load/create the dataset using Pandas.
2. Inspect missing values.
3. Separate input features and target.
4. Split data into 80% training and 20% testing.
5. Fill numerical missing values with the median.
6. Fill categorical missing values with the most frequent value.
7. Standardize numerical features and one-hot encode the categorical feature.
8. Train Logistic Regression, Decision Tree, and Random Forest models.
9. Evaluate Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
10. Visualize the confusion matrix, ROC curves, and F1-score comparison.

### 4. Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.817 | 0.850 | 0.938 | 0.892 | 0.739 |
| Decision Tree | 0.742 | 0.837 | 0.845 | 0.841 | 0.640 |
| Random Forest | 0.817 | 0.832 | 0.969 | **0.895** | 0.684 |

### 5. Best Model
Random Forest achieved the highest F1 Score of **0.895** in the supplied experiment. F1 Score was used as the main selection metric because it balances precision and recall.

### 6. Confusion Matrix
- True Positive: correctly predicted default.
- True Negative: correctly predicted no default.
- False Positive: predicted default when the customer did not default.
- False Negative: predicted no default when the customer defaulted.

### 7. Expected Outcome
The project demonstrates supervised learning, data preprocessing, model comparison, and model evaluation. The workflow can be adapted to a real dataset after appropriate privacy and domain validation.

### 8. Conclusion
The project successfully demonstrates a complete predictive-modeling workflow. Three classification algorithms were trained and objectively compared. Random Forest produced the highest F1 Score in this experiment. The project also demonstrates missing-value handling, feature preprocessing, confusion-matrix analysis, ROC curves, and quantitative model evaluation.
