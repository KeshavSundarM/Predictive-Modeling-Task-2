"""
Task 2 - Predictive Modeling Using Machine Learning
Loan Default Prediction

Run:
    python predictive_modeling.py

The script creates a reproducible sample dataset if the CSV is not present,
cleans missing values, trains three classification models, compares them,
and saves evaluation results and charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

RANDOM_STATE = 42
DATA_FILE = Path("loan_default_dataset.csv")


def create_dataset(n=600):
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(21, 66, n)
    income = rng.integers(25, 151, n)
    credit = rng.integers(500, 851, n)
    employed = rng.choice(["Yes", "No"], n, p=[0.78, 0.22])
    loan_amount = rng.integers(5, 101, n)
    dti = np.round(rng.uniform(5, 55, n), 1)
    previous_defaults = rng.poisson(0.35, n).clip(0, 3)

    risk = (
        1.7
        - 0.010 * (credit - 650)
        + 0.020 * (dti - 30)
        + 0.012 * (loan_amount - 50)
        - 0.008 * (income - 80)
        + 0.65 * previous_defaults
        + np.where(employed == "No", 0.8, 0)
        + np.where(age < 25, 0.35, 0)
        + rng.normal(0, 0.55, n)
    )
    probability = 1 / (1 + np.exp(-risk))
    default = rng.binomial(1, probability)

    data = pd.DataFrame({
        "Age": age,
        "Income_K": income,
        "Credit_Score": credit,
        "Employed": employed,
        "Loan_Amount_K": loan_amount,
        "Debt_to_Income": dti,
        "Previous_Defaults": previous_defaults,
        "Default": default
    })

    # Add missing values to demonstrate data cleaning.
    missing_indices = rng.choice(n, size=18, replace=False)
    data.loc[missing_indices[:6], "Income_K"] = np.nan
    data.loc[missing_indices[6:12], "Debt_to_Income"] = np.nan
    data.loc[missing_indices[12:], "Employed"] = np.nan
    return data


# 1. Load or create the dataset
if not DATA_FILE.exists():
    df = create_dataset()
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Separate features and target
X = df.drop(columns="Default")
y = df["Default"]

numeric_features = [
    "Age", "Income_K", "Credit_Score",
    "Loan_Amount_K", "Debt_to_Income", "Previous_Defaults"
]
categorical_features = ["Employed"]

# 3. Data preprocessing
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# 5. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    )
}

results = []
trained_models = {}

# 6. Training and evaluation
for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probability)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1 Score": f1_score(y_test, prediction, zero_division=0),
        "ROC AUC": auc(fpr, tpr)
    })
    trained_models[name] = (pipeline, prediction, probability)

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# 7. Best model by F1 score
best_name = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
best_pipeline, best_prediction, best_probability = trained_models[best_name]

print("\nBEST MODEL:", best_name)
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, best_prediction,
    target_names=["No Default", "Default"],
    zero_division=0
))

# 8. Confusion matrix
cm = confusion_matrix(y_test, best_prediction)
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
for (i, j), value in np.ndenumerate(cm):
    plt.text(j, i, str(value), ha="center", va="center")
plt.xticks([0, 1], ["No Default", "Default"])
plt.yticks([0, 1], ["No Default", "Default"])
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=180)
plt.close()

# 9. ROC curves
plt.figure(figsize=(7, 5))
for name, (_, _, probability) in trained_models.items():
    fpr, tpr, _ = roc_curve(y_test, probability)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=180)
plt.close()

# 10. F1 comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1 Score"])
plt.ylabel("F1 Score")
plt.title("Model Comparison - F1 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=180)
plt.close()

print("\nProject completed successfully.")
