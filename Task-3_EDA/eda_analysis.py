"""
Task 3 - Exploratory Data Analysis (EDA)
Employee Performance and Workforce Analysis

Run:
    python eda_analysis.py

The script creates/loads a dataset, performs data cleaning,
generates statistical summaries, analyzes correlations and groups,
and saves six visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_STATE = 42
DATA_FILE = Path("employee_eda_dataset.csv")

def create_dataset(n=600):
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.integers(21, 61, n)
    experience = np.clip(age - rng.integers(20, 30, n), 0, 40)
    department = rng.choice(["IT", "Finance", "HR", "Marketing", "Operations"], n,
                             p=[0.30, 0.18, 0.12, 0.16, 0.24])
    education = rng.choice(["Bachelor", "Master", "PhD"], n, p=[0.55, 0.38, 0.07])
    monthly_income = np.clip(
        18000 + experience * 2500 + (education == "Master") * 8000
        + (education == "PhD") * 18000 + rng.normal(0, 9000, n),
        18000, 180000).round(0)
    job_satisfaction = np.clip(
        3.0 + (department == "IT") * 0.25 + (department == "Finance") * 0.10
        + rng.normal(0, 0.8, n), 1, 5).round(1)
    work_hours = np.clip(
        40 + (department == "IT") * 2 + (department == "Operations") * 3
        + rng.normal(0, 5, n), 25, 65).round(1)
    performance = np.clip(
        55 + experience * 0.45 + job_satisfaction * 4
        - np.maximum(work_hours - 50, 0) * 0.6
        + (education == "Master") * 2 + (education == "PhD") * 4
        + rng.normal(0, 8, n), 30, 100).round(1)
    annual_leave = np.clip(
        12 + job_satisfaction * 1.2 + rng.normal(0, 4, n), 5, 30).round(1)
    attrition_prob = np.clip(
        0.18 + np.maximum(work_hours - 45, 0) * 0.025
        - (job_satisfaction - 3) * 0.12 - experience * 0.008, 0.03, 0.65)
    attrition = rng.binomial(1, attrition_prob)
    data = pd.DataFrame({
        "Age": age, "Experience_Years": experience, "Department": department,
        "Education": education, "Monthly_Income": monthly_income,
        "Job_Satisfaction": job_satisfaction,
        "Work_Hours_Per_Week": work_hours,
        "Performance_Score": performance,
        "Annual_Leave_Days": annual_leave,
        "Attrition": np.where(attrition == 1, "Yes", "No")
    })
    for col, count in [("Monthly_Income", 10), ("Job_Satisfaction", 8),
                       ("Work_Hours_Per_Week", 7)]:
        idx = rng.choice(data.index, count, replace=False)
        data.loc[idx, col] = np.nan
    return pd.concat([data, data.iloc[:5]], ignore_index=True)

if not DATA_FILE.exists():
    df = create_dataset()
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

print("Original shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

clean = df.drop_duplicates().copy()
numeric_cols = ["Age", "Experience_Years", "Monthly_Income",
                "Job_Satisfaction", "Work_Hours_Per_Week",
                "Performance_Score", "Annual_Leave_Days"]
for col in numeric_cols:
    clean[col] = clean[col].fillna(clean[col].median())
clean["Attrition_Flag"] = clean["Attrition"].map({"No": 0, "Yes": 1})

summary = clean[numeric_cols].describe().T.round(2)
summary.to_csv("statistical_summary.csv")
print("\nSTATISTICAL SUMMARY")
print(summary)

corr = clean[numeric_cols + ["Attrition_Flag"]].corr().round(3)
corr.to_csv("correlation_matrix.csv")
print("\nCORRELATION MATRIX")
print(corr)

dept = clean.groupby("Department").agg(
    Employees=("Department", "size"),
    Avg_Income=("Monthly_Income", "mean"),
    Avg_Performance=("Performance_Score", "mean"),
    Avg_Satisfaction=("Job_Satisfaction", "mean"),
    Attrition_Rate=("Attrition_Flag", "mean")).round(2)
dept["Attrition_Rate"] = (dept["Attrition_Rate"] * 100).round(2)
dept.to_csv("department_analysis.csv")
print("\nDEPARTMENT ANALYSIS")
print(dept)

edu = clean.groupby("Education").agg(
    Employees=("Education", "size"),
    Avg_Income=("Monthly_Income", "mean"),
    Avg_Performance=("Performance_Score", "mean")).round(2)
edu.to_csv("education_analysis.csv")
print("\nEDUCATION ANALYSIS")
print(edu)

plt.figure(figsize=(8,5))
plt.hist(clean["Monthly_Income"], bins=25)
plt.title("Distribution of Monthly Income")
plt.xlabel("Monthly Income"); plt.ylabel("Number of Employees")
plt.tight_layout(); plt.savefig("01_income_distribution.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
clean.groupby("Department")["Performance_Score"].mean().sort_values().plot(kind="bar")
plt.title("Average Performance Score by Department")
plt.xlabel("Department"); plt.ylabel("Average Performance Score")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("02_department_performance.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
clean.groupby("Education")["Monthly_Income"].mean().sort_values().plot(kind="bar")
plt.title("Average Monthly Income by Education")
plt.xlabel("Education"); plt.ylabel("Average Monthly Income")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("03_education_income.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
clean.groupby("Department")["Attrition_Flag"].mean().mul(100).sort_values().plot(kind="bar")
plt.title("Attrition Rate by Department")
plt.xlabel("Department"); plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("04_department_attrition.png", dpi=180); plt.close()

plt.figure(figsize=(8,5))
plt.scatter(clean["Job_Satisfaction"], clean["Performance_Score"], alpha=0.5)
plt.title("Job Satisfaction vs Performance Score")
plt.xlabel("Job Satisfaction"); plt.ylabel("Performance Score")
plt.tight_layout(); plt.savefig("05_satisfaction_performance.png", dpi=180); plt.close()

plt.figure(figsize=(9,7))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Matrix")
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(label="Correlation")
plt.tight_layout(); plt.savefig("06_correlation_matrix.png", dpi=180); plt.close()

print("\nEDA completed successfully.")
