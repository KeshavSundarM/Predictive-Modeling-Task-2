# TASK 3 – EXPLORATORY DATA ANALYSIS (EDA) PROJECT

## Employee Performance and Workforce Analysis

### 1. Introduction
The purpose of this project is to perform Exploratory Data Analysis (EDA) on an employee dataset to discover patterns, relationships, and factors that may influence employee performance and attrition.

### 2. Dataset
The dataset contains Age, Experience, Department, Education, Monthly Income, Job Satisfaction, Work Hours per Week, Performance Score, Annual Leave Days, and Attrition. The raw dataset contains 605 rows including duplicate records and missing values. After cleaning, 600 unique records are analyzed.

### 3. Data Cleaning
- Load data using Pandas.
- Check missing values and duplicate rows.
- Remove duplicates.
- Fill missing numerical values using the median.
- Convert Attrition to a numeric flag for correlation analysis.

### 4. Statistical and Group Analysis
The project produces descriptive statistics and department/education summaries. In the supplied experiment, IT has the highest average performance and Marketing has the highest attrition rate. PhD has the highest average income among education groups.

### 5. Correlation Analysis
Key correlations in the supplied experiment:
- Monthly Income vs Performance Score: 0.514
- Job Satisfaction vs Performance Score: 0.346
- Work Hours vs Attrition: 0.072

Correlation indicates association, not causation.

### 6. Key Findings
1. Employee performance varies across departments.
2. Education level is associated with differences in average income.
3. Job satisfaction has a measurable relationship with performance.
4. Work hours show a small positive association with attrition in this sample.
5. Income, satisfaction, experience, and working hours should be considered together when studying workforce outcomes.
6. Overall attrition is approximately 9.83% in the analyzed sample.

### 7. Visualizations
The Python program generates:
- Income distribution
- Average performance by department
- Average income by education
- Attrition rate by department
- Job satisfaction vs performance
- Correlation matrix

### 8. Conclusion
The EDA identifies useful patterns and relationships in the employee dataset and provides a foundation for future predictive modeling such as attrition or performance prediction. These exploratory findings should be validated with larger, domain-verified data before real organizational decisions are made.
