# Diabetes-Prediction-ML

## Project Overview

This project was developed as part of a Machine Learning course to predict diabetes using supervised machine learning algorithms. The objective is to compare the performance of **Logistic Regression** and **Random Forest** in predicting diabetes based on patients' medical information.

---

## Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early detection can improve treatment outcomes and reduce complications. This project aims to develop a machine learning model that predicts whether a patient is **Non-Diabetic, Prediabetic, or Diabetic** based on medical measurements.

**Problem Type:** Classification

---

## Dataset

**Dataset Name:** Diabetes Prediction Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset/data

### Dataset Information

- **Total Records:** 1000
- **Total Features:** 14
- **Missing Values:** None
- **Target Variable:** CLASS

### Features

| Feature | Description |
|----------|-------------|
| Gender | Patient gender |
| AGE | Patient age |
| Urea | Blood urea level |
| Cr | Creatinine level |
| HbA1c | Hemoglobin A1c |
| Chol | Cholesterol |
| TG | Triglycerides |
| HDL | High-density lipoprotein |
| LDL | Low-density lipoprotein |
| VLDL | Very low-density lipoprotein |
| BMI | Body Mass Index |
| CLASS | Diabetes classification |

**Identifier columns removed before training:**

- ID
- No_Pation

---

# Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset overview (`df.info()`)
- Class distribution
- Correlation heatmap
- Histograms

### Key Findings

- The dataset contains **1000 patient records**.
- No missing values were found.
- Gender and CLASS were categorical variables requiring encoding.
- ID and No_Pation were removed because they are identifier columns and do not contribute to prediction.

---

# Data Cleaning

The following steps were applied before model training:

- Removed columns (ID and No_Pation)
- Encoded categorical variables using LabelEncoder
- Standardized numerical features for Logistic Regression using StandardScaler
- Split the dataset into:
  - 80% Training Set
  - 20% Testing Set

---

# Machine Learning Models

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

### Hyperparameter Tuning

GridSearchCV was used to optimize:

- C
- Solver
- Maximum Iterations

---

## 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

### Hyperparameter Tuning

GridSearchCV was used to optimize:

- Number of Trees (n_estimators)
- Maximum Depth

# Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Performance Comparison

| Metric | Logistic Regression | Random Forest |
|---------|--------------------:|--------------:|
| Accuracy | **94.50%** | **99.50%** |
| Precision (Weighted Avg.) | 0.95 | **1.00** |
| Recall (Weighted Avg.) | 0.94 | **0.99** |
| F1-Score (Weighted Avg.) | 0.95 | **0.99** |
| Misclassifications | 11 | **1** |

### Model Comparison


# Results

The experimental results show that **Random Forest outperformed Logistic Regression**.

### Logistic Regression

- Accuracy: **94.50%**
- Misclassifications: **11**

### Random Forest

- Accuracy: **99.50%**
- Misclassifications: **1**

Random Forest achieved better Precision, Recall, and F1-score across all classes, particularly for the **Prediabetic** class, making it the best-performing model for this dataset.

---

# Technologies Used

- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


