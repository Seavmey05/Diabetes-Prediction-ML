# Diabetes Prediction Project

An end-to-end Machine Learning classification project developed for the **Machine Learning** course at **Phnom Penh International University (PPIU)**.

---

## 📌 Project Overview

The objective of this project is to build and evaluate machine learning models capable of predicting whether a patient is Non-Diabetic, Prediabetic, or Diabetic based on clinical and medical data. The study compares two primary supervised learning algorithms: **Logistic Regression** and **Random Forest**.

* **Institution:** Phnom Penh International University (PPIU)
* **Course:** Machine Learning
* **Lecturer:** Him Soklong
* **Group 2 Members:** 
  * Ly Seavmey
  * Chamroeun Vichita

---

## 📊 Dataset Summary

* **Source:** Kaggle (Diabetes Prediction Dataset)
* **Dataset Size:** 1,000 samples / rows, 14 features
* **Problem Type:** Multi-class Classification

### Target Variable (`CLASS`)
| Label / Code | Category | Distribution Count | Percentage |
| :--- | :--- | :--- | :--- |
| **`0` / `N`** | Non-Diabetic | 103 | 10.3% |
| **`1` / `P`** | Pre-Diabetic | 53 | 5.3% |
| **`2` / `Y`** | Diabetic | 844 | 84.4% |

### Feature Descriptions
| Feature Column | Data Type | Description |
| :--- | :--- | :--- |
| `ID` | `int64` | Unique patient identifier (dropped during modeling) |
| `No_Pation` | `int64` | Patient record number (dropped during modeling) |
| `Gender` | `object` | Gender of the patient (Male / Female) |
| `AGE` | `int64` | Patient age in years |
| `Urea` | `float64` | Blood urea level |
| `Cr` | `int64` | Creatinine level |
| `HbA1c` | `float64` | Average blood glucose level over past 2–3 months (%) |
| `Chol` | `float64` | Total Cholesterol level |
| `TG` | `float64` | Triglycerides level |
| `HDL` | `float64` | High-Density Lipoprotein ("good" cholesterol) |
| `LDL` | `float64` | Low-Density Lipoprotein ("bad" cholesterol) |
| `VLDL` | `float64` | Very Low-Density Lipoprotein |
| `BMI` | `float64` | Body Mass Index (kg/m²) |
| `CLASS` | `object` | Target variable: `0` / `N`, `1` / `P`, `2` / `Y` |

---

## 🔍 Key Findings from Exploratory Data Analysis (EDA)

1. **HbA1c Level Ranges by Class:**
   * **Class 0 (Non-Diabetic):** 4.0 – 5.2
   * **Class 1 (Pre-Diabetic):** 5.7 – 6.3
   * **Class 2 (Diabetic):** 7.0 – 10.5 (up to 16.0)
2. **Age Distribution:** Concentrated heavily around 50–60 years (peak near 53 years); very few patients under 30 or over 70.
3. **Top Correlations with Target Class:**
   * **BMI:** $0.57$
   * **HbA1c:** $0.56$
   * **AGE:** $0.44$

---

## 🔄 Project Workflow

```
1. Load Dataset ➡️ 2. Data Cleaning & EDA ➡️ 3. Preprocessing (StandardScaler, Encoding)
                                                                   ⬇️
6. Model Deployment ⬅️ 5. Model Evaluation (Metrics) ⬅️ 4. Hyperparameter Tuning (GridSearch)
```

1. **Data Cleaning & EDA:** Removed trailing spaces from categorical strings (`Gender`, `CLASS`), analyzed distributions and correlations via plots/heatmaps.
2. **Preprocessing:** One-hot/label encoding, feature scaling (`StandardScaler` applied specifically for Logistic Regression), dropped non-predictive identifier columns (`ID`, `No_Pation`), split data into **80% Train / 20% Test**.
3. **Model Training & Tuning:** Hyperparameter optimization via `GridSearchCV` on Logistic Regression and Random Forest.
4. **Evaluation:** Assessed performance using Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.

---

## 📈 Feature Importance (Random Forest)

1. **HbA1c** (Highest influence)
2. **BMI**
3. **AGE**
4. **Chol**
5. **TG**
6. **VLDL**
7. **LDL**
8. **Urea**
9. **Cr**
10. **HDL**
11. **Gender** (Lowest influence)

---

## 📊 Comparative Performance Analysis

### Evaluation Metrics (Test Set - 200 samples)
| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 94.50% | **99.50%** |
| **Precision** | 93.52% | **99.50%** |
| **Recall** | 94.50% | **99.50%** |
| **F1-Score** | 93.35% | **99.49%** |

### Detailed Class Breakdown & Misclassifications
| Class | True Label | Logistic Regression (Correct) | Random Forest (Correct) |
| :--- | :--- | :--- | :--- |
| **Class 0** | Non-Diabetic | 19 / 21 | **21 / 21** |
| **Class 1** | Pre-Diabetic | 2 / 10 | **9 / 10** |
| **Class 2** | Diabetic | 168 / 169 | **169 / 169** |
| **Total Misclassifications** | — | **11** | **1** |

> **Key Observation:** Logistic Regression struggled significantly with Class 1 (Pre-Diabetic), misclassifying 8 out of 10 cases into Diabetic/Non-Diabetic. Random Forest handled the multi-class distinction seamlessly, misclassifying only 1 Pre-Diabetic patient as Diabetic.

---

## ⚠️ Challenges & Lessons Learned

* **Data Cleaning:** Categorical columns contained trailing whitespace strings which caused encoding issues prior to normalization.
* **Preprocessing Rules:** Learned the importance of model-specific preprocessing (e.g., applying `StandardScaler` to Logistic Regression vs. leaving decision tree ensembles on unscaled numerical ranges).
* **Class Imbalance:** Handled the high proportion of Class 2 (84.4%) through appropriate evaluation metrics beyond pure accuracy.

---

## 🏁 Conclusion

* **Random Forest** is the recommended model for this diabetes prediction task, achieving superior performance with **99.50% accuracy** compared to **94.50% for Logistic Regression**.
* The project successfully demonstrates how machine learning algorithms can analyze clinical metrics (especially HbA1c, BMI, and Age) to assist medical professionals in early diabetes diagnosis.
# To Run the Demo 
* Python 3.12.4
* python -m pip install fastapi uvicorn streamlit scikit-learn joblib requests
* python -m uvicorn api:app --reload
* python -m streamlit run app.py
# Technologies Used

- Google Collab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


