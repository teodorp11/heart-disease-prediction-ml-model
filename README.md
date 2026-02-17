# Heart Disease Prediction using Logistic Regression

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository contains a professional Machine Learning pipeline designed to predict the **10-year risk of future coronary heart disease (CHD)**. Using the **Framingham Heart Study** dataset, the project implements a Logistic Regression classifier to estimate the probability of cardiovascular events based on clinical and behavioral patient attributes.

By leveraging modular Python programming, this project demonstrates a complete data science workflow: from raw data cleaning and feature engineering to model evaluation and visualization.

## Repository Structure

```text
heart-disease-prediction-ml-model/
├── data/               # Raw dataset (framingham.csv)
├── src/                # Core Package Logic
│   ├── __init__.py     # Package initializer
│   ├── preprocess.py   # Data cleaning & StandardScaler logic
│   ├── model.py        # Logistic Regression training & evaluation
│   └── visualize.py    # Result visualization (Confusion Matrix Heatmaps)
├── main.py             # Execution script
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Methodology & Features

### Data Dictionary

The model utilizes a specific subset of clinical and behavioral predictors based on the **Framingham Heart Study** parameters.

| Category         | Feature      | Description                           |
| :--------------- | :----------- | :------------------------------------ |
| **Demographics** | `Age`        | Patient age in years                  |
|                  | `male`       | Binary indicator (1: Male, 0: Female) |
| **Behavioral**   | `cigsPerDay` | Average number of cigarettes per day  |
| **Clinical**     | `totChol`    | Total cholesterol level               |
|                  | `sysBP`      | Systolic blood pressure               |
|                  | `glucose`    | Blood glucose level                   |

---

### Data Processing Pipeline

To ensure the model's reliability and convergence, the following transformation steps were implemented:

1. **Feature Engineering & Cleaning**
   - **Dimensionality Reduction:** Dropped the `education` column as it showed low predictive correlation with CHD risk.
   - **Handling Missing Data:** Employed _Listwise Deletion_ (`dropna`) to maintain the integrity of the clinical records.
2. **Standardization**

   > **Note:** Logistic Regression is sensitive to the scale of input features.
   - Applied `StandardScaler` to transform features to have a mean of $0$ and a standard deviation of $1$. This ensures that high-magnitude features (like `totChol`) do not overpower smaller ones (like `male`).

3. **Validation Strategy**
   - **Split Ratio:** 70% Training / 30% Testing.
   - **Reproducibility:** Used a fixed `random_state=4` to ensure consistent results across different execution environments.

---

## Installation & Usage

### 1. Environment Setup

It is recommended to use a virtual environment to isolate project dependencies and prevent version conflicts.

**Windows:**

```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
```

### 2. Install Dependecies

```bash
pip install -r requirements.txt
```

### 3. Run the Model

Ensure framingham.csv is located in the data/ directory, then execute:

```bash
python main.py
```

### Performance Results

The model performance was evaluated using a standard train-test split. While the accuracy appears high, the class-specific metrics reveal a significant bias toward the majority class (No CHD).

## Key Metrics

| Metric                  | Value    |
| :---------------------- | :------- |
| **Overall Accuracy**    | `84.90%` |
| **Precision (Class 0)** | `0.85`   |
| **Recall (Class 1)**    | `0.08`   |

## Confusion Matrix

|                 | Predicted Negative | Predicted Positive |
| :-------------- | :----------------- | :----------------- |
| Actual Negative | 942 (TN)           | 9 (FP)             |
| Actual Positive | 161 (FN)           | 14 (TP)            |

[!IMPORTANT]

Analytical Note: The high accuracy is contrasted by a critically low recall for CHD-positive cases. In a medical context, this indicates a high rate of False Negatives, meaning the model frequently fails to identify patients with heart disease. This is a common challenge in imbalanced medical datasets.

### References

Dataset: [Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

Article: [Geeks for Geeks Heart Disease Prediction using Logistic Regression](https://www.geeksforgeeks.org/machine-learning/ml-heart-disease-prediction-using-logistic-regression/)
