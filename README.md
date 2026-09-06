# 🌸 Iris Flower Classification & Multi-Model Comparative Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaibhavisnayak/model_for_iris/blob/main/iris_flower_dataset_vaibhavi.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

An end-to-end Machine Learning project analyzing the classic **Iris Flower Dataset**. This study builds, visualizes, evaluates, and compares **six distinct classification algorithms** to identify flower species based on morphological measurements, utilizing multi-metric evaluation and **Occam's Razor** for optimal model selection.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset Details](#-dataset-details)
- [Machine Learning Models](#-machine-learning-models)
- [Evaluation Metrics & Methodology](#-evaluation-metrics--methodology)
- [Experimental Results](#-experimental-results)
- [Key Insights & Model Selection](#-key-insights--model-selection)
- [Repository Structure](#-repository-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [Dependencies](#-dependencies)
- [Author](#-author)

---

## 📖 Overview

The objective of this project is to benchmark classical machine learning classification techniques on multi-class botanical data. The pipeline encompasses:
- Data acquisition and exploratory inspection.
- Stratified 80/20 train-test splitting.
- Training and hyperparameter configuration across 6 diverse supervised learning paradigms.
- Performance profiling with comprehensive metrics: **Accuracy, RMSE, Precision, Recall, F1-Score**.
- Architectural visualization (Decision Tree diagram, Feature Importance ranking, Confusion Matrices).
- Systematic decision-making using the principle of **Occam's Razor** to select the best production-grade model.

---

## 📊 Dataset Details

The dataset used is the well-known **Fisher's Iris Dataset**, loaded directly via `sklearn.datasets.load_iris`:
- **Total Samples:** 150 instances (balanced: 50 per class)
- **Features (4 continuous numerical measurements in cm):**
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`
- **Target Classes (3 species):**
  - `0`: **Iris Setosa**
  - `1`: **Iris Versicolor**
  - `2`: **Iris Virginica**
- **Data Splitting:**
  - Training Set: **120 samples (80%)**
  - Testing Set: **30 samples (20%)**
  - Reproducibility: Seeded with `random_state=42`

---

## 🤖 Machine Learning Models

The repository implements and benchmarks six supervised classification models:

| # | Model | Algorithm Class | Key Configuration |
|---|---|---|---|
| 1 | **K-Nearest Neighbors (KNN)** | Instance-based / Non-parametric | `n_neighbors=3` |
| 2 | **Logistic Regression** | Linear / Generalized Linear Model | `max_iter=200` |
| 3 | **Decision Tree** | Tree-based / Partitioning | `criterion='gini'`, `random_state=42` |
| 4 | **Support Vector Machine (SVM)** | Maximum Margin Classifier | `kernel='linear'` |
| 5 | **Gaussian Naive Bayes** | Probabilistic / Bayesian Classifier | Default Gaussian priors |
| 6 | **Random Forest** | Ensemble / Bagging | `n_estimators=100`, `random_state=42` |

---

## 📐 Evaluation Metrics & Methodology

Each algorithm is subjected to rigorous evaluation using multiple statistical metrics:

- **Accuracy Score:** Percentage of total correct classifications.
- **Root Mean Squared Error (RMSE):** Measure of prediction variance and deviation from actual class indices.
- **Precision (Weighted):** Ratio of correctly predicted positive observations to total predicted positives.
- **Recall (Weighted):** Ratio of correctly predicted positive observations to all observations in actual class.
- **F1-Score (Weighted):** Harmonic mean of Precision and Recall.
- **Confusion Matrix:** Annotated heatmap showing exact true vs. predicted counts per species.
- **Classification Report:** Detailed breakdown of macro/weighted precision, recall, and support.

---

## 📈 Experimental Results

### Final Test Accuracy Report

On the 20% test partition ($N = 30$), all six trained models achieved exceptional classification capability:

| Rank | Model | Accuracy (%) | RMSE | Precision | Recall | F1 Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **K-Nearest Neighbors** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | **Logistic Regression** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | **Random Forest** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | **Support Vector Machine** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | **Decision Tree** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | **Naive Bayes** | **100.00%** | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

### Per-Class Performance Breakdown (Test Set)

```text
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

---

## 💡 Key Insights & Model Selection

1. **Occam's Razor Verdict (Logistic Regression Selected):**
   - When all candidate models achieve an identical 100% test score, the principle of **Occam's Razor** dictates choosing the simplest sufficient hypothesis.
   - **Logistic Regression** is crowned the optimal choice for production: it requires minimal compute, provides transparent probabilistic coefficients, runs with ultra-low latency, and avoids the risk of overfitting present in deep tree ensembles.

2. **Feature Importance (Random Forest Analysis):**
   - Analysis of Gini importance revealed that **`petal length (cm)`** and **`petal width (cm)`** contribute the vast majority of predictive power, clearly separating Iris Setosa from Versicolor and Virginica.
   - `sepal length` and `sepal width` provide supplementary separation boundaries.

3. **Visual Diagnostics:**
   - **Decision Tree Logic:** Rendered via `sklearn.tree.plot_tree` to expose exact threshold splitting rules.
   - **Confusion Matrices:** Confirmed zero false positives and zero false negatives across all target classes.

---

## 📁 Repository Structure

```text
model_for_iris/
├── iris_flower_dataset_vaibhavi.ipynb   # Main Jupyter notebook containing data pipeline, models & plots
├── Untitled0.ipynb                     # Original interactive notebook checkpoint
└── README.md                           # Comprehensive project documentation
```

---

## 🚀 Installation & Getting Started

### 1. Run via Google Colab (Zero Setup)
Click the badge below to run the notebook interactively in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaibhavisnayak/model_for_iris/blob/main/iris_flower_dataset_vaibhavi.ipynb)

### 2. Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vaibhavisnayak/model_for_iris.git
   cd model_for_iris
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn notebook
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook iris_flower_dataset_vaibhavi.ipynb
   ```

---

## 📦 Dependencies

- **Python:** `3.8+`
- **scikit-learn:** Model training, metrics, and dataset loading
- **pandas:** Dataframe manipulation and tabular reporting
- **numpy:** Numerical computations and metric calculations
- **matplotlib:** Plotting decision trees and charts
- **seaborn:** Heatmap confusion matrices and feature importance bar plots

---

## 👤 Author

- **Vaibhavi S Nayak**
- GitHub: [@vaibhavisnayak](https://github.com/vaibhavisnayak)
