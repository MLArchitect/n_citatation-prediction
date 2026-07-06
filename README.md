# Citation Year Prediction

Predicting the year of publication of scientific papers based on their metadata (title, abstract, author, editor, publisher).

## Overview

This project tackles the challenge of predicting when a scientific paper was published using only its metadata. The evaluation metric is **Mean Absolute Error (MAE)**.

## Technologies

Python, Scikit-learn, XGBoost, LightGBM, NLTK, CountVectorizer, TF-IDF Vectorizer, NumPy, Pandas, SciPy (sparse matrices), JSON

## Approach

1. **Preprocessing & Feature Engineering** — Cleaned text data using NLTK (lowercasing, tokenization, stemming, stopword removal), filled missing values, removed duplicates, and combined metadata fields into a single feature. Text was vectorized using Scikit-learn's CountVectorizer and TF-IDF Vectorizer, stored in SciPy CSR sparse format.

2. **Model Selection** — Evaluated multiple Scikit-learn regressors: Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, and Gradient Boosting. Lasso Regression was used for feature selection.

3. **Hyperparameter Tuning** — Tuned XGBoost and LightGBM using Scikit-learn's GridSearchCV with 3-fold cross-validation. Best results: XGBoost (500 estimators, max depth 7, learning rate 0.3) MAE 3.417; LightGBM (300 estimators, learning rate 0.2) MAE 3.491.

4. **Ensemble** — Combined XGBoost and LightGBM predictions by averaging, achieving a final **MAE of 3.343**.

## Results

| Model | MAE |
|---|---|
| XGBoost | 3.417 |
| LightGBM | 3.491 |
| **Combined (Ensemble)** | **3.343** |
