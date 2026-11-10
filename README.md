# Credit Card Fraud Detection

A project that builds a machine learning model to flag fraudulent credit card transactions.

Fraud is rare: only about 0.17% of transactions in the dataset are fraudulent, so the main challenge is learning from heavily imbalanced data while keeping **recall** high (catching as many frauds as possible) without flooding the bank with false alarms.

## Dataset

- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle 
- 284,807 transactions, of which 492 are frauds
- Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are the only original features
- Target: `Class` (1 = fraud, 0 = genuine)

## Approach

1. **Exploratory data analysis**: missing values, outliers, class distribution
2. **Preprocessing**: stratified train/test split, `StandardScaler` on `Amount`, power transform to reduce skewness
3. **Models on the imbalanced data**: Logistic Regression, Decision Tree, XGBoost, tuned with `GridSearchCV` and k-fold cross-validation
4. **Balancing the classes**: Random Oversampling, **SMOTE** and **ADASYN**, then the same three models on each
5. **Model selection**: compare ROC-AUC, F1 and recall on the test set, and pick the best probability threshold from the ROC curve

## Results (test set)

| Balancing | Logistic Regression | Decision Tree | XGBoost |
|---|---|---|---|
| None (imbalanced) | ROC 98%, F1 70.3% | ROC 96%, F1 71.9% | ROC 98%, F1 88.3% |
| Random Oversampling | ROC 98%, F1 10.5% | ROC 85%, F1 28.8% | ROC 98%, F1 89.0% |
| SMOTE | ROC 98%, F1 9.8% | ROC 92%, F1 15.6% | ROC 99%, F1 81.0% |
| ADASYN | ROC 98%, F1 3.2% | ROC 95%, F1 7.1% | **ROC 99%, F1 81.2%** |

**Chosen model: XGBoost with ADASYN.** The goal is high **recall**, because a missed fraud costs the bank far more than a false alarm. This model reached a 99% ROC-AUC and 86% recall. Oversampling made Logistic Regression and Decision Trees flag many genuine transactions as fraud (very low F1), while XGBoost stayed strong with every balancing method.

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
jupyter notebook Credit-Card-Fraud-Detection-Capstone-Project.ipynb
```

## Technologies

Python, Jupyter Notebook, NumPy, pandas, Matplotlib, seaborn, scikit-learn, imbalanced-learn, XGBoost
