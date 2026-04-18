# Gender Bias and Fairness in Machine Learning-Based Loan Approval

## Overview

This project investigates gender bias in machine learning models used for loan approval prediction. It evaluates both predictive performance and fairness across gender groups.

## Dataset

* 45,000 loan applications (synthetic dataset)
* 13 input features + 1 target variable
* Imbalanced dataset (77.78% rejected, 22.22% approved)
* Link [https://github.com/Layanx1/ML-loan-approval-fairness-project](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

## Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

## Methodology

* Preprocessing using Pipeline and ColumnTransformer
* Feature engineering (4 domain-informed features)
* 80/20 train-test split with stratification
* 5-fold cross-validation

## Evaluation Metrics

* Accuracy, Precision, Recall, F1-score, ROC-AUC
* Fairness metrics: DPD, DIR, EOD

## Results

* Random Forest achieved the best performance (F1: 0.831, ROC-AUC: 0.976)
* Low gender disparity across all models

## Repository Structure

* notebook.ipynb → full implementation
* loan_data.csv → dataset
* GroupF.pdf → final report
