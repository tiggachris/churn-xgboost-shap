# Decoding Customer Churn: An Explainable AI Approach Using XGBoost and SHAP

This repository contains the complete machine learning pipeline and research artifacts for a data-driven study on **customer churn prediction in the telecommunications industry**.

The project investigates whether customer demographic, subscription, billing, tenure, and service-related characteristics contain identifiable patterns associated with customer churn.

The study compares multiple machine learning approaches and applies **Explainable AI (XAI)** using **SHAP (SHapley Additive exPlanations)** to understand which features contribute to churn predictions.

---

## 📌 Project Overview

Customer churn is an important problem for subscription-based businesses. Identifying customers who are likely to discontinue their services can help organizations understand customer behavior and develop appropriate retention strategies.

This project uses the **Maven Communications Churn Dataset** to investigate the relationship between customer characteristics and churn.

The workflow includes:

- Data exploration
- Data preprocessing
- Feature engineering
- Classification model training
- Model comparison
- Churn-focused evaluation
- XGBoost-based prediction
- SHAP-based model interpretation

The primary focus is not only on predicting churn, but also on understanding **why the model predicts that a customer is likely to churn**.

---

# 🎯 Research Objective

The main objective of this study is:

> **To investigate the effectiveness of machine learning techniques for predicting customer churn using demographic, subscription, billing, tenure, and service-related customer information, while using Explainable AI to identify the factors contributing to churn predictions.**

The study investigates the following questions:

1. Can customer characteristics be used to predict customer churn?
2. How do different machine learning algorithms perform on the same churn dataset?
3. How does class imbalance affect churn prediction?
4. Which customer features contribute most strongly to churn predictions?
5. Can SHAP provide interpretable insights into the model's predictions?

---

# 📊 Dataset

## Maven Communications Churn Dataset

The project uses the **Maven Communications Churn Dataset**, a telecommunications customer churn dataset containing customer-level demographic, contractual, billing, tenure, and service information.

The dataset includes variables representing areas such as:

- Customer demographics
- Customer tenure
- Contract type
- Monthly charges
- Service subscriptions
- Payment information
- Customer support/service interactions
- Customer churn status

These variables provide a suitable basis for investigating whether customer behavior contains patterns associated with service cancellation.

---

## 🔐 Data Leakage Prevention

To ensure that the model only uses information that would realistically be available when predicting churn, features that directly reveal the outcome or are only known after churn were excluded.

For example:

- `Churn Reason`
- Post-churn information
- Non-predictive customer identifiers

were excluded from the predictive feature set.

This helps ensure that the model learns from customer characteristics and behavior rather than directly accessing information that reveals the churn outcome.

---

# ⚙️ Methodology

The project follows the following machine learning workflow:

```text
Maven Churn Dataset
        │
        ▼
Data Exploration
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Train / Test Split
        │
        ▼
Model Training
        │
        ├───────────────┐
        ▼               ▼
Logistic Regression   Random Forest
        │               │
        └───────┬───────┘
                ▼
             XGBoost
                │
                ▼
        Model Evaluation
                │
                ▼
        XGBoost Selection
                │
                ▼
         SHAP Explainability
                │
        ┌───────┴────────┐
        ▼                ▼
 SHAP Summary Plot   SHAP Dependence
                         Plot
