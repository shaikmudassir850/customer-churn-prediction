# Customer Churn Prediction | End-to-End Machine Learning Project

> An end-to-end Machine Learning application that predicts customers who are likely to churn, helping businesses identify at-risk customers and support retention strategies.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-success)](https://customer-churn-prediction-z6bu.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black)](https://flask.palletsprojects.com/)

## 🚀 Live Demo

**Try the deployed application:**  
https://customer-churn-prediction-z6bu.onrender.com

---

## 📌 About the Project

Customer churn is a major business challenge where identifying customers who are likely to leave can help organizations take proactive retention actions.

I developed this project as an **end-to-end Machine Learning solution** that takes customer information as input and predicts whether the customer is:

- 🟢 **Likely to Stay**
- 🔴 **Likely to Churn**

The project covers the complete ML lifecycle — from data preprocessing and exploratory analysis to model training, class balancing, hyperparameter tuning, model serialization, Flask development, and cloud deployment.

---

## 🎯 Business Problem

Businesses need to identify customers who are at risk of leaving before they actually churn.

### Solution

The Machine Learning model analyzes customer attributes such as:

- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Payment method
- Average monthly usage
- Satisfaction score
- Autopay status

and predicts the customer's churn status.

---

## 🧠 Machine Learning Workflow

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Train / Test Split
       ↓
Feature Scaling
       ↓
SMOTE Class Balancing
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
Final Model
       ↓
Model Serialization
       ↓
Flask Web Application
       ↓
Render Deployment
---
## 🤖 Machine Learning Models

Multiple Machine Learning classification algorithms were trained and evaluated to predict whether a customer is likely to churn.

### 📌 Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- AdaBoost
- XGBoost

### 📊 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 🎯 Model Selection

The models were compared based on their classification performance, with particular attention to **Recall and F1-Score** because correctly identifying customers who may churn is important in a churn prediction use case.
---

## 📊 Model Evaluation

After training the models, their performance was evaluated using multiple classification metrics.

### Evaluation Metrics

- **Accuracy** – Measures the overall percentage of correct predictions.
- **Precision** – Measures how many predicted churners were actually churners.
- **Recall** – Measures how many actual churners were correctly identified.
- **F1-Score** – Provides a balance between Precision and Recall.
- **Confusion Matrix** – Shows the number of correct and incorrect predictions for each class.

### Why Recall Matters

In a customer churn problem, identifying actual churners is important because these customers may require further analysis or retention strategies.

Therefore, **Recall and F1-Score** were considered along with Accuracy during model evaluation.

---

## 🎯 Final Model Performance

During experimentation, the **AdaBoost Classifier** achieved an accuracy of approximately **80.14%**.

### Classification Report

```text
              Precision    Recall    F1-Score

Class 0          0.82       0.76       0.79
Class 1          0.79       0.84       0.81
## 🔧 Hyperparameter Tuning

To improve the performance of the Machine Learning model, hyperparameter tuning was performed using **GridSearchCV**.

### 🎯 Objective

The objective of hyperparameter tuning was to find the best combination of model parameters that provides better predictive performance.

### ⚙️ GridSearchCV

`GridSearchCV` tests different combinations of predefined hyperparameters using cross-validation.

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train_resampled,
    y_train_resampled
)
## 🚀 Deployment

The Customer Churn Prediction application was deployed as a **Flask web application** and hosted on **Render**.

### 🌐 Live Application

🔗 **[Customer Churn Prediction – Live Demo](https://customer-churn-prediction-z6bu.onrender.com)**

The deployed application allows users to enter customer details and receive a Machine Learning-based churn prediction.

### 🔄 Deployment Workflow

```text
Machine Learning Model
        ↓
Save Model using Joblib
        ↓
Build Flask Application
        ↓
Create requirements.txt
        ↓
Configure Gunicorn
        ↓
Push Project to GitHub
        ↓
Connect GitHub Repository to Render
        ↓
Deploy Flask Application
        ↓
Live Web Application
