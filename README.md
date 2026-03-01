# 📊 Customer Churn Prediction - Machine Learning Project

## 📌 Project Overview

This project analyzes customer churn in a telecommunications company and builds predictive machine learning models to identify customers likely to leave the service.

The goal is to help businesses take proactive actions to improve customer retention.

---

## 🎯 Problem Statement

Customer churn is a major issue in the telecom industry. Acquiring new customers is more expensive than retaining existing ones. 

This project predicts whether a customer will churn using historical customer data.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📂 Project Structure
Customer-Churn-Prediction/
│
├── Telco_Customer_Churn_Dataset.csv
├── data_preprocessing.py
├── model_training.py
├── README.md

---

## 🔍 Data Preprocessing

- Converted `TotalCharges` to numeric
- Handled missing values
- Encoded categorical variables
- Applied one-hot encoding
- Scaled numerical features
- Performed 80-20 train-test split with stratification

---

## 🤖 Models Implemented

- Logistic Regression (Balanced)
- Decision Tree
- Random Forest
- Gradient Boosting

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Special focus was given to **Recall for churn class**, since identifying at-risk customers is critical.

---

## 🏆 Key Insights

- Customers with short tenure are more likely to churn.
- Month-to-month contract users show higher churn.
- Higher monthly charges increase churn probability.
- Electronic check payment users churn more frequently.

---

## 📊 Results

The selected model achieved strong performance with high ROC-AUC and improved recall for churn prediction.

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Feature engineering
- SHAP explainability
- Model deployment using Streamlit

---

## 👨‍💻 Author

HMAbhishekk  
Machine Learning & AI Enthusiast