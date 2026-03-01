import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Telco_Customer_Churn_Dataset  (1).csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df.drop('customerID', axis=1, inplace=True)

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

df['gender'] = df['gender'].map({'Male':1, 'Female':0})

df = pd.get_dummies(df, drop_first=True)

df = df.astype(int)

scaler = StandardScaler()
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.info())
print(df.head())