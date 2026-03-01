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
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("Churn distribution in full dataset:")
print(y.value_counts(normalize=True))

print("Churn distribution in training set:")
print(y_train.value_counts(normalize=True))

print("Churn distribution in testing set:")
print(y_test.value_counts(normalize=True))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Evaluation:\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = df.corr()
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)

important_corr = churn_corr[abs(churn_corr) > 0.10]

plt.figure(figsize=(8, 12))

sns.heatmap(
    important_corr.to_frame(),
    annot=True,
    cmap='RdBu_r',
    center=0,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Strength'}
)

plt.title("Strong Feature Correlation with Churn", fontsize=14)
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
import pandas as p


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

models = {
    "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append([name, accuracy, recall, roc_auc])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall (Churn)", "ROC-AUC"])

print("\nModel Comparison:\n")
print(results_df.sort_values(by="ROC-AUC", ascending=False))
from sklearn.linear_model import LogisticRegression

final_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

final_model.fit(X_train, y_train)

print("\nFinal Model Training Completed Successfully!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nFinal Model Evaluation on Test Data:\n")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred))
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="Model ROC Curve")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()