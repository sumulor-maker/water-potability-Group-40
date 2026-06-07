# -*- coding: utf-8 -*-
"""Water Potability.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import missingno as msno
import joblib  # <-- added for saving models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

"""##DATA LOADING AND EXPLORATION"""

df = pd.read_csv("water_potability.csv")

df.head()
df.info()
df.shape
df.duplicated().sum()
df.isnull().sum()
df.describe().T

df['ph'].mean()
df['Sulfate'].mean()
df['Trihalomethanes'].mean()

ph_mean = df['ph'].mean()
print(ph_mean)

Sulfate_mean = df['Sulfate'].mean()
print(Sulfate_mean)

Trihalomethanes_mean = df['Trihalomethanes'].mean()
print(Trihalomethanes_mean)

df["ph"] = df['ph'].fillna(ph_mean)
df["Sulfate"] = df['Sulfate'].fillna(Sulfate_mean)
df["Trihalomethanes"] = df['Trihalomethanes'].fillna(Trihalomethanes_mean)

df.isnull().sum()

"""##RANDOM FOREST"""

X = df.drop(columns=['Potability'])
y = df['Potability']

X = X.fillna(X.mean(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

rf = accuracy_score(y_test, y_pred)
print(rf)

print("Confusion_matrix:", confusion_matrix(y_test, y_pred))
print("precision_score:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("f1:", f1_score(y_test, y_pred))

# ✅ Save Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved.")

"""##Correlation Between Features"""

df.corr()

sns.clustermap(df.corr(), cmap="vlag", dendrogram_ratio=(0.1, 0.2), annot=True, linewidths=.8, figsize=(9,10))
plt.show()

corr = df.corr()
c1 = corr.abs().unstack()
c1.sort_values(ascending=False)[12:24:2]

"""##Distribution of Features"""

non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")

plt.figure(figsize=(15,15))
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3, 3, ax + 1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col], label="Non Potable")
    sns.kdeplot(x=potable[col], label="Potable")
    plt.legend()
plt.tight_layout()

ax = sns.countplot(x="Potability", data=df, saturation=0.8)
plt.xticks(ticks=[0, 1], labels=["Not Potable", "Potable"])
plt.show()

x = df.Potability.value_counts()
labels = [0, 1]
print(x)

plt.rcParams['figure.figsize'] = [20, 10]
df.hist()
plt.show()

fig = px.pie(df, names="Potability", hole=0.4, template="plotly_dark")
fig.show()

"""##LOGISTIC REGRESSION"""

scaler = StandardScaler()

X = df.drop(columns=['Potability'])
X = X.fillna(X.mean(numeric_only=True))

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

y_proba = model.predict_proba(X_test)[:, 1]
print(y_proba)

lg = accuracy_score(y_test, y_pred)
print(lg)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

pred_lg = model.predict(X_test)
cm1 = confusion_matrix(y_test, pred_lg)
sns.heatmap(cm1/np.sum(cm1), annot=True, fmt='0.2%', cmap='Reds')

# ✅ Save Logistic Regression model + its scaler
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Logistic Regression model and scaler saved.")

"""##DECISION TREE"""

model_dt = DecisionTreeClassifier(max_depth=8, random_state=42)
model_dt.fit(X_train, y_train)

pred_dt = model_dt.predict(X_test)

dt = accuracy_score(y_test, pred_dt)
print(dt)

print(classification_report(y_test, pred_dt))

cm2 = confusion_matrix(y_test, pred_dt)
sns.heatmap(cm2/np.sum(cm2), annot=True, fmt='0.2%', cmap='Reds')

# ✅ Save Decision Tree model
joblib.dump(model_dt, "decision_tree_model.pkl")
print("Decision Tree model saved.")

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy_score': [lg, dt, rf]
})
models

sns.barplot(x='Accuracy_score', y='Model', data=models)
