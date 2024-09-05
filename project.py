# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://github.com/vanshjaiswal/Machine-Learning-Projects/blob/main/Project_Dataset/framingham.csv?raw=true'
data = pd.read_csv(url)

# Initial Inspection
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualizations
# Histograms for continuous variables
data.hist(bins=20, figsize=(14, 10))
plt.show()

# Bar plots for nominal variables
sns.countplot(data['male'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Handling Missing Values
data = data.fillna(data.mean())

# Encoding Categorical Variables
data['male'] = data['male'].map({1: 'Male', 0: 'Female'})
data = pd.get_dummies(data, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
columns_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Splitting Data
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression Evaluation
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))
print(confusion_matrix(y_test, y_pred_logreg))

# Random Forest Evaluation
y_pred_rf = rf.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# SVM Evaluation
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine:")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))


import joblib

# Save the best model (assuming Random Forest here)
joblib.dump(rf, 'best_model_rf.pkl')


