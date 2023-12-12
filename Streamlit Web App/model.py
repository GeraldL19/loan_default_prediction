
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import train_test_split

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image

#Import dataset
df = pd.read_csv("C:/Users/geral/Documents/Westminster university/Final Year Project/Loan Approval/Dataset/credit_risk_dataset.csv")

df.dtypes
df.isnull().sum()

#Encode categorical values into nnumerical values
df = df.replace({'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3})
df = df.replace({'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5})
df = df.replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
df = df.replace({'Y': 0, 'N': 1})

#Delete N/A values
df.fillna(df.mean(numeric_only=True).round(1), inplace=True)

cols_to_norm = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']

# Fit the scaler on the training data and normalise data
scaler = StandardScaler()
scaler.fit(df[cols_to_norm])     
df[cols_to_norm] = scaler.transform(df[cols_to_norm])

#Create variable dataset and target dataset
x = df.drop('loan_status', axis=1)
y = df['loan_status']
     
#Creating train (70%) and test (30%) dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, shuffle=True)
     
#Fit RF classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#Predict test datset
y_pred = rf.predict(X_test)

#Accurcy measures
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
recall = recall_score(y_test, y_pred)
print("recall:", recall)
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()


# save the model to disk
joblib.dump(rf, "rf_model.sav")  
# Save the scaler
joblib.dump(scaler, "standard_scaler.sav")