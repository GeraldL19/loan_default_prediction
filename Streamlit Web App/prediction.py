import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

columns = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]
cols_to_norm = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']

def predict(data):
    #Load trained model
    rf = joblib.load("rf_model.sav")
    # Load the StandardScaler
    scaler = joblib.load("standard_scaler.sav")
    
    # Perform one-hot encoding for categorical variables
    data = data.replace({'rent': 0, 'own': 1, 'mortgage': 2, 'other': 3})
    data = data.replace({'personal': 0, 'education': 1, 'medical': 2, 'venture': 3, 'home improvement': 4, 'debt consolidation': 5})
    data = data.replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    data = data.replace({'Y': 0, 'N': 1})
    # Scale the input data using the loaded scaler
    data[cols_to_norm] = scaler.transform(data[cols_to_norm])

    #Return prediction
    return rf.predict(data)




