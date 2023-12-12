import streamlit as st 
import pandas as pd
import numpy as np
from prediction import predict

st.title("Streamlit Trial")
st.markdown("""
Predict your loan approval.
         """)

#User inputs
age = st.number_input("Age: ", 0,100)
income = st.number_input("Income: ", 0,100000000)
ownership = st.selectbox("House ownership: ", ["rent", "own", "mortgage"])
employement = st.number_input("Employement length: ", 0,200)
loan_intent = st.selectbox("Loan purpose: ", ["personal", "education","medical", "home improvement", "debt consolidation"])
grade = st.selectbox("Loan grade: ", ["A", "B", "C", "D", "E", "F"])
amount = st.number_input("Loan amount: ", 0,1000000)
interest = st.number_input("Interest rate: ", 0,50)
percent = st.number_input("Loan percent: ", 0.0,0.1)
default = st.selectbox("Previously defaulted? ", ["Y", "N"])
credit_hist = st.number_input("credithist: ", 0,100)

# "loan_status"
columns = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]
cols_to_norm = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length','person_income','person_income','person_income']

#Create dataframe with user input values
user_input = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [ownership],
    'person_emp_length': [employement],
    'loan_intent': [loan_intent],
    'loan_grade': [grade],
    'loan_amnt': [amount],
    'loan_int_rate': [interest],
    'loan_percent_income': [percent],
    'cb_person_default_on_file': [default],
    'cb_person_cred_hist_length': [credit_hist],
})

#Display user input into a tabel
st.table(user_input)

#if st.button("Predict type of Iris"):
#    result = predict(user_input)
 #   st.text(result[0])
#st.button("Predict", on_click=predict)

if st.button("Predict type of Iris"):
    result = predict(user_input)
    if result[0] == 0:
        st.text("Accepted")
    else:
        st.text("Rejected")


