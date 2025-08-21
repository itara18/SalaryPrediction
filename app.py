import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('rfr_prediction_model.pkl')
seniority_encoder = joblib.load('seniority_encoder.pkl')
training_columns = joblib.load('training_columns.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("Salary Prediction App")
st.write("Enter the details below to predict the salary.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal and Job Info")
    age = st.slider("Age", 18, 70, 30)
    seniority = st.selectbox("Seniority Level", ['na', 'jr', 'senior'])
    seniority_encoded = seniority_encoder.transform([[seniority]])[0]
    jd_length = st.slider("Job Description Length", 0, 10000, 1000)
    job_role = st.selectbox("Job Role", ["data scientist", "data engineer", "analyst", "manager", "mle", "director", "na"])
    num_comp = st.slider("Number of Competitors", 0, 100, 5)

with col2:
    st.subheader("Company Info")
    rating = st.slider("Company Rating", 0.0, 5.0, 3.5, 0.1)
    founded = st.number_input("Company Founded Year", 1800, 2025, 2000)
    company_size = st.selectbox("Company Size", ["1 to 50", "51 to 200", "201 to 500", "501 to 1000", "1001 to 5000", "5001 to 10000", "10000+"])
    job_state = st.selectbox("Job State", ["CA", "NY", "TX", "WA", "Other"])
    type_ownership = st.selectbox("Type of Ownership", ["Private", "Public", "Subsidiary", "Other"])
    industry = st.selectbox("Industry", ["Tech", "Healthcare", "Finance", "Other"])
    sector = st.selectbox("Sector", ["IT", "Education", "Finance", "Other"])
    revenue = st.selectbox("Revenue", ["<$1M", "$1M-$10M", "$10M-$100M", ">$100M", "Unknown"])

st.subheader("Required Skills")
col3, col4, col5 = st.columns(3)
with col3:
    python_req = st.radio("Python Required?", ["No", "Yes"])
    r_req = st.radio("R Required?", ["No", "Yes"])

with col4:
    spark_req = st.radio("Spark Required?", ["No", "Yes"])
    aws_req = st.radio("AWS Required?", ["No", "Yes"])

with col5:
    excel_req = st.radio("Excel Required?", ["No", "Yes"])

python_req = 1 if python_req == "Yes" else 0
r_req = 1 if r_req == "Yes" else 0
spark_req = 1 if spark_req == "Yes" else 0
aws_req = 1 if aws_req == "Yes" else 0
excel_req = 1 if excel_req == "Yes" else 0

input_df = pd.DataFrame({
    "age": [age],
    "rating": [rating],
    "founded": [founded],
    "Seniority Level": [seniority_encoded],
    "Python Required": [python_req],
    "R Required": [r_req],
    "Spark Required": [spark_req],
    "AWS Required": [aws_req],
    "Excel Required": [excel_req],
    "Number of Competitors": [num_comp],
    "Job Description Length": [jd_length],
    "Job Role": [job_role],
    "Company Size": [company_size],
    "Job State": [job_state],
    "Type of Ownership": [type_ownership],
    "Industry": [industry],
    "Sector": [sector],
    "Revenue": [revenue]
})

categorical_cols = ['Job Role', 'Company Size', 'Job State', 'Type of Ownership', 'Industry', 'Sector', 'Revenue']
input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
input_df = input_df.reindex(columns=training_columns,fill_value=0)
input_scaled = scaler.transform(input_df)

if st.button("Predict Salary"):
    pred_salary = model.predict(input_scaled)[0]
    st.success(f"Predicted Salary: **${pred_salary:.2f}K**")
    st.balloons()