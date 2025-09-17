import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model, scaler, and training columns
model = joblib.load('credit_model.joblib')
scaler = joblib.load('scaler.joblib')
training_columns = joblib.load('training_columns.joblib')

# --- Mappings for User-Friendly Input (Indian Context) ---
personal_status_map = {
    "Male: Single": "A93",
    "Female: Divorced/Separated/Married": "A92",
    "Male: Married/Widowed": "A94",
    "Male: Divorced/Separated": "A91"
}
job_map = {
    "Skilled Employee / Salaried": "A173",
    "Unskilled - Resident": "A172",
    "Management / Self-employed / Professional": "A174",
    "Unemployed / Unskilled": "A171"
}
purpose_map = {
    "New Car": "A40",
    "Used Car": "A41",
    "Furniture/Equipment": "A42",
    "Electronics/TV": "A43",
    "Home Appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Business": "A49",
    "Other": "A410"
}
credit_history_map = {
    "All existing loans paid back on time": "A32",
    "Critical account / Other loans existing": "A34",
    "Delay in paying loans in the past": "A33",
    "All loans at this bank paid back on time": "A31",
    "No previous loans / All loans paid back": "A30"
}
checking_account_map = {
    "No Checking Account": "A14",
    "< ₹0 Balance": "A11",
    "₹0 to ₹20,000 Balance": "A12",
    ">= ₹20,000 Balance": "A13"
}
savings_account_map = {
    "Unknown / No Savings Account": "A65",
    "< ₹10,000": "A61",
    "₹10,000 to ₹50,000": "A62",
    "₹50,000 to ₹1,00,000": "A63",
    ">= ₹1,00,000": "A64"
}

# --- UI Setup ---
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("Loan Eligibility Predictor 🇮🇳")
st.write("Enter the applicant's details to check their loan eligibility.")

# --- Input Form ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Personal Information")
    age = st.slider("Age", 21, 70, 35)
    personal_status_text = st.selectbox("Personal Status & Sex", options=list(personal_status_map.keys()))
    job_text = st.selectbox("Job Type", options=list(job_map.keys()))
    
with col2:
    st.header("Loan Details")
    duration = st.slider("Loan Duration (months)", 6, 60, 24)
    
    credit_amount = st.number_input("Loan Amount (₹)", min_value=25000, max_value=2500000, value=300000, step=10000)
    purpose_text = st.selectbox("Purpose of Loan", options=list(purpose_map.keys()))

with col3:
    st.header("Financial History")
    credit_history_text = st.selectbox("Credit History", options=list(credit_history_map.keys()))
    checking_account_text = st.selectbox("Checking Account Status", options=list(checking_account_map.keys()))
    savings_account_text = st.selectbox("Savings Account / Deposits", options=list(savings_account_map.keys()))


if st.button("Check Loan Eligibility", type="primary"):

    personal_status_sex = personal_status_map[personal_status_text]
    job = job_map[job_text]
    purpose = purpose_map[purpose_text]
    credit_history = credit_history_map[credit_history_text]
    existing_checking_account = checking_account_map[checking_account_text]
    savings_account = savings_account_map[savings_account_text]

    user_input = {
        'age': age, 'duration': duration, 'credit_amount': credit_amount,
        'personal_status_sex': personal_status_sex, 'job': job, 'purpose': purpose,
        'credit_history': credit_history, 'existing_checking_account': existing_checking_account,
        'savings_account': savings_account, 'installment_rate': 4, 'present_residence_since': 4,
        'number_of_existing_credits': 1, 'number_of_people_liable': 1, 'present_employment': 'A75',
        'other_debtors_guarantors': 'A101', 'property': 'A121', 'other_installment_plans': 'A143',
        'housing': 'A152', 'telephone': 'A191', 'foreign_worker': 'A201'
    }
    input_df = pd.DataFrame([user_input])
    input_processed = pd.get_dummies(input_df).reindex(columns=training_columns, fill_value=0)
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
      
        st.success(f"Loan Approved ✅ (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.error(f"Loan Declined ❌ (Confidence: {prediction_proba[0][0]*100:.2f}%)")