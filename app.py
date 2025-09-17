import streamlit as st
import pandas as pd
import joblib

# Load the components
model = joblib.load('credit_model.joblib')
scaler = joblib.load('scaler.joblib')
training_columns = joblib.load('training_columns.joblib')

# Mappings
personal_status_map = {
    "Male: Single": "A93", "Female: Divorced/Separated/Married": "A92",
    "Male: Married/Widowed": "A94", "Male: Divorced/Separated": "A91"
}
job_map = {
    "Skilled Employee / Salaried": "A173", "Unskilled - Resident": "A172",
    "Management / Self-employed / Professional": "A174", "Unemployed / Unskilled": "A171"
}
purpose_map = {
    "New Car": "A40", "Used Car": "A41", "Furniture/Equipment": "A42",
    "Electronics/TV": "A43", "Home Appliances": "A44", "Repairs": "A45",
    "Education": "A46", "Business": "A49", "Other": "A410"
}
credit_history_map = {
    "All existing loans paid back on time": "A32", "Critical account / Other loans existing": "A34",
    "Delay in paying loans in the past": "A33", "All loans at this bank paid back on time": "A31",
    "No previous loans / All loans paid back": "A30"
}
checking_account_map = {
    "No Checking Account": "A14", "< ₹0 Balance": "A11",
    "₹0 to ₹20,000 Balance": "A12", ">= ₹20,000 Balance": "A13"
}
savings_account_map = {
    "Unknown / No Savings Account": "A65", "< ₹10,000": "A61",
    "₹10,000 to ₹50,000": "A62", "₹50,000 to ₹1,00,000": "A63",
    ">= ₹1,00,000": "A64"
}

# UI Setup
st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("Loan Eligibility Predictor 🇮🇳")
st.write("Enter the applicant's details to check their loan eligibility.")

# Input Form
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 21, 70, 50)
    personal_status_text = st.selectbox("Personal Status & Sex", list(personal_status_map.keys()), index=2)
    job_text = st.selectbox("Job Type", list(job_map.keys()), index=2)
with col2:
    duration = st.slider("Loan Duration (months)", 6, 60, 12)
    credit_amount = st.number_input("Loan Amount (₹)", 25000, 2500000, 150000)
    purpose_text = st.selectbox("Purpose of Loan", list(purpose_map.keys()), index=2)
with col3:
    credit_history_text = st.selectbox("Credit History", list(credit_history_map.keys()))
    checking_account_text = st.selectbox("Checking Account Status", list(checking_account_map.keys()), index=3)
    savings_account_text = st.selectbox("Savings Account / Deposits", list(savings_account_map.keys()), index=4)

if st.button("Check Loan Eligibility", type="primary"):
    # Create a dataframe of all zeros with the correct columns
    input_data = pd.DataFrame(columns=training_columns, index=[0])
    input_data.fillna(0, inplace=True)

    # Fill numerical values
    input_data.loc[0, 'age'] = age
    input_data.loc[0, 'duration'] = duration
    input_data.loc[0, 'credit_amount'] = credit_amount
    input_data.loc[0, 'installment_rate'] = 4
    input_data.loc[0, 'present_residence_since'] = 4
    input_data.loc[0, 'number_of_existing_credits'] = 1
    input_data.loc[0, 'number_of_people_liable'] = 1

    # Construct and set one-hot encoded features
    categorical_inputs = {
        'personal_status_sex_': (personal_status_text, personal_status_map),
        'job_': (job_text, job_map),
        'purpose_': (purpose_text, purpose_map),
        'credit_history_': (credit_history_text, credit_history_map),
        'existing_checking_account_': (checking_account_text, checking_account_map),
        'savings_account_': (savings_account_text, savings_account_map)
    }
    for prefix, (selected_text, code_map) in categorical_inputs.items():
        col_name = prefix + code_map[selected_text]
        if col_name in input_data.columns:
            input_data.loc[0, col_name] = 1
    
    # Set default one-hot encoded values
    default_cols = [
        'property_A122', 'present_employment_A75', 'other_debtors_guarantors_A101',
        'other_installment_plans_A143', 'housing_A152', 'telephone_A191', 'foreign_worker_A201'
    ]
    for col in default_cols:
        if col in input_data.columns:
            input_data.loc[0, col] = 1

    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- BUSINESS RULE OVERRIDE ---
    # If credit history is critical, decline the loan regardless of the model's prediction.
    if credit_history_text == "Critical account / Other loans existing":
        prediction[0] = 0 # Force prediction to Decline
        st.subheader("Prediction Result")
        st.error("Loan Declined ❌ (Reason: Critical Credit History)")
    else:
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"Loan Approved ✅ (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.error(f"Loan Declined ❌ (Confidence: {prediction_proba[0][0]*100:.2f}%)")