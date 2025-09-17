import pandas as pd
import joblib

# Load the final, unbiased components
model = joblib.load('credit_model.joblib')
scaler = joblib.load('scaler.joblib')
training_columns = joblib.load('training_columns.joblib')

# Define the absolute WORST-CASE credit profile
high_risk_profile = {
    'age': 21,
    'duration': 60,
    'credit_amount': 1500000,
    'personal_status_sex': 'A93',  # Male: Single
    'job': 'A171',                 # Unemployed / Unskilled
    'purpose': 'A41',                # Used Car
    'credit_history': 'A34',         # Critical account
    'existing_checking_account': 'A11', # < 0 Balance
    'savings_account': 'A65'       # No Savings Account
}

# --- Data Processing (Identical to app.py) ---
input_data = pd.DataFrame(columns=training_columns, index=[0])
input_data.fillna(0, inplace=True)

# Fill numerical values
for key in ['age', 'duration', 'credit_amount']:
    input_data.loc[0, key] = high_risk_profile[key]
input_data.loc[0, 'installment_rate'] = 4
input_data.loc[0, 'present_residence_since'] = 4
input_data.loc[0, 'number_of_existing_credits'] = 1
input_data.loc[0, 'number_of_people_liable'] = 1

# Construct column names for one-hot encoded features
categorical_map = {
    'personal_status_sex_': high_risk_profile['personal_status_sex'],
    'job_': high_risk_profile['job'],
    'purpose_': high_risk_profile['purpose'],
    'credit_history_': high_risk_profile['credit_history'],
    'existing_checking_account_': high_risk_profile['existing_checking_account'],
    'savings_account_': high_risk_profile['savings_account']
}
for prefix, value in categorical_map.items():
    col_name = prefix + value
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

# Scale the data and make the final prediction
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Print the final result
print("--- Final Model Test ---")
if prediction[0] == 0:
    print("Result: Loan Declined (Prediction: [0])")
else:
    print("Result: Loan Approved (Prediction: [1])")