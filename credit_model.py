import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib 


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

# --- Day 1: Loading Data ---
column_names = [
    'existing_checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'present_employment', 'installment_rate', 'personal_status_sex',
    'other_debtors_guarantors', 'present_residence_since', 'property', 'age',
    'other_installment_plans', 'housing', 'number_of_existing_credits', 'job',
    'number_of_people_liable', 'telephone', 'foreign_worker', 'creditworthiness'
]
df = pd.read_csv(url, sep=r'\s+', header=None, names=column_names)


df['creditworthiness'] = df['creditworthiness'].map({1: 1, 2: 0})
categorical_features = df.select_dtypes(include=['object']).columns
df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)


X = df_processed.drop('creditworthiness', axis=1)
y = df_processed['creditworthiness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_scaled, y_train)



joblib.dump(model, 'credit_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

joblib.dump(X.columns, 'training_columns.joblib')


print("Model, scaler, and training columns have been saved successfully!")