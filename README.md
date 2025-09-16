# Credit Scoring & Loan Eligibility Predictor 🇮🇳

This project is a machine learning application that predicts an individual's creditworthiness and loan eligibility based on their past financial data. The model is deployed in an interactive web application built with Streamlit.

![App Screenshot](Screenshot-of-App.png)
*(Note: You will need to replace 'Screenshot-of-App.png' with the actual name of your screenshot file after uploading it to GitHub.)*

---

## 📋 Features

- **End-to-End Machine Learning Pipeline:** From data loading and preprocessing to model training and evaluation.
- **Predictive Model:** Uses a **Logistic Regression** model trained on the German Credit Data dataset.
- **Data Preprocessing:** Implements one-hot encoding for categorical variables and feature scaling (`StandardScaler`) for improved model performance.
- **Model Evaluation:** The model's performance is assessed using key metrics like **Precision, Recall, F1-Score, and ROC-AUC Score**.
- **Interactive UI:** A user-friendly web interface built with **Streamlit** that allows for real-time predictions.
- **Localized for India:** The UI has been adapted for an Indian context, using Rupees (₹) and more familiar financial terms.

---

## 🛠️ Technologies Used

- **Python:** The core programming language.
- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** For building and evaluating the machine learning model.
- **Streamlit:** For creating and serving the web application.
- **Joblib:** For saving and loading the trained model and scaler.

---

## 🚀 How to Run Locally

To run this application on your own machine, follow these steps:

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
2. Install Dependencies:
It's recommended to create a virtual environment first. Then, install the required libraries from the requirements.txt file.

Bash

pip install -r requirements.txt
3. Run the Streamlit App:
Once the dependencies are installed, run the main application file:

Bash

streamlit run app.py
The application should now be running and accessible in your web browser.

📁 Project Structure
├── app.py                     # The main Streamlit application script
├── credit_model.py            # Script for data processing, model training, and evaluation
├── credit_model.joblib        # Saved trained Logistic Regression model
├── scaler.joblib              # Saved feature scaler
├── training_columns.joblib    # Saved column order for prediction
├── requirements.txt           # List of Python dependencies
└── README.md                  # This file





