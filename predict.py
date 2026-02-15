import joblib
import pandas as pd

# Load saved files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

def predict_customer(input_data):
    df = pd.DataFrame([input_data])

    # Convert to dummy variables
    df = pd.get_dummies(df)

    # Add missing columns to match training data
    for col in features:
        if col not in df:
            df[col] = 0

    df = df[features]

    # Scale
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        print("⚠ Customer is likely to CHURN")
    else:
        print("✅ Customer is likely to STAY")

# Example test customer
new_customer = {
    "tenure": 3,
    "MonthlyCharges": 85,
    "TotalCharges": 250,
    "SeniorCitizen": 0
}

predict_customer(new_customer)
