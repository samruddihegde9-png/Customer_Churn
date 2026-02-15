import pandas as pd

def load_and_preprocess(path):
    # Load dataset
    df = pd.read_csv(path)

    print("Dataset Loaded. Shape:", df.shape)

    # Drop customer ID (not useful for prediction)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target variable (Yes/No → 1/0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Convert categorical features to numeric
    df = pd.get_dummies(df, drop_first=True)

    print("Preprocessing Complete. Shape:", df.shape)

    return df


# Run this file alone to test preprocessing
if __name__ == "__main__":
    df = load_and_preprocess("data/Telco-Customer-Churn.csv")
    print(df.head())
