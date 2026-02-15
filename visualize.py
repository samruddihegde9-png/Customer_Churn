import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (raw data for visualization)
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric for plotting
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# -------------------------------
# 1️⃣ Churn Distribution
# -------------------------------
plt.figure(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------
# 2️⃣ Tenure vs Churn
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(df[df["Churn"]=="No"]["tenure"], bins=30, alpha=0.6, label="Stayed")
plt.hist(df[df["Churn"]=="Yes"]["tenure"], bins=30, alpha=0.6, label="Churned")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customers")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ Monthly Charges vs Churn
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(df[df["Churn"]=="No"]["MonthlyCharges"], bins=30, alpha=0.6, label="Stayed")
plt.hist(df[df["Churn"]=="Yes"]["MonthlyCharges"], bins=30, alpha=0.6, label="Churned")
plt.title("Monthly Charges vs Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Customers")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 4️⃣ Contract Type vs Churn
# -------------------------------
contract_churn = pd.crosstab(df["Contract"], df["Churn"])
contract_churn.plot(kind="bar", figsize=(6,4))
plt.title("Contract Type Impact on Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("Visualizations generated successfully!")
