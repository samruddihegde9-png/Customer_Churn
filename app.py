import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.title("📉 Customer Churn Dashboard")

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])

# -------------------------
# TAB 1 – PREDICTION
# -------------------------

with tab1:

    st.header("Customer Churn Prediction")

    gender = st.selectbox("Gender", ["Female", "Male"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    if st.button("Predict Churn"):

        input_dict = {feature: 0 for feature in features}

        input_dict["tenure"] = tenure
        input_dict["MonthlyCharges"] = monthly
        input_dict["TotalCharges"] = total

        if gender == "Male":
            input_dict["gender_Male"] = 1

        if contract == "One year":
            input_dict["Contract_One year"] = 1
        elif contract == "Two year":
            input_dict["Contract_Two year"] = 1

        if internet == "Fiber optic":
            input_dict["InternetService_Fiber optic"] = 1
        elif internet == "No":
            input_dict["InternetService_No"] = 1

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        st.progress(prob)

        st.write(f"Churn Risk: **{prob*100:.2f}%**")

        if prediction == 1:
            st.error("⚠️ Customer likely to churn")
        else:
            st.success("✅ Customer likely to stay")


# -------------------------
# TAB 2 – MODEL INSIGHTS
# -------------------------

with tab2:

    st.header("Model Insights")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()

    ax.barh(importance_df["Feature"], importance_df["Importance"])

    ax.set_title("Top Features Influencing Churn")

    ax.invert_yaxis()

    st.pyplot(fig)