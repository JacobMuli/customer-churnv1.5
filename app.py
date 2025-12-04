import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(page_title="Customer Churn App", page_icon="📊", layout="wide")

# ----------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------
menu = st.sidebar.selectbox(
    "📌 Select a Page",
    [
        "🔮 Prediction",
        "📊 EDA Overview",
        "📈 Numeric Feature Analysis",
        "📉 Categorical Analysis",
        "📑 Statistical Insights",
    ]
)

# ----------------------------------------
# LOAD MODEL ARTIFACTS
# ----------------------------------------
model = joblib.load("model/lgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# Load dataset (for EDA pages)
test_df = pd.read_csv("data/test_processed.csv")

# ----------------------------------------
# PAGE: PREDICTION
# ----------------------------------------
if menu == "🔮 Prediction":
    st.title("🔮 Customer Churn Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
            usage = st.number_input("Usage Frequency", min_value=0, value=10)
            support_calls = st.number_input("Support Calls", min_value=0, value=2)

        with col2:
            payment_delay = st.number_input("Payment Delay (Days)", min_value=0, value=1)
            total_spend = st.number_input("Total Spend", min_value=0, value=500)
            last_interaction = st.number_input("Last Interaction (Days)", min_value=0, value=30)

        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Annual", "Monthly", "Quarterly"])

        submit = st.form_submit_button("Predict Churn")

    def preprocess():
        gender_val = 1 if gender == "Female" else 0
        subs = {
            "Subscription Type_Basic": 1 if subscription == "Basic" else 0,
            "Subscription Type_Standard": 1 if subscription == "Standard" else 0,
            "Subscription Type_Premium": 1 if subscription == "Premium" else 0,
        }
        contracts = {
            "Contract Length_Annual": 1 if contract == "Annual" else 0,
            "Contract Length_Monthly": 1 if contract == "Monthly" else 0,
            "Contract Length_Quarterly": 1 if contract == "Quarterly" else 0,
        }

        data = {
            "Age": age,
            "Gender": gender_val,
            "Tenure": tenure,
            "Usage Frequency": usage,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
            **subs,
            **contracts,
            "Avg_Monthly_Spend": total_spend / max(tenure, 1),
            "Support_Intensity": support_calls / max(usage, 1),
            "Recency_Tenure_Ratio": last_interaction / max(tenure, 1),
        }

        df = pd.DataFrame([data])

        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]
        return scaler.transform(df)

    if submit:
        X = preprocess()
        proba = model.predict_proba(X)[0][1]
        pred = int(proba > 0.5)

        st.subheader("🔍 Result")
        st.write(f"**Churn Probability:** `{proba:.2f}`")

        if pred == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")


# ----------------------------------------
# PAGE: EDA OVERVIEW
# ----------------------------------------
elif menu == "📊 EDA Overview":
    st.title("📊 EDA Overview")

    st.subheader("Dataset Preview")
    st.write(test_df.head())

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=test_df, x="Churn", ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(test_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------------------------------
# PAGE: NUMERIC FEATURE ANALYSIS
# ----------------------------------------
elif menu == "📈 Numeric Feature Analysis":
    st.title("📈 Numeric Feature Analysis")

    numeric_cols = [
        'Age','Tenure','Usage Frequency','Support Calls',
        'Payment Delay','Total Spend','Last Interaction',
        'Avg_Monthly_Spend','Support_Intensity','Recency_Tenure_Ratio'
    ]

    for col in numeric_cols:
        st.subheader(f"{col}")
        fig, ax = plt.subplots()
        sns.histplot(test_df[col], kde=True, ax=ax)
        st.pyplot(fig)

# ----------------------------------------
# PAGE: CATEGORICAL FEATURE ANALYSIS
# ----------------------------------------
elif menu == "📉 Categorical Analysis":
    st.title("📉 Categorical Feature Analysis")

    cat_cols = [
        'Gender',
        'Subscription Type_Basic','Subscription Type_Standard','Subscription Type_Premium',
        'Contract Length_Annual','Contract Length_Monthly','Contract Length_Quarterly'
    ]

    for col in cat_cols:
        st.subheader(col)
        fig, ax = plt.subplots()
        sns.countplot(data=test_df, x=col, hue="Churn", ax=ax)
        st.pyplot(fig)

# ----------------------------------------
# PAGE: STATISTICAL INSIGHTS
# ----------------------------------------
elif menu == "📑 Statistical Insights":
    st.title("📑 Statistical Insights")

    numeric_cols = [
        'Age','Tenure','Usage Frequency','Support Calls',
        'Payment Delay','Total Spend','Last Interaction',
        'Avg_Monthly_Spend','Support_Intensity','Recency_Tenure_Ratio'
    ]

    results = []
    for col in numeric_cols:
        c0 = test_df[test_df["Churn"] == 0][col]
        c1 = test_df[test_df["Churn"] == 1][col]
        stat, p = ttest_ind(c0, c1, equal_var=False)
        results.append({"Feature": col, "p-value": p})

    st.dataframe(pd.DataFrame(results))
