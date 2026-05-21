import streamlit as st
import joblib
import pandas as pd

#Load model 
model = joblib.load('models/churn_pipeline.pkl')

#Page config 
st.set_page_config(page_title="Churn Predictor", page_icon="📡")
st.title("📡 Telco Customer Churn Predictor")
st.markdown("Fill in customer details to predict whether they will churn.")

# Input form 
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

st.subheader("Billing")
col4, col5 = st.columns(2)

with col4:
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col5:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

#Predict
if st.button("Predict Churn", type="primary"):

    # Build input dataframe — must match exact column names from training
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **churn**.")
    else:
        st.success(f"✅ This customer is likely to **stay**.")

    st.metric("Churn Probability", f"{probability:.1%}")

    # Show the input back as a sanity check
    with st.expander("See input data"):
        st.dataframe(input_data)