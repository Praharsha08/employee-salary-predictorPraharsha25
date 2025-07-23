import streamlit as st
import pandas as pd
import joblib

# ------------------------
# App Title & Description
# ------------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E4053;'> Employee Salary Predictor using AI/ML</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #34495E;'>Predict an employee's salary based on age, gender, education level, job title, and experience.</p>",
    unsafe_allow_html=True
)

# ------------------------
# Load the model and features
# ------------------------
model = joblib.load("salary_model.pkl")
expected_columns = joblib.load("model_columns.pkl")

# ------------------------
# Form for input
# ------------------------
with st.form("prediction_form"):
    st.subheader("Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    # Extract job titles from expected columns for dropdown
    job_title_options = sorted([col.replace("Job Title_", "") for col in expected_columns if col.startswith("Job Title_")])
    job_title = st.selectbox("Job Title", job_title_options)
    experience = st.slider("Years of Experience", 0, 40, 3)

    submit = st.form_submit_button("🎯 Predict Salary")

# ------------------------
# Prediction Logic
# ------------------------
if submit:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])

    # One-hot encoding and alignment
    encoded = pd.get_dummies(input_data)
    encoded = encoded.reindex(columns=expected_columns, fill_value=0)

    # Predict
    predicted_salary = model.predict(encoded)[0]
    st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")

# ------------------------
# Footer
# ------------------------
st.markdown(
    "<hr><div style='text-align: center; color: gray; font-size: 14px;'>Developed by Your Name | Dept. of CSE | Final Year Project</div>",
    unsafe_allow_html=True
)