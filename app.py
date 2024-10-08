import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # Import joblib for scikit-learn models
import shap
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained models
logistic_regression_model = joblib.load('models/Logistic_Regression.pkl')  # scikit-learn model
naive_bayes_model = joblib.load('models/Naive_Bayes.pkl')  # scikit-learn model
svm_model = joblib.load('models/SVC.pkl')  # scikit-learn model
decision_tree_model = joblib.load('models/Decision_Tree.pkl')  # scikit-learn model
random_forest_model = joblib.load('models/Random_Forest.pkl')  # scikit-learn model

# Load dataset for mean and mode calculation
@st.cache_data
def load_data():
    return pd.read_csv('data/train_2v-clean.csv')

data = load_data()

# Title and introduction
st.title("Stroke Prediction App")
st.write("This app predicts the likelihood of stroke based on input data.")

# Display averages (mean) of numerical variables and mode of categorical variables
st.subheader("Dataset Overview")
st.write("Below you can see the average and most common values from the dataset.")

# Calculate and display the averages and modes
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
categorical_columns = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'ever_smoked', 'gender', 'smoking_status']

# Display mean for numerical variables
st.write("### Mean of numerical variables:")
st.write(data[numerical_columns].mean())

# Display mode for categorical variables
st.write("### Mode of categorical variables:")
st.write(data[categorical_columns].mode().iloc[0])

# Create the form for user input
st.subheader("Enter the details to predict stroke risk")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, step=1)
gender_display = st.selectbox("Gender", ["Male", "Female"])
hypertension_display = st.selectbox("Do you have hypertension?", ["Yes", "No"])
heart_disease_display = st.selectbox("Do you have heart disease?", ["Yes", "No"])
ever_married_display = st.selectbox("Have you ever been married?", ["Yes", "No"])
work_type_display = st.selectbox("Work Type", ["Children", "Government Job", "Never Worked", "Private", "Self-employed"])
residence_type_display = st.selectbox("Residence Type", ["Rural", "Urban"])
avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=250.0, step=0.1)
bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, step=0.1)
ever_smoked_display = st.selectbox("Have you ever smoked?", ["Yes", "No"])
smoking_status_display = st.selectbox("Smoking Status", ["Smokes", "Formerly smoked", "Never smoked"])

# Mapping input to model input
# Ensure encoding and matching feature format with the model
gender_encoded = 1 if gender_display == "Male" else 0
hypertension_encoded = 1 if hypertension_display == "Yes" else 0
heart_disease_encoded = 1 if heart_disease_display == "Yes" else 0
ever_married_encoded = 1 if ever_married_display == "Yes" else 0
work_type_mapping = {
    "Children": 0, "Government Job": 1, "Never Worked": 2,
    "Private": 3, "Self-employed": 4
}
work_type_encoded = work_type_mapping[work_type_display]
residence_type_encoded = 0 if residence_type_display == "Rural" else 1
ever_smoked_encoded = 1 if ever_smoked_display == "Yes" else 0
smoking_status_mapping = {"Smokes": 0, "Formerly smoked": 1, "Never smoked": 2}
smoking_status_encoded = smoking_status_mapping[smoking_status_display]

# Preparing input for the model (11 features expected by the model)
input_data = np.array([[gender_encoded, hypertension_encoded, heart_disease_encoded, ever_married_encoded, 
                        work_type_encoded, residence_type_encoded, smoking_status_encoded, ever_smoked_encoded, 
                        age, avg_glucose_level, bmi]])

# Convert input_data to DataFrame for SHAP interpretability
input_df = pd.DataFrame(input_data, columns=['gender_encoded', 'hypertension_encoded', 'heart_disease_encoded', 'ever_married_encoded', 
                                             'work_type_encoded', 'Residence_type_encoded', 'smoking_status_encoded', 'ever_smoked_encoded', 
                                             'age', 'avg_glucose_level', 'bmi'])

# Button to make the prediction
if st.button("Predict Stroke Risk"):
    # Get predictions from all models
    prediction1 = logistic_regression_model.predict(input_data)
    prediction2 = naive_bayes_model.predict(input_data)
    prediction3 = svm_model.predict(input_data)
    prediction4 = decision_tree_model.predict(input_data)
    prediction5 = random_forest_model.predict(input_data)
    
    # Display the prediction result for each model
    st.write("### Predictions from each model:")
    
    st.write(f"Logistic Regression: {'High risk' if prediction1[0] > 0.5 else 'Low risk'}")
    st.write(f"Naive Bayes: {'High risk' if prediction2[0] > 0.5 else 'Low risk'}")
    st.write(f"SVM: {'High risk' if prediction3[0] > 0.5 else 'Low risk'}")
    st.write(f"Decision Tree: {'High risk' if prediction4[0] > 0.5 else 'Low risk'}")
    st.write(f"Random Forest: {'High risk' if prediction5[0] > 0.5 else 'Low risk'}")
    
    # SHAP interpretability for each model
    st.write("### Model interpretability with SHAP:")
    
    # Expander for each model's interpretability
    with st.expander("Logistic Regression SHAP Interpretation", expanded=False):
        explainer1 = shap.Explainer(logistic_regression_model.predict_proba, input_df)
        shap_values1 = explainer1(input_df)
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values1, input_df, plot_type="bar", show=False)
        st.pyplot(fig1)
        
    with st.expander("Naive Bayes SHAP Interpretation", expanded=False):
        explainer2 = shap.Explainer(naive_bayes_model.predict_proba, input_df)
        shap_values2 = explainer2(input_df)
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values2, input_df, plot_type="bar", show=False)
        st.pyplot(fig2)

    with st.expander("SVM SHAP Interpretation", expanded=False):
        explainer3 = shap.Explainer(svm_model.predict_proba, input_df)
        shap_values3 = explainer3(input_df)
        fig3, ax3 = plt.subplots()
        shap.summary_plot(shap_values3, input_df, plot_type="bar", show=False)
        st.pyplot(fig3)

    with st.expander("Decision Tree SHAP Interpretation", expanded=False):
        explainer4 = shap.Explainer(decision_tree_model.predict_proba, input_df)
        shap_values4 = explainer4(input_df)
        fig4, ax4 = plt.subplots()
        shap.summary_plot(shap_values4, input_df, plot_type="bar", show=False)
        st.pyplot(fig4)

    with st.expander("Random Forest SHAP Interpretation", expanded=False):
        explainer5 = shap.Explainer(random_forest_model.predict_proba, input_df)
        shap_values5 = explainer5(input_df)
        fig5, ax5 = plt.subplots()
        shap.summary_plot(shap_values5, input_df, plot_type="bar", show=False)
        st.pyplot(fig5)

# Power BI Dashboard
st.title("Power BI Dashboard")
st.write("Below is an interactive Power BI dashboard related to stroke analysis.")

# Embed the Power BI report
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiOTY3YzhhN2YtMjA1OC00MjNkLTljZTUtYTZhMTk4NzlmY2MyIiwidCI6ImI2NDE3Y2QwLTFmNzMtNDQ3MS05YTM5LTIwOTUzODIyYTM0YSIsImMiOjN9"
st.components.v1.iframe(power_bi_url, width=900, height=600, scrolling=True)