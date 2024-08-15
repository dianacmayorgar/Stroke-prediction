import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder

# Inject CSS for styling adjustments to improve space usage
st.markdown("""
    <style>
    .stTextInput, .stNumberInput, .stSelectbox {
        font-size: 16px;
        width: 90%;
    }
    .stButton {
        font-size: 18px;
        width: 90%;
    }
    .css-1v0mbdj, .css-18e3th9 {  # Max layout width
        max-width: 1200px;
        margin: 0 auto;
    }
    .block-container {
        padding: 2rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained models
@st.cache_resource
def load_models():
    logistic_regression_model = joblib.load('models/Logistic_Regression.pkl')
    naive_bayes_model = joblib.load('models/Naive_Bayes.pkl')
    svm_model = joblib.load('models/SVC.pkl')
    decision_tree_model = joblib.load('models/Decision_Tree.pkl')
    random_forest_model = joblib.load('models/Random_Forest.pkl')
    return (logistic_regression_model, naive_bayes_model, svm_model, decision_tree_model, random_forest_model)

# Load all models once, caching them
(logistic_regression_model, naive_bayes_model, svm_model, decision_tree_model, random_forest_model) = load_models()

# Load dataset to calculate mean and mode
@st.cache_data
def load_data():
    return pd.read_csv('data/train_2v-clean.csv')

# Load the data
data = load_data()

# Encode categorical columns if not already encoded
def encode_columns(df):
    label_encoders = {}
    
    # Categorical columns to be encoded
    categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                           'work_type', 'Residence_type', 'ever_smoked', 'smoking_status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Apply encoding to the data
data, encoders = encode_columns(data)

# Set page title and introduction text
st.title("Stroke Prediction App")
st.write("This app predicts the likelihood of stroke based on input data.")

# Identify numerical and categorical columns
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
categorical_columns = ['gender_encoded', 'hypertension_encoded', 'heart_disease_encoded', 
                       'ever_married_encoded', 'work_type_encoded', 'Residence_type_encoded', 
                       'ever_smoked_encoded', 'smoking_status_encoded']

# Create input fields in 3 columns layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    age = st.number_input(f"Age (average {data['age'].mean():.2f})", min_value=0, max_value=100, step=1)
    gender_display = st.selectbox("Gender", ["Male", "Female"])
    hypertension_display = st.selectbox("Do you have hypertension?", ["Yes", "No"])
    heart_disease_display = st.selectbox("Do you have heart disease?", ["Yes", "No"])

with col2:
    ever_married_display = st.selectbox("Have you ever been married?", ["Yes", "No"])
    work_type_display = st.selectbox("Work Type", ["Children", "Government Job", "Never Worked", "Private", "Self-employed"])
    residence_type_display = st.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.number_input(f"Average Glucose Level (mg/dL) (average {data['avg_glucose_level'].mean():.2f})", min_value=50.0, max_value=250.0, step=0.1)

with col3:
    bmi = st.number_input(f"Body Mass Index (BMI) (average {data['bmi'].mean():.2f})", min_value=10.0, max_value=50.0, step=0.1)
    ever_smoked_display = st.selectbox("Have you ever smoked?", ["Yes", "No"])
    smoking_status_display = st.selectbox("Smoking Status", ["Smokes", "Formerly smoked", "Never smoked"])

# Map input values to their encoded versions for the model
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

# Prepare the input for the models (11 features)
input_data = np.array([[gender_encoded, hypertension_encoded, heart_disease_encoded, ever_married_encoded, 
                        work_type_encoded, residence_type_encoded, ever_smoked_encoded, 
                        smoking_status_encoded, age, avg_glucose_level, bmi]])

# Create LimeTabularExplainer to explain predictions
explainer = LimeTabularExplainer(
    data[numerical_columns + categorical_columns].values,
    feature_names=numerical_columns + categorical_columns,
    class_names=['Low Risk', 'High Risk'],
    categorical_features=list(range(8)),  # Indexes of categorical features
    discretize_continuous=True
)

# Predict button to trigger model predictions
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
    
    # Define layout for 5 plots in 3 rows
    col1, col2 = st.columns(2)  # First row with 2 columns
    col3, col4 = st.columns(2)  # Second row with 2 columns
    col5 = st.columns(1)[0]     # Third row with 1 column, need to unpack

    # Logistic Regression LIME Interpretation
    with col1:
        with st.expander("Logistic Regression LIME Interpretation", expanded=False):
            exp1 = explainer.explain_instance(input_data[0], logistic_regression_model.predict_proba)
            fig1 = exp1.as_pyplot_figure()
            st.pyplot(fig1)

    # Naive Bayes LIME Interpretation
    with col2:
        with st.expander("Naive Bayes LIME Interpretation", expanded=False):
            exp2 = explainer.explain_instance(input_data[0], naive_bayes_model.predict_proba)
            fig2 = exp2.as_pyplot_figure()
            st.pyplot(fig2)

    # SVM LIME Interpretation
    with col3:
        with st.expander("SVM LIME Interpretation", expanded=False):
            exp3 = explainer.explain_instance(input_data[0], svm_model.predict_proba)
            fig3 = exp3.as_pyplot_figure()
            st.pyplot(fig3)

    # Decision Tree LIME Interpretation
    with col4:
        with st.expander("Decision Tree LIME Interpretation", expanded=False):
            exp4 = explainer.explain_instance(input_data[0], decision_tree_model.predict_proba)
            fig4 = exp4.as_pyplot_figure()
            st.pyplot(fig4)

    # Random Forest LIME Interpretation
    with col5:
        with st.expander("Random Forest LIME Interpretation", expanded=False):
            exp5 = explainer.explain_instance(input_data[0], random_forest_model.predict_proba)
            fig5 = exp5.as_pyplot_figure()
            st.pyplot(fig5)

# Embed Power BI dashboard for interactive data visualization
st.title("Power BI Dashboard")
st.write("Below is an interactive Power BI dashboard related to stroke analysis.")

# Embed Power BI report in the Streamlit app
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiOTY3YzhhN2YtMjA1OC00MjNkLTljZTUtYTZhMTk4NzlmY2MyIiwidCI6ImI2NDE3Y2QwLTFmNzMtNDQ3MS05YTM5LTIwOTUzODIyYTM0YSIsImMiOjN9"
st.components.v1.iframe(power_bi_url, width=800, height=500, scrolling=True)