# -- coding: utf-8 --
"""
Created on Sat Feb 22 20:20:05 2025

@author: acojh
"""

import streamlit as st
import pickle
import pandas as pd

# Apply custom background color and center-aligned button styling
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff; /* Light pastel blue */
        }
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        h3 {
            text-align: center;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Replaced st.html (which doesn't exist) with st.markdown
st.markdown("<h3>Take the 2-Minute Hair Fall Quiz Get Your Personalized Risk Level!</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Getting user input
q1 = col1.selectbox("What is your age group?", ["Below 18", "18-25", "26-40", "41-55", "Above 55"])
q2 = col2.selectbox("What is your gender?", ["Female", "Male"])
q3 = col1.selectbox("Do you have a family history of hair loss?", ["No", "Yes"])
q4 = col2.selectbox("How many hours of sleep do you get per night?", ["Less than 5 hours", "5-6 hours", "7-9 hours", "Above 9 hours"])
q5 = col1.selectbox("Do you feel stressful?", ["Not at all", "Rarely", "Sometimes", "Often", "Always"])
q6 = col2.selectbox("How would you describe your diet?", ["Healthy (rich in proteins, vitamins, and nutrients)", "Moderate (balanced but not consistent)", "Poor (junk food, irregular meals)"])
q7 = col1.selectbox("Do you smoke and/or consume alcohol?", ["Not at all", "Occasionally", "Regularly", "Daily"])
q8 = col2.selectbox("Have you ever been diagnosed with a hormonal imbalance?", ["Yes", "No"])
q9 = col1.selectbox("Have you experienced sudden weight gain or loss in the last 6 months?", ["No significant change", "Gained weight unexpectedly", "Lost weight unexpectedly"])
q10 = col2.selectbox("Do you experience irregular menstrual cycles (for females) or signs of low testosterone (for males)?", ["No everything is normal", "Occasionally", "Frequently"])
q11 = col1.selectbox("How would you describe your hair shedding rate?", ["No shedding", "Mild shedding", "Moderate shedding", "Severe shedding"])
q12 = col2.selectbox("How would you describe your scalp health?", ["Healthy", "Dry& Flaky", "Oily", "Dandruff-prone"])
q13 = col1.selectbox("How oily does your scalp feel?", ["Normal", "Dry", "Oily"])

btn = st.button('Predict', type="primary")

df_pred = pd.DataFrame([[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13]],
                       columns=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13'])

# Mapping functions
def a(data):
    return {'Below 18': 0, '18-25': 1, '26-40': 3, '41-55': 2, 'Above 55': 1}.get(data, 0)

def b(data):
    return 1 if data == 'Male' else 0

def c(data):
    return 1 if data == 'Yes' else 0

def d(data):
    return {'Above 9 hours': 1, '5-6 hours': 2, 'Less than 5 hours': 3}.get(data, 0)

def e(data):
    return {'Sometimes': 1, 'Often': 2, 'Always': 3}.get(data, 0)

def f(data):
    return {'Moderate (balanced but not consistent)': 1, 'Poor (junk food, irregular meals)': 2}.get(data, 0)

def g(data):
    return {'Occasionally': 1, 'Regularly': 2, 'Daily': 3}.get(data, 0)

def h(data):
    return 1 if data == 'Yes' else 0

def i(data):
    return 1 if data in ['Gained weight unexpectedly', 'Lost weight unexpectedly'] else 0

def j(data):
    return {'Occasionally': 1, 'Frequently': 2}.get(data, 0)

def k(data):
    return {'Mild shedding': 1, 'Moderate shedding': 2, 'Severe shedding': 3}.get(data, 0)

def l(data):
    return {'Dry& Flaky': 1, 'Oily': 2, 'Dandruff-prone': 3}.get(data, 0)

def n(data):
    return {'Dry': 1, 'Oily': 2}.get(data, 0)

# Apply mapping
df_pred['q1'] = df_pred['q1'].apply(a)
df_pred['q2'] = df_pred['q2'].apply(b)
df_pred['q3'] = df_pred['q3'].apply(c)
df_pred['q4'] = df_pred['q4'].apply(d)
df_pred['q5'] = df_pred['q5'].apply(e)
df_pred['q6'] = df_pred['q6'].apply(f)
df_pred['q7'] = df_pred['q7'].apply(g)
df_pred['q8'] = df_pred['q8'].apply(h)
df_pred['q9'] = df_pred['q9'].apply(i)
df_pred['q10'] = df_pred['q10'].apply(j)
df_pred['q11'] = df_pred['q11'].apply(k)
df_pred['q12'] = df_pred['q12'].apply(l)
df_pred['q13'] = df_pred['q13'].apply(n)

st.dataframe(df_pred)

# Load the saved model
with open('svm_model_pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Prediction
if btn:
    prediction = svm_model.predict(df_pred)
    st.markdown(f"<b>Your Risk Level:</b> {prediction[0]}", unsafe_allow_html=True)
