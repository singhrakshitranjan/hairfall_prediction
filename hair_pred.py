# -- coding: utf-8 --
"""
Created on Sat Feb 22 20:20:05 2025

@author: acojh
"""

import streamlit as st
import pickle
import pandas as pd

st.html("<h3>Take the 2-Minute Hair Fall Quiz Get Your Personalized Risk Level!</h3>")

col1, col2, = st.columns(2)

# getting user input
q1 = col1.selectbox("1.What is your age group?",["Below 18", "18-25", "26-40", "41-55","Above 55"])
q2 = col2.selectbox("2.What is your gender?",["Female", "Male"])
q3 = col1.selectbox("3.Do you have a family history of hair loss?",["No", "Yes"])
q4 = col2.selectbox("4.How many hours of sleep do you get per night?",["Less than 5 hours", "5-6 hours","7-9 hours","Above 9 hours"])


q5 = col1.selectbox("5.Do you feel stressful?",["Not at all", "Rarely", "Sometimes", "Often","Always"])
q6 = col2.selectbox("6.How would you describe your diet?",["Healthy (rich in proteins, vitamins, and nutrients)", "Moderate (balanced but not consistent)", "Poor (junk food, irregular meals)"])
q7 = col1.selectbox("7.Do you smoke and/or consume alcohol?",["Not at all", "Occasionally", "Regularly", "Daily"])
q8 = col2.selectbox("8.Have you ever been diagnosed with a hormonal imbalance (e.g., PCOS, thyroid disorder)?",["Yes", "No"])

q9 = col1.selectbox("9.THave you experienced sudden weight gain or loss in the last 6 months?",["No significant change", "Gained weight unexpectedly", "Lost weight unexpectedly"])
q10 = col2.selectbox("10.Do you experience irregular menstrual cycles (for females) or signs of low testosterone (for males, e.g., fatigue, muscle loss)?",["No everything is normal", "Occasionally", "Frequently"])
q11 = col1.selectbox("11.How would you describe your hair shedding rate?",["No shedding", "Mild shedding", "Moderate shedding", "Severe shedding"])
q12 = col2.selectbox("12.How would you describe your scalp health?",["Healthy", "Dry& Flaky", "Oily", "Dandruff-prone"])

q13 = col1.selectbox("13.How oily does your scalp feel?",["Normal", "Dry", "Oily"])

btn=st.button('Predict', type="primary")

df_pred = pd.DataFrame([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13]],columns= ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13'])

def a(data):
    result = 0
    if(data=='Below 18'):
        result = 0
    elif(data=='18-25'or 'Above 55'):
        result = 1
    elif(data=='41-55'):
        result = 2
    elif(data=='26-40'):
        result = 3
    return(result)

def b(data):
    result = 0
    if(data=='Male'):
        result = 1
    return(result)

def c(data):
    result = 0
    if(data=='Yes'):
        result = 1
    return(result)

def d(data):
    result = 0
    if(data=='Above 9 hours'):
        result = 1
    elif(data=='5-6 hours'):
        result = 2
    elif(data=='Less than 5 hours'):
        result = 3
    return(result)

def e(data):
    result = 0
    if(data=='Sometimes'):
        result = 1
    elif(data=='Often'):
        result = 2
    elif(data=='Always'):
        result = 3
    return(result)

def f(data):
    result = 0
    if(data=='Moderate (balanced but not consistent)'):
        result = 1
    elif(data=='Poor (junk food, irregular meals)'):
        result = 2
    return(result)

def g(data):
    result = 0
    if(data=='Occasionally'):
        result = 1
    elif(data=='Regularly'):
        result = 2
    elif(data=='Daily'):
        result = 3
    return(result)

def h(data):
    result = 0
    if(data=='Yes'):
        result = 1
    return(result)

def i(data):
    result = 0
    if(data=='Gained weight unexpectedly' or 'Lost weight unexpectedly'):
        result = 1
    return(result)

def j(data):
    result = 0
    if(data=='Occasionally'):
        result = 1
    elif(data=='Frequently'):
        result = 2
    return(result)

def k(data):
    result = 0
    if(data=='Mild shedding'):
        result = 1
    elif(data=='Moderate shedding'):
        result = 2
    elif(data=='Severe shedding'):
        result = 3
    return(result)

def l(data):
    result = 0
    if(data=='	Dry & flaky '):
        result = 1
    elif(data=='Oily'):
        result = 2
    elif(data=='Dandruff-prone'):
        result = 3
    return(result)

def n(data):
    result = 0
    if(data=='Dry'):
        result = 1
    elif(data=='Oily'):
        result = 2
    return(result)



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

#st.dataframe(df_pred)
#checking convertion

# Load the saved model for prediction
with open('svm_model_pkl', 'rb') as file:
    svm_model = pickle.load(file)

prediction = svm_model.predict(df_pred)
    
if btn:
  st.write("<b>Your Risk Level: </b>", prediction, unsafe_allow_html=True)
