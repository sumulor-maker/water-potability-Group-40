import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    return model

model = load_model()

st.title("💧 Water Potability Predictor")
st.markdown("Enter water quality parameters to predict if water is safe to drink.")

ph          = st.slider("pH", 0.0, 14.0, 7.0)
hardness    = st.number_input("Hardness", value=150.0)
solids      = st.number_input("Solids", value=20000.0)
chloramines = st.number_input("Chloramines", value=7.0)
sulfate     = st.number_input("Sulfate", value=333.0)
conductivity= st.number_input("Conductivity", value=400.0)
organic_carbon = st.number_input("Organic Carbon", value=14.0)
trihalomethanes = st.number_input("Trihalomethanes", value=66.0)
turbidity   = st.number_input("Turbidity", value=3.9)

if st.button("Predict"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Water is POTABLE (safe to drink)")
    else:
        st.error("❌ Water is NOT POTABLE (unsafe)")
