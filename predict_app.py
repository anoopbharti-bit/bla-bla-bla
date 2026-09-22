"""
Landslide Prediction - Interactive Interface
===============================================
Trained model (landslide_model.joblib) ko load karke ek simple web
interface deta hai jisme feature values daal ke prediction le sakte ho.

SETUP:
    pip install streamlit joblib pandas

RUN:
    streamlit run predict_app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="Uttarakhand Landslide Predictor",
    page_icon="⛰️",
    layout="centered"
)

st.title("⛰️ Uttarakhand Landslide Prediction")
st.write(
    "put the features of a particular point , and it will predict the landslide"
)

# ---------- Model load karo ----------
@st.cache_resource
def load_model():
    model = joblib.load("landslide_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return model, feature_cols

model, feature_cols = load_model()

st.divider()
st.subheader("Feature Values Daalo")

# ---------- Input fields (2 columns me clean layout) ----------
col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input("Elevation (meters)", min_value=0.0, max_value=8000.0, value=2200.0)
    slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=30.0)
    aspect = st.number_input("Aspect (degrees, 0-360)", min_value=0.0, max_value=360.0, value=180.0)
    ndvi = st.number_input("NDVI (-1 to 1)", min_value=-1.0, max_value=1.0, value=0.5)
    landcover = st.number_input("Land Cover Class (ESA WorldCover code)", min_value=0, max_value=100, value=10)

with col2:
    dist_to_river = st.number_input("Distance to River (meters)", min_value=0.0, value=1000.0)
    rain_15day = st.number_input("Rainfall - 15 Day Antecedent (mm)", min_value=0.0, value=150.0)
    rain_annual = st.number_input("Rainfall - Annual Sum, 3yr (mm)", min_value=0.0, value=4300.0)
    rain_monsoon = st.number_input("Rainfall - Monsoon Sum, 3yr (mm)", min_value=0.0, value=3300.0)

st.divider()

# ---------- Predict button ----------
if st.button("🔍 Predict Landslide Risk", use_container_width=True):
    input_data = pd.DataFrame([{
        'NDVI': ndvi,
        'aspect': aspect,
        'dist_to_river': dist_to_river,
        'elevation': elevation,
        'landcover': landcover,
        'rainfall_15day_antecedent': rain_15day,
        'rainfall_annual_sum': rain_annual,
        'rainfall_monsoon_sum': rain_monsoon,
        'slope': slope,
    }])

    # Feature order training ke time jaisa hi rakhna zaroori hai
    input_data = input_data[feature_cols]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Landslide hone ka chance hai (Probability: {probability:.1%})")
    else:
        st.success(f"✅ LOW RISK — Landslide hone ka chance kam hai (Probability: {probability:.1%})")

    st.progress(float(probability))

st.divider()
st.caption(
    "Model: Random Forest | Trained on Uttarakhand landslide inventory + "
    "topographic, climatic, and vegetation features (Google Earth Engine)."
)
