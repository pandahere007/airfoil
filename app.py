
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from airfoil_prediction import model as airfoil_model
from fault_diagnosis import model as fault_model

st.title("AI/ML for Aerospace Design & Diagnostics")

# ---------- Airfoil Sound Pressure Prediction ----------
st.header("Airfoil Sound Pressure Prediction")
features = ['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_thickness']
airfoil_input = [st.slider(f, 0.0, 100.0, 20.0) for f in features]

if st.button("Predict Sound Pressure"):
    pred = airfoil_model.predict([airfoil_input])[0]
    st.success(f"Predicted Sound Pressure: {pred:.2f} dB")

    # Save to CSV for Power BI
    result_df = pd.DataFrame([airfoil_input], columns=features)
    result_df["prediction_dB"] = pred
    result_df["timestamp"] = datetime.now()
    result_df.to_csv("airfoil_predictions.csv", mode='a', index=False, header=False)
    st.info("Prediction logged to airfoil_predictions.csv")

# ---------- Component Fault Diagnosis ----------
st.header("Component Fault Diagnosis")
sensor_features = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
sensor_values = [st.slider(f, 0.0, 100.0, 50.0) for f in sensor_features]

if st.button("Diagnose Component"):
    result = fault_model.predict([sensor_values])[0]
    status = "🔴 Fault Detected" if result == 1 else "🟢 Normal"
    st.warning(status)

    # Save to CSV for Power BI
    fault_df = pd.DataFrame([sensor_values], columns=sensor_features)
    fault_df["diagnosis"] = "Fault" if result == 1 else "Normal"
    fault_df["timestamp"] = datetime.now()
    fault_df.to_csv("component_diagnosis.csv", mode='a', index=False, header=False)
    st.info("Diagnosis logged to component_diagnosis.csv")
