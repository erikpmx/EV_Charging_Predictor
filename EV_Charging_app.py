# File: ev_charging_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from PIL import Image

# -------------------------------------------------------------------
# 0) Mostrar GIF animado en la interfaz
# -------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
gif_path = os.path.join(script_dir, "ev_shuffle.gif")
if os.path.exists(gif_path):
    st.image(gif_path, use_column_width=True)
else:
    st.warning(f"⚠️ No se encontró el GIF en:\n{gif_path}")

# -------------------------------------------------------------------
# 1) Carga del modelo (cached per file path) con manejo de errores
# -------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model:\n{e}")
        st.stop()

# -------------------------------------------------------------------
# 2) Selección de modelo en la barra lateral
# -------------------------------------------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a regression model",
    ("Linear Regression", "Support Vector Machine", "Random Forest")
)

# Mapeo de elección → archivo .joblib
if model_choice == "Linear Regression":
    model_file = "EV_Charging_LR.joblib"
elif model_choice == "Support Vector Machine":
    model_file = "EV_Charging_SVM.joblib"
else:
    model_file = "EV_Charging_RF.joblib"

model_path = os.path.join(script_dir, model_file)
if not os.path.exists(model_path):
    st.error(f"❌ No se encontró el archivo de modelo:\n{model_path}")
    st.stop()

model = load_model(model_path)

# -------------------------------------------------------------------
# 3) Inputs de usuario
# -------------------------------------------------------------------
st.header("Predicción de consumo eléctrico de EV")
connection_time = st.datetime_input("Fecha y hora de inicio de carga")
space_id = st.text_input("Space ID")
station_id = st.text_input("Station ID")

# -------------------------------------------------------------------
# 4) Preprocesamiento y predicción
# -------------------------------------------------------------------
if st.button("Calcular predicción"):
    # Aquí iría tu pipeline de transformación de features...
    # Por ejemplo:
    df = pd.DataFrame({
        "connectionTime": [connection_time.strftime("%a, %d %b %Y %H:%M:%S GMT")],
        "SpaceID": [space_id],
        "StationID": [station_id],
    })
    try:
        # Asumiendo que tu pipeline está encapsulado en el modelo
        pred = model.predict(df)[0]
        st.success(f"🔋 Consumo estimado: {pred:.2f} kWh")
    except Exception as e:
        st.error(f"❌ Error durante la predicción:\n{e}")