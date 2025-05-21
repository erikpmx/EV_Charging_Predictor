import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------------------------------------------
# 1) Carga del modelo (cached per file path) con manejo de errores
# -------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

# -------------------------------------------------------------------
# 2) Selección de modelo en la barra lateral
# -------------------------------------------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a regression model",
    ("Linear Regression", "Support Vector Machine", "Random Forest")
)

# Mapear elección de modelo al archivo .joblib correspondiente
script_dir = os.path.dirname(os.path.abspath(__file__))
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
# 3) Extraer categorías y construir mapping espacio→estaciones real
# -------------------------------------------------------------------
preproc = model.named_steps["preprocessor"]
onehot  = preproc.named_transformers_["cat"].named_steps["onehot"]
all_spaces = list(onehot.categories_[1])

# Cargar CSV con las parejas (spaceID, stationID)
csv_path = os.path.join(script_dir, "ev_charging_sessions_16000.csv")
if not os.path.exists(csv_path):
    st.error(f"❌ No se encontró el archivo CSV de sesiones:\n{csv_path}")
    st.stop()

df_map = pd.read_csv(csv_path, usecols=["spaceID","stationID"]).drop_duplicates()
space_to_stations = df_map.groupby("spaceID")["stationID"].apply(list).to_dict()

# -------------------------------------------------------------------
# 4) Layout de la app
# -------------------------------------------------------------------
st.title("⚡ EV Charging Energy Predictor")
st.markdown(f"**Using model:** {model_choice}")

st.sidebar.header("Session Details")

# Persistir fecha entre reruns
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.now().date()
date = st.sidebar.date_input(
    "Charging start date",
    value=st.session_state.start_date,
    key="start_date"
)

# Persistir hora entre reruns
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now().time().replace(second=0, microsecond=0)
time = st.sidebar.time_input(
    "Charging start time",
    value=st.session_state.start_time,
    key="start_time"
)

# Selección encadenada: Space → Station
space = st.sidebar.selectbox("Space ID", all_spaces, key="space_select")
station_options = space_to_stations.get(space, [])
station = st.sidebar.selectbox("Station ID", station_options, key="station_select")

st.sidebar.header("Predict")
predict = st.sidebar.button("Predict kWh delivered")

# -------------------------------------------------------------------
# 5) Predicción
# -------------------------------------------------------------------
if predict:
    # Construir datetime y features
    dt    = datetime.combine(date, time)
    ts    = int(dt.timestamp())
    month = dt.month
    dow   = dt.weekday()  # Monday=0

    features = {
        "connectionTime_numeric": [ts],
        "month_sin":             [np.sin(2 * np.pi * (month - 1) / 12)],
        "month_cos":             [np.cos(2 * np.pi * (month - 1) / 12)],
        "is_weekend":            [int(dow >= 5)],
        "day_sin":               [np.sin(2 * np.pi * dow / 7)],
        "day_cos":               [np.cos(2 * np.pi * dow / 7)],
        "hour_sin":              [np.sin(2 * np.pi * (dt.hour + dt.minute/60) / 24)],
        "hour_cos":              [np.cos(2 * np.pi * (dt.hour + dt.minute/60) / 24)],
        "stationID":             [station],
        "spaceID":               [space],
        "day_of_week":           [dt.strftime("%A")]
    }
    X = pd.DataFrame(features)

    # Ejecutar predicción con manejo de excepciones
    try:
        pred_kwh = model.predict(X)[0]
        st.success(f"Predicted energy delivered: **{pred_kwh:.2f} kWh**")
    except Exception as e:
        st.error(f"Error during prediction:\n{e}")
