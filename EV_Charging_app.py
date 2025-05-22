import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------------
# Clase custom para deserializar el pipeline entrenado
# -------------------------------------------------------------------
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column="connectionTime",
                 fmt="%a, %d %b %Y %H:%M:%S GMT"):
        self.datetime_column = datetime_column
        self.fmt = fmt

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(
            X[self.datetime_column],
            format=self.fmt,
            errors='coerce',
            utc=True
        )
        X['connectionTime_numeric'] = dt.view('int64') // 10**9
        month = dt.dt.month
        dow   = dt.dt.weekday
        X['month_sin']  = np.sin(2 * np.pi * (month - 1) / 12)
        X['month_cos']  = np.cos(2 * np.pi * (month - 1) / 12)
        X['is_weekend'] = (dow >= 5).astype(int)
        X['day_sin']    = np.sin(2 * np.pi * dow / 7)
        X['day_cos']    = np.cos(2 * np.pi * dow / 7)
        hour = dt.dt.hour + dt.dt.minute / 60
        X['hour_sin']   = np.sin(2 * np.pi * hour / 24)
        X['hour_cos']   = np.cos(2 * np.pi * hour / 24)
        X['day_of_week'] = dt.dt.day_name()
        return X

# -------------------------------------------------------------------
# 1) Carga del modelo
# -------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model:\n{e}")
        st.stop()

# -------------------------------------------------------------------
# 2) Selección de modelo
# -------------------------------------------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a regression model",
    ("Linear Regression", "Support Vector Machine", "Random Forest", "XGBoost")
)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = {
    "Linear Regression":      "EV_Charging_LR.joblib",
    "Support Vector Machine": "EV_Charging_SVM.joblib",
    "Random Forest":          "EV_Charging_RF.joblib",
    "XGBoost":                "EV_Charging_XGBoost.joblib"
}[model_choice]

model_path = os.path.join(script_dir, model_file)
if not os.path.exists(model_path):
    st.error(f"❌ No se encontró el archivo de modelo:\n{model_path}")
    st.stop()

model = load_model(model_path)

# -------------------------------------------------------------------
# 3) Mapeo espacio → estaciones
# -------------------------------------------------------------------
steps = list(model.steps)
dt_feats, preproc, regressor = [step[1] for step in steps]

onehot = preproc.named_transformers_["cat"].named_steps["onehot"]
all_spaces = list(onehot.categories_[1])

csv_path = os.path.join(script_dir, "ev_charging_sessions_16000.csv")
if not os.path.exists(csv_path):
    st.error(f"❌ No se encontró el archivo CSV de sesiones:\n{csv_path}")
    st.stop()

df_map = pd.read_csv(csv_path, usecols=["spaceID","stationID"]).drop_duplicates()
space_to_stations = df_map.groupby("spaceID")["stationID"].apply(list).to_dict()

# -------------------------------------------------------------------
# 4) Diseño de la app
# -------------------------------------------------------------------
st.title("⚡ EV Charging Energy Predictor")

# --- Incorporación del GIF ---
gif_path = os.path.join(script_dir, "ev_shuffle.gif")
if os.path.exists(gif_path):
    st.image(
        gif_path,
        caption="Simulación de carga EV",
        use_container_width=True
    )
else:
    st.warning(f"GIF no encontrado en: {gif_path}")

st.markdown(
    """
    Esta interfaz permite predecir la cantidad de energía (kWh) que un vehículo eléctrico utilizará durante una sesión de carga en una estación ubicada en California.

    Para realizar la predicción, es necesario que selecciones:
    - El modelo de regresión que deseas emplear  
    - La fecha y hora de inicio de la carga  
    - Y la ubicación específica de la estación de carga en California (Space ID y Station ID)
    """
)

st.sidebar.header("Session Details")
# Ahora pasamos directamente el valor por defecto al widget y dejamos que Streamlit gestione el estado
date = st.sidebar.date_input(
    "Charging start date",
    datetime.now().date(),
    key="start_date"
)
time = st.sidebar.time_input(
    "Charging start time",
    datetime.now().time().replace(second=0, microsecond=0),
    key="start_time"
)

space = st.sidebar.selectbox("Space ID", all_spaces, key="space_select")
station = st.sidebar.selectbox(
    "Station ID",
    space_to_stations.get(space, []),
    key="station_select"
)

st.sidebar.header("Predict")
predict = st.sidebar.button("Predict kWh delivered")

# -------------------------------------------------------------------
# 5) Predicción
# -------------------------------------------------------------------
if predict:
    dt = datetime.combine(date, time)
    connection_str = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    X_input = pd.DataFrame({
        "connectionTime": [connection_str],
        "spaceID":        [space],
        "stationID":      [station]
    })

    try:
        if model_choice == "XGBoost":
            X_dt  = dt_feats.transform(X_input)
            X_pre = preproc.transform(X_dt)
            dmat    = xgb.DMatrix(X_pre, feature_names=preproc.get_feature_names_out())
            booster = regressor.get_booster()
            pred_kwh = booster.predict(dmat)[0]
        else:
            pred_kwh = model.predict(X_input)[0]

        st.success(f"Predicted energy delivered: **{pred_kwh:.2f} kWh**")
    except Exception as e:
        st.error(f"Error durante la predicción:\n{e}")