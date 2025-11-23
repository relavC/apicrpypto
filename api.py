# api.py
"""
Servicio FastAPI para generar precios sintéticos de BTC usando
el modelo BiLSTM entrenado sobre retornos + tiempo + ruido bootstrap.

Arranque:
    uvicorn api:app --reload
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tensorflow.keras.models import load_model

# -----------------------------
# Configuración básica
# -----------------------------

DATA_PATH = "./data/Precios Historicos.xlsx"   # ya sin tilde
MODEL_PATH = "./models/bilstm_returns_sensible.keras"
SCALER_PATH = "./models/scaler_returns.pkl"

FEATURE_COLS = ["PX_LAST", "PX_LOW", "PX_HIGH"]
RET_COLS = [f"RET_{c}" for c in FEATURE_COLS]
TIME_COLS = ["sin_doy", "cos_doy"]

WINDOW_SIZE = 15
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="BTC BiLSTM Synthetic Generator API",
    version="1.0.0",
    description="Servicio para generar precios sintéticos de BTC con BiLSTM + retornos + estacionalidad + ruido bootstrap."
)

# -----------------------------
# Pydantic models (schemas)
# -----------------------------

class SyntheticPoint(BaseModel):
    date: datetime
    PX_LAST: float
    PX_LOW: float
    PX_HIGH: float


class SyntheticResponse(BaseModel):
    n_future: int
    points: List[SyntheticPoint]


# -----------------------------
# Helpers de preprocesamiento
# -----------------------------

def load_price_data(path: str, feature_cols: list) -> pd.DataFrame:
    """Carga Excel, asegura columna Dates y ordena por fecha."""
    df = pd.read_excel(path)
    df.rename(columns={df.columns[0]: "Dates"}, inplace=True)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df[["Dates"] + feature_cols].dropna()
    df = df.sort_values("Dates").reset_index(drop=True)
    return df


def add_returns_and_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Añade retornos logarítmicos + codificación sen/cos del día del año."""
    for col in FEATURE_COLS:
        df[f"RET_{col}"] = np.log(df[col] / df[col].shift(1))

    df = df.dropna().reset_index(drop=True)

    df["day_of_year"] = df["Dates"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    return df


def create_sequences_with_time(ret_scaled: np.ndarray,
                               time_enc: np.ndarray,
                               window: int):
    """
    Crea secuencias:
      X: [retornos escalados + tiempo]  shape -> (N_seq, window, 5)
      y: retornos escalados objetivo     shape -> (N_seq, 3)
    """
    X_list, y_list = [], []
    for i in range(len(ret_scaled) - window):
        window_ret = ret_scaled[i:i + window, :]      # (window, 3)
        window_time = time_enc[i:i + window, :]       # (window, 2)
        window_feat = np.concatenate([window_ret, window_time], axis=1)
        X_list.append(window_feat)
        y_list.append(ret_scaled[i + window])         # retorno siguiente
    return np.array(X_list), np.array(y_list)


def generate_synthetic_returns_with_noise(
    df: pd.DataFrame,
    model,
    scaler_ret,
    residuals: np.ndarray,
    window_size: int,
    n_future: int
):
    """
    Genera n_future días de retornos + precios sintéticos usando:
      - modelo BiLSTM sobre retornos + tiempo
      - ruido bootstrap tomado de residuals
      - codificación temporal sen/cos para fechas futuras
    """
    ret_all = df[RET_COLS].values
    time_all = df[TIME_COLS].values
    dates_all = df["Dates"].values
    prices_all = df[FEATURE_COLS].values

    # Escalar retornos con scaler ya entrenado
    ret_all_scaled = scaler_ret.transform(ret_all)

    # Ventana inicial = últimos 'window_size' puntos
    window_ret_scaled = ret_all_scaled[-window_size:, :].copy()
    window_time_enc = time_all[-window_size:, :].copy()

    last_price = prices_all[-1, :].copy()
    last_date = pd.to_datetime(dates_all[-1])

    synthetic_returns = []
    synthetic_prices = []
    synthetic_dates = []

    for _ in range(n_future):
        # Fecha futura
        next_date = last_date + timedelta(days=1)
        day_of_year = next_date.timetuple().tm_yday
        sin_doy = np.sin(2 * np.pi * day_of_year / 365.0)
        cos_doy = np.cos(2 * np.pi * day_of_year / 365.0)

        # Construir entrada (1, window, 5)
        X_window = np.concatenate([window_ret_scaled, window_time_enc], axis=1)
        X_window_batch = np.expand_dims(X_window, axis=0)

        # Predicción de retornos escalados
        pred_scaled = model.predict(X_window_batch, verbose=0)[0]
        pred_ret = scaler_ret.inverse_transform(pred_scaled.reshape(1, -1))[0]

        # Ruido bootstrap de residuos reales
        idx = np.random.randint(0, residuals.shape[0])
        noise = residuals[idx]
        ret_noisy = pred_ret + noise

        # Reconstrucción de precio
        next_price = last_price * np.exp(ret_noisy)

        # Guardar
        synthetic_returns.append(ret_noisy)
        synthetic_prices.append(next_price)
        synthetic_dates.append(next_date)

        # Actualizar ventana (para el siguiente paso)
        ret_noisy_scaled = scaler_ret.transform(ret_noisy.reshape(1, -1))[0]
        window_ret_scaled = np.vstack([window_ret_scaled[1:], ret_noisy_scaled])

        next_time_enc = np.array([[sin_doy, cos_doy]])
        window_time_enc = np.vstack([window_time_enc[1:], next_time_enc])

        last_price = next_price
        last_date = next_date

    synthetic_returns = np.array(synthetic_returns)
    synthetic_prices = np.array(synthetic_prices)
    synthetic_dates = np.array(synthetic_dates)

    return synthetic_dates, synthetic_prices, synthetic_returns


# -----------------------------
# Contexto global del modelo
# -----------------------------

model = None
scaler_ret = None
df_full = None
df_rt = None
residuals_all = None


def init_context():
    """
    Se ejecuta una vez al arrancar el servidor:
      - carga datos
      - añade retornos y codificación temporal
      - carga modelo y scaler
      - calcula residuos train+val (para bootstrap)
    """
    global model, scaler_ret, df_full, df_rt, residuals_all

    # 1) Cargar datos de precios
    df_full = load_price_data(DATA_PATH, FEATURE_COLS)
    df_rt = add_returns_and_time_encoding(df_full.copy())

    values_returns = df_rt[RET_COLS].values
    values_time = df_rt[TIME_COLS].values

    N = len(df_rt)
    train_size = int(N * TRAIN_RATIO)
    val_size = int(N * VAL_RATIO)

    ret_train = values_returns[:train_size]
    ret_val = values_returns[train_size:train_size + val_size]

    time_train = values_time[:train_size]
    time_val = values_time[train_size:train_size + val_size]

    # 2) Cargar scaler de retornos ya entrenado
    with open(SCALER_PATH, "rb") as f:
        scaler_ret_local = pickle.load(f)

    # 3) Escalar retornos con ese scaler (no re-entrenar)
    ret_train_scaled = scaler_ret_local.transform(ret_train)
    ret_val_scaled = scaler_ret_local.transform(ret_val)

    # 4) Crear secuencias para train y val
    X_train, y_train = create_sequences_with_time(ret_train_scaled,
                                                  time_train,
                                                  WINDOW_SIZE)
    X_val, y_val = create_sequences_with_time(ret_val_scaled,
                                              time_val,
                                              WINDOW_SIZE)

    # 5) Cargar modelo BiLSTM
    model_local = load_model(MODEL_PATH)

    # 6) Predicciones en train y val para obtener residuos
    y_train_pred_scaled = model_local.predict(X_train, verbose=0)
    y_val_pred_scaled = model_local.predict(X_val, verbose=0)

    y_train_true = scaler_ret_local.inverse_transform(y_train)
    y_val_true = scaler_ret_local.inverse_transform(y_val)

    y_train_pred = scaler_ret_local.inverse_transform(y_train_pred_scaled)
    y_val_pred = scaler_ret_local.inverse_transform(y_val_pred_scaled)

    residuals_train = y_train_true - y_train_pred
    residuals_val = y_val_true - y_val_pred
    residuals = np.vstack([residuals_train, residuals_val])

    # Asignar a globales
    model = model_local
    scaler_ret = scaler_ret_local
    residuals_all = residuals

    print(">>> Contexto inicializado")
    print("   Datos:", df_rt.shape)
    print("   Residuales:", residuals_all.shape)


# -----------------------------
# Eventos de inicio
# -----------------------------

@app.on_event("startup")
def startup_event():
    try:
        init_context()
    except Exception as e:
        # Esto ayuda a ver errores al arrancar uvicorn
        print("Error inicializando contexto:", e)
        raise


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "detail": "BTC BiLSTM API running"}


@app.get("/synthetic", response_model=SyntheticResponse)
def generate_synthetic(n_future: int = 60):
    """
    Genera n_future días sintéticos de BTC.
    Usa el modelo BiLSTM entrenado + ruido bootstrap de residuos.
    """
    if n_future <= 0 or n_future > 365:
        raise HTTPException(
            status_code=400,
            detail="n_future debe estar entre 1 y 365 días."
        )

    if model is None or scaler_ret is None or df_rt is None or residuals_all is None:
        raise HTTPException(
            status_code=500,
            detail="El contexto del modelo no está inicializado correctamente."
        )

    syn_dates, syn_prices, _ = generate_synthetic_returns_with_noise(
        df=df_rt,
        model=model,
        scaler_ret=scaler_ret,
        residuals=residuals_all,
        window_size=WINDOW_SIZE,
        n_future=n_future
    )

    points = []
    for d, price_vec in zip(syn_dates, syn_prices):
        points.append(SyntheticPoint(
            date=pd.to_datetime(d),
            PX_LAST=float(price_vec[0]),
            PX_LOW=float(price_vec[1]),
            PX_HIGH=float(price_vec[2]),
        ))

    return SyntheticResponse(
        n_future=n_future,
        points=points
    )
