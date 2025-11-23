from src.data_loader import load_price_data
from src.preprocessing import add_returns_and_time_encoding, split_and_scale
from src.sequences import create_sequences
from src.model_bilstm import build_bilstm
from src.train_bilstm import train_bilstm
from src.eval_bilstm import evaluate_bilstm
from src.generate_synthetic import generate_synthetic
import matplotlib.pyplot as plt
import os, pickle

# CONFIG
FEATURE_COLS = ["PX_LAST", "PX_LOW", "PX_HIGH"]
WINDOW = 15
N_FUTURE = 60

# 1️⃣ Cargar datos
df = load_price_data("./data/Precios Historicos.xlsx", FEATURE_COLS)

# 2️⃣ Retornos + codificación temporal
df, ret_cols, time_cols = add_returns_and_time_encoding(df, FEATURE_COLS)

# 3️⃣ Split + scaler
splits, scaler = split_and_scale(df, ret_cols, time_cols)
X_train, y_train = create_sequences(splits["ret_train_scaled"], splits["time_train"], WINDOW)
X_val, y_val     = create_sequences(splits["ret_val_scaled"], splits["time_val"], WINDOW)
X_test, y_test   = create_sequences(splits["ret_test_scaled"], splits["time_test"], WINDOW)

# 4️⃣ Modelo
model = build_bilstm(WINDOW, X_train.shape[2], y_train.shape[1])
history = train_bilstm(model, X_train, y_train, X_val, y_val)

# 5️⃣ Evaluación + reconstrucción
rec_real, rec_pred, rec_boot = evaluate_bilstm(
    df, FEATURE_COLS, model, scaler, X_test, y_test,
    splits["train_size"], splits["val_size"]
)

# 6️⃣ Generación sintética
df_synth = generate_synthetic(
    df, FEATURE_COLS, ret_cols, time_cols,
    model, scaler,
    rec_real - rec_pred,
    WINDOW,
    N_FUTURE
)

print(df_synth.head())

# 7️⃣ Guardar modelos
os.makedirs("./models", exist_ok=True)
model.save("./models/bilstm_returns_sensible.keras")
with open("./models/scaler_returns.pkl", "wb") as f:
    pickle.dump(scaler, f)

df_synth.to_csv("./models/synthetic_prices.csv", index=False)
print("Todo guardado correctamente.")
