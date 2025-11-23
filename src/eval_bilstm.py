import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

def evaluate_bilstm(df, feature_cols, model, scaler, X_test, y_test, t0, tv):
    pred_scaled = model.predict(X_test)
    y_true = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(pred_scaled)

    # Métricas
    print("\n=== Métricas en TEST ===")
    for i, col in enumerate(feature_cols):
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"{col} → RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # Retornos reales vs predichos
    plt.figure(figsize=(12,6))
    for i, col in enumerate(feature_cols):
        plt.subplot(3,1,i+1)
        plt.plot(y_true[:,i])
        plt.plot(y_pred[:,i])
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Reconstrucción con ruido bootstrap
    residuals = y_true - y_pred

    base_price = df[feature_cols].iloc[t0+tv-1].values
    rec_real, rec_pred, rec_boot = [], [], []
    p_r = base_price.copy()
    p_p = base_price.copy()
    p_b = base_price.copy()

    for t in range(len(y_true)):
        p_r = p_r*np.exp(y_true[t])
        p_p = p_p*np.exp(y_pred[t])

        noise = residuals[np.random.randint(0,len(residuals))]
        p_b = p_b*np.exp(y_pred[t] + noise)

        rec_real.append(p_r.copy())
        rec_pred.append(p_p.copy())
        rec_boot.append(p_b.copy())

    return np.array(rec_real), np.array(rec_pred), np.array(rec_boot)
