import numpy as np
import pandas as pd
from datetime import timedelta

def generate_synthetic(df, feature_cols, ret_cols, time_cols,
                       model, scaler, residuals, window_size, days):

    ret_all  = df[ret_cols].values
    time_all = df[time_cols].values
    dates    = df["Dates"].values
    prices   = df[feature_cols].values

    ret_scaled = scaler.transform(ret_all)
    win_ret  = ret_scaled[-window_size:].copy()
    win_time = time_all[-window_size:].copy()

    last_price = prices[-1].copy()
    last_date = pd.to_datetime(dates[-1])

    syn_returns = []
    syn_prices = []
    syn_dates = []

    for _ in range(days):
        next_date = last_date + timedelta(days=1)
        doy = next_date.timetuple().tm_yday
        sin_doy = np.sin(2*np.pi*doy/365)
        cos_doy = np.cos(2*np.pi*doy/365)

        X = np.concatenate([win_ret, win_time], axis=1)
        X = X[np.newaxis, :, :]
        pred_scaled = model.predict(X, verbose=0)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(1,-1))[0]

        noise = residuals[np.random.randint(0,len(residuals))]
        ret_noisy = pred + noise

        next_price = last_price*np.exp(ret_noisy)

        syn_returns.append(ret_noisy)
        syn_prices.append(next_price)
        syn_dates.append(next_date)

        win_ret = np.vstack([win_ret[1:], scaler.transform(ret_noisy.reshape(1,-1))[0]])
        win_time = np.vstack([win_time[1:], [sin_doy, cos_doy]])

        last_price = next_price
        last_date = next_date

    df_synth = pd.DataFrame(syn_prices, columns=feature_cols)
    df_synth.insert(0, "Dates", syn_dates)
    return df_synth
