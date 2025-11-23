import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_returns_and_time_encoding(df, feature_cols):
    for col in feature_cols:
        df[f"RET_{col}"] = np.log(df[col] / df[col].shift(1))

    df = df.dropna().reset_index(drop=True)

    df["day_of_year"] = df["Dates"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    ret_cols = [f"RET_{c}" for c in feature_cols]
    time_cols = ["sin_doy", "cos_doy"]
    return df, ret_cols, time_cols


def split_and_scale(df, ret_cols, time_cols, train_ratio=0.7, val_ratio=0.15):
    values_returns = df[ret_cols].values
    values_time = df[time_cols].values
    N = len(df)

    train_size = int(train_ratio * N)
    val_size = int(val_ratio * N)

    ret_train = values_returns[:train_size]
    ret_val = values_returns[train_size:train_size+val_size]
    ret_test = values_returns[train_size+val_size:]

    time_train = values_time[:train_size]
    time_val = values_time[train_size:train_size+val_size]
    time_test = values_time[train_size+val_size:]

    scaler = StandardScaler()
    ret_train_scaled = scaler.fit_transform(ret_train)
    ret_val_scaled   = scaler.transform(ret_val)
    ret_test_scaled  = scaler.transform(ret_test)

    return {
        "ret_train": ret_train,
        "ret_val": ret_val,
        "ret_test": ret_test,
        "time_train": time_train,
        "time_val": time_val,
        "time_test": time_test,
        "ret_train_scaled": ret_train_scaled,
        "ret_val_scaled": ret_val_scaled,
        "ret_test_scaled": ret_test_scaled,
        "train_size": train_size,
        "val_size": val_size,
    }, scaler
