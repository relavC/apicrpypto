import pandas as pd

def load_price_data(path, feature_cols):
    df = pd.read_excel(path)

    # Asegurar columna de fechas
    df.rename(columns={df.columns[0]: "Dates"}, inplace=True)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df[["Dates"] + feature_cols].dropna()
    df = df.sort_values("Dates").reset_index(drop=True)
    return df
