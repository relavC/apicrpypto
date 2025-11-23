from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def build_bilstm(window_size, n_features, n_outputs):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(window_size, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(n_outputs)
    ])
    model.compile(optimizer="adam", loss="mae", metrics=["mse"])
    return model
