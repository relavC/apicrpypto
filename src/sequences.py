import numpy as np

def create_sequences(ret_scaled, time_enc, window):
    X_list, y_list = [], []
    for i in range(len(ret_scaled) - window):
        window_ret  = ret_scaled[i:i+window]
        window_time = time_enc[i:i+window]
        X = np.concatenate([window_ret, window_time], axis=1)
        X_list.append(X)
        y_list.append(ret_scaled[i+window])
    return np.array(X_list), np.array(y_list)
