from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_bilstm(model, X_train, y_train, X_val, y_val):
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr    = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=32,
        callbacks=[early, lr],
        verbose=1
    )
    return history
