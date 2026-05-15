import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape_shift(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    min_val = min(np.min(y_true), np.min(y_pred))
    a = max(0.0, 1.0 - min_val)

    return 100 * np.mean(
        np.abs((y_true - y_pred) / (y_true + a))
    )

