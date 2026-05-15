import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape_shift(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    min_val = min(np.min(y_true), np.min(y_pred))
    a = max(0.0, 1.0 - min_val)

    return 100 * np.mean(
        np.abs((y_true - y_pred) / (y_true + a))
    )

