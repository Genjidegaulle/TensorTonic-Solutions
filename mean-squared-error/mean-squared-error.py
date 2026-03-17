import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    length = len(y_pred)
    pair = zip(y_pred, y_true)
    sum = 0
    for x, y in pair:
        sum += (x-y)**2

    return sum / length
