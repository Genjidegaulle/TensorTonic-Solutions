import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    error = [x-y for x,y in zip(y_true, y_pred)]

    loss = 0
    for e in error:
        if np.abs(e) <= delta:
            loss += (e**2) / 2
        else:
            loss += delta * (np.abs(e) - delta/2)

    return loss/len(y_true)
        