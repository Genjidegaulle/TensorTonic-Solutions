import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x)
    return np.where(x > 0, x, 0)