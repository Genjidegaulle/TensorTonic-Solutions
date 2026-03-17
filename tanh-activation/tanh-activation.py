import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    num = np.exp(x) - np.exp(np.negative(x))
    denom = np.exp(x) + np.exp(np.negative(x))

    return num / denom