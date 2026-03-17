import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    vector = np.vectorize(math.erf)
    x = np.array(x)
    y = vector(x/np.sqrt(2))
    return (1 + y) * np.array(x)/2
