import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.array(x)
    return list(np.where(np.array(x) > 0, x, alpha * (np.exp(x) - 1)))