import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    pair = zip(x1, x2)
    x_dot = sum([x*y for x,y in pair])

    norm1 = np.sqrt(sum([x**2 for x in x1]))
    norm2 = np.sqrt(sum([x**2 for x in x2]))

    cos = x_dot / (norm1 * norm2)

    if label == 1:
        return 1 - cos
    else:
        return max(0, cos - margin)