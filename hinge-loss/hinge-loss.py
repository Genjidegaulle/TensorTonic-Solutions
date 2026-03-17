import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    pair = zip(y_true, y_score)

    loss_list = [max(0, margin - x*y) for x,y in pair]

    if reduction == "mean":
        return sum(loss_list)/len(loss_list)
    else:
        return sum(loss_list)