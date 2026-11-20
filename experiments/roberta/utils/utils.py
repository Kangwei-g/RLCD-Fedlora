"""Accuracy helper for clinical classification."""

import torch

def get_accuracy(y_pred, y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor ): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    y_pred = torch.argmax(y_pred, axis=1)
    return 1 / len(y_actual) * torch.sum(torch.round(y_pred.float()) == y_actual.float())
