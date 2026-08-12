import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    N = len(y_true)

    CE = -1 / N * np.sum(np.log(y_pred[np.arange(N), y_true]))

    return CE