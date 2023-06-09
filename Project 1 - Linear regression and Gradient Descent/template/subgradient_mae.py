import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # compute subgradient gradient vector for MAE
    N = len(y)
    e = y - tx.dot(w)
    return (1/N) * np.sum(np.sign(e)[:, np.newaxis] * tx, axis=0)
