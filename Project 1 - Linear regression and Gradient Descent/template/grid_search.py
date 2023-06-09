# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


## Paste your implementation of grid_search
def grid_search(y, tx, w0, w1):
    losses = np.zeros((len(w0), len(w1)))
    for i, w0 in enumerate(w0):
        for j, w1 in enumerate(w1):
            w = np.array([w0, w1])
            losses[i, j] = compute_loss(y, tx, w)

    return losses
