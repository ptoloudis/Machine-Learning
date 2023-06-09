# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_loss
import numpy as np

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # implement stochastic gradient computation. It's the same as the usual gradient.
    n = len(y)
    random_index = np.random.randint(n, size=1)
    y_n = y[random_index]
    tx_n = tx[random_index]
    e = y_n - tx_n.dot(w)
    return -1/n * tx_n.T.dot(e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        # implement stochastic gradient descent.
        indices = np.random.choice(len(y), size=batch_size, replace=False)
        batch_y = y[indices]
        batch_tx = tx[indices, :]
        grad = compute_stoch_gradient(batch_y, batch_tx, w)
        
        # update the model parameters using the gradient and the stepsize
        w = w - gamma * grad
        
        # compute the loss using the updated model parameters and store it
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        
        # store the updated model parameters
        ws.append(w)

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws