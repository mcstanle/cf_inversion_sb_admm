"""
Defining forward and adjoint evaluation wrappers for different applications.
===============================================================================
Currently supported applications:
1. Unfolding
    -> forward function: forward_eval_unfold()
    -> adjoint function: adjoint_eval_unfold()
2. Carbon Flux inversion
    -> forward function: forward_eval_cf()
    -> adjoint function: adjoint_eval_cf()
===============================================================================
Author        : Mike Stanley
Created       : May 26, 2023
Last Modified : May 26, 2023
===============================================================================
"""


def forward_eval_unfold(x, K):
    """
    Performs a forward evaluation for the unfolding problem.

    Parameters
    ----------
        x (np arr) : p x 1 - input vector
        K (np arr) : m x p - forward model matrix

    Returns
    -------
        Kx (np arr) : m x 1
    """
    return K @ x


def adjoint_eval_unfold(w, K):
    """
    Performs an adjoint evaluation for the unfolding problem.

    Parameters
    ----------
        w (np arr) : m x 1 - input vector
        K (np arr) : m x p - forward model matrix

    Returns
    -------
        K^Tw (np arr) : p x 1
    """
    return K.T @ w


def forward_eval_cf():
    """
    Forward model evaluation for carbon flux inversion.

    TODO
    """
    pass


def adjoint_eval_cf():
    """
    Adjoint model evaluation for carbon flux inversion.

    TODO
    """
    pass