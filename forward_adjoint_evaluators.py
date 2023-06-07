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
Last Modified : Jun 07, 2023
===============================================================================
"""
import pickle


def forward_eval_unfold(x, K, L_inv):
    """
    Performs a forward evaluation for the unfolding problem.

    Parameters
    ----------
        x     (np arr) : p x 1 - input vector
        K     (np arr) : m x p - forward model matrix
        L_inv (np arr) : m x m - inverse of l-tri cholesky factor

    Returns
    -------
        Kx (np arr) : m x 1
    """
    return L_inv @ K @ x


def adjoint_eval_unfold(w, K, L_inv, h_tabl_fp):
    """
    Performs an adjoint evaluation for the unfolding problem.

    Saves the K^Tw evaluation to a dictionary key'ed by hash(w.tobytes()).

    Parameters
    ----------
        w         (np arr) : m x 1 - input vector
        K         (np arr) : m x p - forward model matrix
        L_inv     (np arr) : m x m - inverse of l-tri cholesky factor
        h_tabl_fp (str)    : filepath to pickled dictionary hash table

    Returns
    -------
        K^Tw (np arr) : p x 1
    """
    # evaluate the adjoint
    KTw = K.T @ L_inv @ w

    # read in file
    with open(h_tabl_fp, 'rb') as f:
        adjoint_ht = pickle.load(f)

    # add new key value pair
    key = hash(w.tobytes())
    adjoint_ht[key] = KTw

    # write out new updated hash table
    with open(h_tabl_fp, 'wb') as f:
        pickle.dump(adjoint_ht, f)

    return KTw


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
