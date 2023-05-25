"""
Operational code for the ADMM optimizer.
===============================================================================
Sources of inspiration:
1. ~/Research/Carbon_Flux/optimization/ADMM_dual_ascent/
   unfolding_with_admm_rank_deficient.ipynb
===============================================================================
General notation:
1. w -- optimization variable for first sub-optimization
2. c -- optimization variable for second sub-optimization
===============================================================================
Comments
1. The penalty parameter mu_k is provided instead of a strictly constant mu to
   provide the flexibility to dyanmically choose the penalty parameter, though
   this does not necessarily need to be the case.
===============================================================================
Author        : Mike Stanley
Created       : May 24, 2023
Last Modified : May 25, 2023
===============================================================================
"""
import numpy as np


def w_update_obj(
    w, lambda_k, c_k, mu_k, lep,
    psi_alpha, adjoint_eval,
    y, A, b, h
):
    """
    Objective function for the w update step.

    w^{k + 1} = argmin_w L_mu(w, c^k, lambda_k)

    Parameters
    ----------
        w            (np arr)   : m x 1
        lambda_k     (np arr)   : p x 1
        c_k          (np arr)   : d x 1
        mu_k         (float)    : penalty parameter > 0
        lep          (bool)     : True means optimize the lower endpoint
        psi_alpha    (float)    :
        adjoint_eval (function) : returns K^T w, where K is lin. forward model
        y            (np arr)   : m x 1
        A            (np arr)   : d x p
        b            (np arr)   : d x 1
        h            (np arr)   : p x 1

    Returns
    -------
        objective function at w
    """
    lep_switch = -1 if lep else 1

    # call adjoint model to get K^T w evaluation
    KTw = adjoint_eval(w)  # TODO -- any other I/O checks required here?

    return_val = psi_alpha * np.linalg.norm(w) + lep_switch * np.dot(w, y)
    return_val -= lambda_k.T @ KTw
    return_val += (mu_k / 2) * np.dot(KTw, KTw)
    return_val -= mu_k * np.dot(h - lep_switch * A.T @ c_k, KTw)

    return return_val


def w_update_obj_grad(
    w, lambda_k, c_k, mu_k, lep,
    psi_alpha, forward_eval, adjoint_eval,
    y, A, h
):
    """
    Gradient of the above w suboptimization objective function.

    Requires 1 adjoint run and 1 forward run.

    Parameters
    ----------
        w            (np arr)   : m x 1
        lambda_k     (np arr)   : p x 1
        c_k          (np arr)   : d x 1
        mu_k         (float)    : penalty parameter > 0
        lep          (bool)     : True means optimize the lower endpoint
        psi_alpha    (float)    :
        forward_eval (function) : returns Kx, where K is lin. forward model
        adjoint_eval (function) : returns K^T w, where K is lin. forward model
        y            (np arr)   : m x 1
        A            (np arr)   : d x p
        b            (np arr)   : d x 1
        h            (np arr)   : p x 1

    Returns
    -------
        gradient at w
    """
    lep_switch = -1 if lep else 1

    # adjoint evaluation K^T w
    KTw = adjoint_eval(w)

    # construct input for forward model run
    f_model_input = lambda_k - lep_switch * mu_k * KTw - mu_k * h - mu_k * A.T @ c_k

    # evaluate the forward model
    Kx = forward_eval(f_model_input)

    return (psi_alpha / np.linalg.norm(w)) * w + lep_switch * y - Kx


def c_update_obj(
    c, KTwk1, lambda_k, mu_k, lep, A, b, h
):
    """
    Objective function for c update step.

    NOTE: K^Tw_{k+1} needs to be preprocessed.

    Parameters
    ----------
        c            (np arr)   : d x 1
        KTwk1        (np arr)   : p x 1 - adjoint evaluated at w_{k+1}
        lambda_k     (np arr)   : p x 1
        mu_k         (float)    : penalty parameter > 0
        lep          (bool)     : True means optimize the lower endpoint
        A            (np arr)   : d x p
        b            (np arr)   : d x 1
        h            (np arr)   : p x 1

    Returns
    -------
    """
    lep_switch = -1 if lep else 1
    q = b - lep_switch * A @ lambda_k + lep_switch * mu_k * A @ KTwk1
    hAc = h - lep_switch * A.T @ c
    return np.dot(q, c) + (mu_k / 2) * np.dot(hAc, hAc)


def c_update_obj_grad(
    c, KTwk1, lambda_k, mu_k, lep, A, b, h
):
    """
    Gradient of objective function for c update step.

    NOTE: K^Tw_{k+1} needs to be preprocessed.

    Parameters
    ----------
        c            (np arr)   : d x 1
        KTwk1        (np arr)   : p x 1 - adjoint evaluated at w_{k+1}
        lambda_k     (np arr)   : p x 1
        mu_k         (float)    : penalty parameter > 0
        lep          (bool)     : True means optimize the lower endpoint
        A            (np arr)   : d x p
        b            (np arr)   : d x 1
        h            (np arr)   : p x 1

    Returns
    -------
    """
    lep_switch = -1 if lep else 1
    q = b - lep_switch * A @ lambda_k + lep_switch * mu_k * A @ KTwk1
    hAc = h - lep_switch * A.T @ c
    return q - lep_switch * mu_k * A @ hAc
