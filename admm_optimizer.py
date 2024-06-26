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
2. For the output of run_admm(), search "ADMM OUTPUT"
===============================================================================
TODO:
1. expand run_admm() to support non-constant mu
2. add more robust logging capabilities to run_admm
===============================================================================
Author        : Mike Stanley
Created       : May 24, 2023
Last Modified : Mar 17, 2024
===============================================================================
"""
from functools import partial
import numpy as np
import os
import pickle
from scipy.optimize import minimize


def w_update_obj(
    w, lambda_k, c_k, mu_k, lep,
    psi_alpha, forward_eval, adjoint_eval,
    jan_mask,
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
        forward_eval (function) : NOTE: this does nothing here, but needs to
                                  have same structure as w_update_obj_grad()
        adjoint_eval (function) : returns K^T w, where K is lin. forward model
        jan_mask     (bool)     : sets the jan inputs to the forward model to 1
                                : NOTE: does nothing here -- matching format
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
    KTw, adj_cost = adjoint_eval(w)  # TODO -- any other I/O checks required?

    return_val = psi_alpha * np.linalg.norm(w) + lep_switch * np.dot(w, y)
    return_val -= lambda_k.T @ KTw
    return_val += (mu_k / 2) * np.dot(KTw, KTw)
    return_val -= mu_k * np.dot(h - lep_switch * A.T @ c_k, KTw)

    return return_val


def w_update_obj_grad(
    w, lambda_k, c_k, mu_k, lep,
    psi_alpha, forward_eval, adjoint_eval,
    jan_mask,
    y, A, b, h
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
        jan_mask     (bool)     : sets the jan inputs to the forward model to 1
        y            (np arr)   : m x 1
        A            (np arr)   : d x p
        b            (np arr)   : d x 1 -- NOTE not used here, for parallelism
        h            (np arr)   : p x 1

    Returns
    -------
        gradient at w
    """
    lep_switch = -1 if lep else 1

    # adjoint evaluation K^T w
    KTw, adj_cost = adjoint_eval(w)

    # construct input for forward model run
    f_model_input = lambda_k - mu_k * KTw + mu_k * h
    f_model_input -= lep_switch * mu_k * A.T @ c_k

    # set january to 1s
    # if jan_mask:
    #     f_model_input[:(46 * 72)] = 1.

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


def endpoint_objective(w, c, b, y, lep, psi_alpha):
    """
    Interval endpoint objectives for both LEP and UEP

    Parameters
    ----------
        w            (np arr)   : m x 1
        c            (np arr)   : d x 1
        b            (np arr)   : d x 1
        y            (np arr)   : m x 1
        lep          (bool)     : True means optimize the lower endpoint
        psi_alpha    (float)    :

    Returns
    -------
        objective function evaluation (float)
    """
    lep_switch = -1 if lep else 1
    return_val = w @ y + lep_switch * psi_alpha * np.linalg.norm(w)
    return_val += lep_switch * np.dot(b, c)
    return return_val


def callback_save_iters(xk, save_loc):
    """
    Callback function to save the values of the optimization

    NOTE: this function appends to the file, so use appropriately.

    Parameters:
    -----------
        xk       (np arr) : current optimizer variable value
        save_loc (str)    : save location

    Returns:
    --------
        None
    """
    with open(save_loc, 'a') as f:
        np.savetxt(fname=f, X=xk)


def run_admm(
    y, A, b, h, w_start, c_start, lambda_start, mask_path, mu, psi_alpha,
    forward_eval, adjoint_eval, get_KTwk1, start_idx,
    lep, max_iters, maxls, callback_dir, subopt_iters,
    adjoint_ht_fp, int_dict_dir
):
    """
    ADMM interval endpoint optimizer.

    NOTE: this function currently only supports constant mu.

    NOTE: for the mask_path functionality, we currently only support setting
    the mask values all to 0.

    NOTE: if a mask path is given, that sets the variable jan_mask to True,
    which affects the gradient function for the w optimization.

    Sub-optimization details
    1. w optimization is performed with L-BFGS-B
    2. c optimization is performed with L-BFGS-B

    Parameters
    ----------
        y             (np arr) : m x 1 -- observation
        A             (np arr) : d x p -- constraint matrix
        b             (np arr) : d x 1 -- constraint vector
        h             (np arr) : p x 1 -- functional of interest
        w_start       (np arr) : m x 1 -- starting position for w
        c_start       (np arr) : d x 1 -- starting position for c
        lambda_start  (np arr) : d x 1 -- starting position for lambda
        mask_path     (str)    : path to npy file containing mask
        psi_alpha     (float)  : optimized slack factor
        forward_eval  (func)   : returns Kx (e.g., GEOS-Chem at x)
        adjoint_eval  (func)   : returns K^T w
        get_KTwk1     (func)   : returns the previous K^Tw_{k+1} for c opt
        start_idx     (int)    : interation index from which to start algorithm
        mu            (float)  : penalty parameter
        lep           (bool)   : True if solving for the lower endpoint
        max_iters     (int)    : max number of outer loop ADMM iterations
        maxls         (int)    : max number of line search steps
        callback_dir  (str)    : directory for callback func output from w opt
        subopt_iters  (int)    : max number of iterations for w optimization
        adjoint_ht_fp (str)    : filepath to hashtable for storing adjoint vals
        int_dict_dir  (str)    : directory where save intermediate data dict

    Returns
    -------
        at the end:
            1. dictionary of all the optimization and sub optimization output
        intermediate:
            1. callback output from w optimization
            2. optimized objects for each k in ADMM iterations
    """
    # determine the endpoint we're evaluating
    lep_switch = -1 if lep else 1

    # set problem dimensions
    m = y.shape[0]
    d, p = A.shape

    # set january mask
    jan_mask = True if mask_path else False

    # data structures to save output
    f_admm_evals = []
    w_opt_status = np.zeros(max_iters)
    w_opt_nfev = np.zeros(max_iters)
    w_opt_njev = np.zeros(max_iters)
    c_opt_status = np.zeros(max_iters)
    c_opt_messages = [''] * max_iters
    c_opt_nfev = np.zeros(max_iters)
    c_opt_njev = np.zeros(max_iters)

    # read in w mask
    if mask_path:
        with open(mask_path, 'rb') as f:
            mask_arr = np.load(f)

        # create mask bounds object for w optimization
        w_bounds = [(None, None)] * len(mask_arr)
        for i, bool_i in enumerate(mask_arr):
            if bool_i:
                w_bounds[i] = (0., 0.)

    # arrays for saving the intermediate optimized vectors
    w_opt_vecs = np.zeros(shape=(max_iters, m))
    c_opt_vecs = np.zeros(shape=(max_iters, d))
    lambda_opt_vecs = np.zeros(shape=(max_iters, p))
    KTw_vecs = np.zeros(shape=(max_iters, p))

    # initialize optimization vectors
    w_k = w_start.copy()
    c_k = c_start.copy()
    lambda_k = lambda_start.copy()

    for k in range(start_idx, max_iters):

        print(f'--- Iteration {k} ---')

        # create a hash table for the adjoint evaluations
        with open(adjoint_ht_fp, 'wb') as f:
            pickle.dump({0: None}, f)

        # define new callback function for this iteration
        callback_w = partial(
            callback_save_iters,
            save_loc=callback_dir + f'/callback_w_opt_{str(k).zfill(2)}.txt'
        )

        # w - update
        print(f'- w opt : iteration {k} -')
        w_opt_res = minimize(
            fun=w_update_obj,
            x0=w_k,
            jac=w_update_obj_grad,
            args=(
                lambda_k, c_k, mu, lep, psi_alpha,
                forward_eval, adjoint_eval,
                jan_mask,
                y, A, b, h
            ),
            method='L-BFGS-B',
            bounds=w_bounds if mask_path else None,
            options={
                'maxiter': subopt_iters,
                'maxls': maxls,
                'iprint': 99
            },
            callback=callback_w
        )
        # NOTE: remove later
        SAVE_PATH = int_dict_dir + f'/res_text{k}.pkl'
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(obj=w_opt_res, file=f)

        w_k = w_opt_res['x']
        w_opt_status[k] = w_opt_res['success']
        w_opt_nfev[k] = w_opt_res['nfev']
        # w_opt_njev[k] = w_opt_res['njev']

        # access K^T w_{k + 1}
        KTwk1 = get_KTwk1(w_k)
        KTw_vecs[k, :] = KTwk1

        # delete the hash table
        os.remove(adjoint_ht_fp)

        # c - update
        print(f'- c opt : iteration {k} -')
        c_opt_res = minimize(
            fun=c_update_obj,
            x0=c_k,
            jac=c_update_obj_grad,
            args=(
                KTwk1, lambda_k, mu, lep, A, b, h
            ),
            bounds=[(0, np.inf)] * d,
            method='L-BFGS-B',
            options={
                'maxls': 50,
                'disp': True
            }
        )
        c_k = c_opt_res['x']
        c_opt_status[k] = c_opt_res['success']
        c_opt_messages[k] = c_opt_res['message']
        c_opt_nfev[k] = c_opt_res['nfev']
        # c_opt_njev[k] = c_opt_res['njev']

        # dual variable update
        lambda_k += mu * (h - lep_switch * A.T @ c_k - KTwk1)

        # if jan_mask:
        #     lambda_k[:(46 * 72)] = 1.

        # save the objective function value
        f_admm_evals.append(
            endpoint_objective(
                w=w_k, c=c_k, b=b, y=y, lep=lep, psi_alpha=psi_alpha
            )
        )

        # save the optimized vectors
        w_opt_vecs[k, :] = w_k
        c_opt_vecs[k, :] = c_k
        lambda_opt_vecs[k, :] = lambda_k

        # output intermediate dictionary with key data
        DICT_FP = int_dict_dir + f'/opt_output_{str(k).zfill(2)}.pkl'
        int_dict = {
            'objective_evals': f_admm_evals,
            'w_opt_vecs': w_opt_vecs,
            'c_opt_vecs': c_opt_vecs,
            'lambda_opt_vecs': lambda_opt_vecs,
            'KTw_vecs': KTw_vecs
        }
        with open(DICT_FP, 'wb') as f:
            pickle.dump(int_dict, f)

    # NOTE - ADMM OUTPUT
    return {
        'objective_evals': f_admm_evals,
        'w_opt_output': {
            'vectors': w_opt_vecs,
            'status': w_opt_status,
            'nfev': w_opt_nfev,
            'njev': w_opt_njev
        },
        'c_opt_output': {
            'vectors': c_opt_vecs,
            'status': c_opt_status,
            'nfev': c_opt_nfev,
            'njev': c_opt_njev
        },
        'lambda_opt_output': {
            'vectors': lambda_opt_vecs
        },
        'KTw_vecs': KTw_vecs
    }
