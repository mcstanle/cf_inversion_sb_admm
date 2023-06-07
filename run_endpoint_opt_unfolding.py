"""
Script to kick off optimization jobs for the unfolding problem.
===============================================================================
Operation instructions:
$ python run_endpoint_opt_unfolding.py > /fp/to/stdout
===============================================================================
Author        : Mike Stanley
Created       : May 26, 2023
Last Modified : Jun 07, 2023
===============================================================================
"""
from admm_optimizer import run_admm
from functools import partial
from forward_adjoint_evaluators import forward_eval_unfold, adjoint_eval_unfold
import numpy as np
import os
import pickle
from plotting import (
    plot_objective,
    plot_feasibility,
    plot_optimality
)
from scipy import stats


def get_KTwk1(w, adjoint_ht_fp):
    """
    Obtain the last K^Tw_{k + 1} for the next c sub-opt.

    Parameters
    ----------
        w             (np arr) : w for which we want to find K^Tw
        adjoint_ht_fp (str)    : file path to the adjoint eval hash table

    Returns
    -------
        K^T w (np arr) : p x 1
    """
    # read in the hash table
    with open(adjoint_ht_fp, 'rb') as f:
        adjoint_ht = pickle.load(f)

    w_hash = hash(w.tobytes())
    return adjoint_ht[w_hash]


def run_optimizer(
    y, A, b, h,
    w_start, c_start, lambda_start,
    mu, psi_alpha,
    forward_eval, adjoint_eval, get_KTwk1,
    lep, max_iters, subopt_iters, adjoint_ht_fp
):
    """
    Kicks off ADMM optimization. Supports both LEP and UEP optimization.

    Parameters
    ----------
        y             (np arr) : m x 1
        A             (np arr) : d x p
        b             (np arr) : d x 1
        h             (np arr) : p x 1
        w_start       (np arr) : m x 1
        c_start       (np arr) : d x 1
        lambda_start  (np arr) : p x 1
        mu            (float)  : penalty parameter
        psi_alpha     (float)  : sqrt of slack term
        forward_eval  (func)   :
        adjoint_eval  (func)   :
        get_KTwk1     (func)   :
        lep           (bool)   : flag to run lower endpoint optimization
        max_iters     (int)    : number of ADMM iterations
        subopt_iters  (int)    : number of iterations in suboptimizations
        adjoint_ht_fp (str)    : filepath to hashtable for storing adjoint vals

    Returns
    -------
        admm_output_dict (dict) : see admm_optimizer for contents
    """
    # perform checks
    # NONE here

    # run optimization
    admm_output_dict = run_admm(
        y=y,
        A=A,
        b=b,
        h=h,
        w_start=w_start,
        c_start=c_start,
        lambda_start=lambda_start,
        mu=mu,
        psi_alpha=psi_alpha,
        forward_eval=forward_eval,
        adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1,
        lep=lep,
        max_iters=max_iters,
        subopt_iters=subopt_iters,
        adjoint_ht_fp=adjoint_ht_fp
    )

    return admm_output_dict


def check_directories():
    """
    Checks for necessary directories to run code.
    """
    assert os.path.isdir('./data')
    assert os.path.isdir('./data/lep_diagnostic_plots')
    assert os.path.isdir('./data/uep_diagnostic_plots')


if __name__ == "__main__":

    # operational parameters
    LEP_OPT = True
    MAX_ITERS = 20  # number of ADMM iterations
    SUBOPT_ITERS = 12
    MU = 1e3
    MU_S = '1e3'

    # check if necessary directories exist
    check_directories()

    # filepath file
    pfx = 'LEP' if LEP_OPT else 'UEP'
    OBJECTS_FP = './data/unfolding_data.pkl'
    SAVE_PATH = f'./data/unfolding_results_{pfx}_mu{MU_S}.pkl'
    PLOT_PATH = f'./data/{pfx}_diagnostic_plots'

    # read in data objects for unfolding
    with open(OBJECTS_FP, 'rb') as f:
        unfold_objs = pickle.load(f)

    y = unfold_objs['y']
    K = unfold_objs['K']
    L_inv = unfold_objs['L_inv']
    A = unfold_objs['A']
    b = unfold_objs['b']
    psi_alpha_sq = unfold_objs['psi_alpha_sq']
    h = unfold_objs['h']
    osb_int_cvxpy = unfold_objs['osb_int']

    print(f'Psi_alpha_sq: {psi_alpha_sq}')
    print(f'CVXPY Optimized Interval: {osb_int_cvxpy}')

    # define some dimensions
    m = y.shape
    d, p = A.shape

    # directory for adjoint eval hash table
    ADJOINT_EVAL_HT_FP = './data/adjoint_lookup_TEMP.pkl'

    # create wrappers around fuctions involving K matrix
    forward_eval = partial(
        forward_eval_unfold,
        K=K, L_inv=L_inv
    )
    adjoint_eval = partial(
        adjoint_eval_unfold,
        K=K, L_inv=L_inv,
        h_tabl_fp=ADJOINT_EVAL_HT_FP
    )
    get_KTwk1_par = partial(
        get_KTwk1, adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )

    # define starting points
    np.random.seed(12345)
    w_sp = stats.multivariate_normal(mean=np.zeros(m)).rvs()
    c_sp = np.zeros(d)
    lambda_sp = np.zeros(p)

    # transform data by inverse cholsky factor
    y_tilde = L_inv @ y

    # run optimization
    print('Running ADMM...')
    res_dict = run_optimizer(
        y=y_tilde,
        A=A,
        b=b,
        h=h,
        w_start=w_sp,
        c_start=c_sp,
        lambda_start=lambda_sp,
        mu=MU,
        psi_alpha=np.sqrt(psi_alpha_sq),
        forward_eval=forward_eval,
        adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par,
        lep=LEP_OPT,
        max_iters=MAX_ITERS,
        subopt_iters=SUBOPT_ITERS,
        adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )

    # show the computed result
    print(f'Computed {pfx} values: {res_dict["objective_evals"]}')

    # generate plots
    plot_objective(
        results_dict=res_dict,
        save_path=PLOT_PATH + f'/{pfx}_mu{MU_S}_20maxiter_12suboptiter.png',
        true_endpoint=osb_int_cvxpy[0 if LEP_OPT else 1],
        obj_plot_label='Objective function values',
        # ylim=(0, 5e3)
    )
    feas_fp = PLOT_PATH + f'/{pfx}_mu{MU_S}_20maxiter_12suboptiter_feas.png'
    plot_feasibility(
        results_dict=res_dict,
        save_path=feas_fp,
        lep=LEP_OPT, h=h, A=A,
        # cutoff=1e-1
    )
    opt_fp = PLOT_PATH + f'/{pfx}_mu{MU_S}_20maxiter_12suboptiter_opt.png'
    plot_optimality(
        results_dict=res_dict,
        save_path=opt_fp,
        lep=LEP_OPT,
        y=y, K=K, psi_alpha=np.sqrt(psi_alpha_sq)
    )

    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(res_dict, f)
    assert os.path.isfile(SAVE_PATH)
