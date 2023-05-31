"""
Tests run_admm() in admm_optimizer for output it expects to see for both the
lower and upper endpoints. The comparison results are those obtained in the
notebook ~/Research/Carbon_Flux/optimization/ADMM_dual_ascent/
unfolding_with_admm_rank_deficient.ipynb.
===============================================================================
Author        : Mike Stanley
Created       : May 31, 2023
Last Modified : May 31, 2023
===============================================================================
"""
import sys
sys.path.append('../')
from admm_optimizer import run_admm
from forward_adjoint_evaluators import forward_eval_unfold, adjoint_eval_unfold
from functools import partial
import numpy as np
from run_endpoint_opt_unfolding import get_KTwk1
from scipy import stats
from test_objective_and_gradients import load_unfolding_test_objects


test_numbers = {
    'lep': 2101.034800054623,
    'uep': 4682.662766066206
}


def test_run_admm_lep():
    """
    The verification number in here is correct as of May 31, 2023.
    """

    # obtain objects to run tess
    K_tilde, y_tilde, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K_tilde.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    # create wrappers around fuctions involving K matrix
    forward_eval = partial(forward_eval_unfold, K=K_tilde)
    adjoint_eval = partial(adjoint_eval_unfold, K=K_tilde)
    get_KTwk1_par = partial(get_KTwk1, K=K_tilde)

    # define starting points
    np.random.seed(12345)
    w_sp = stats.multivariate_normal(mean=np.zeros(m)).rvs()
    c_sp = np.zeros(p)
    lambda_sp = np.zeros(p)

    # operational parameter
    LEP_OPT = True
    MAX_ITERS = 20  # number of ADMM iterations
    SUBOPT_ITERS = 12
    MU = 1e3

    # run optimization
    output_dict = run_admm(
        y=y_tilde, A=A, b=b, h=h,
        w_start=w_sp, c_start=c_sp, lambda_start=lambda_sp,
        mu=MU, psi_alpha=np.sqrt(psi_sq),
        forward_eval=forward_eval, adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par, lep=LEP_OPT,
        max_iters=MAX_ITERS, subopt_iters=SUBOPT_ITERS
    )

    assert output_dict['objective_evals'][-1] == test_numbers['lep']


def test_run_admm_uep():
    """
    The verification number in here is correct as of May 31, 2023.
    """

    # obtain objects to run tess
    K_tilde, y_tilde, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K_tilde.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    # create wrappers around fuctions involving K matrix
    forward_eval = partial(forward_eval_unfold, K=K_tilde)
    adjoint_eval = partial(adjoint_eval_unfold, K=K_tilde)
    get_KTwk1_par = partial(get_KTwk1, K=K_tilde)

    # define starting points
    np.random.seed(12345)
    w_sp = stats.multivariate_normal(mean=np.zeros(m)).rvs()
    c_sp = np.zeros(p)
    lambda_sp = np.zeros(p)

    # operational parameter
    LEP_OPT = False
    MAX_ITERS = 20  # number of ADMM iterations
    SUBOPT_ITERS = 12
    MU = 1e3

    # run optimization
    output_dict = run_admm(
        y=y_tilde, A=A, b=b, h=h,
        w_start=w_sp, c_start=c_sp, lambda_start=lambda_sp,
        mu=MU, psi_alpha=np.sqrt(psi_sq),
        forward_eval=forward_eval, adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par, lep=LEP_OPT,
        max_iters=MAX_ITERS, subopt_iters=SUBOPT_ITERS
    )

    assert output_dict['objective_evals'][-1] == test_numbers['uep']
