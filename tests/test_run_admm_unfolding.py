"""
Tests run_admm() in admm_optimizer for output it expects to see for both the
lower and upper endpoints. The comparison results are those obtained in the
notebook ~/Research/Carbon_Flux/optimization/ADMM_dual_ascent/
unfolding_with_admm_rank_deficient.ipynb.
===============================================================================
NOTE - used the pytest.fixture decorator to take care  of hashtable file io
===============================================================================
Author        : Mike Stanley
Created       : May 31, 2023
Last Modified : Jun 07, 2023
===============================================================================
"""
import sys
sys.path.append('../')
from admm_optimizer import run_admm
from forward_adjoint_evaluators import forward_eval_unfold, adjoint_eval_unfold
from functools import partial
import numpy as np
import os
import pickle
import pytest
from run_endpoint_opt_unfolding import get_KTwk1
from scipy import stats
from test_objective_and_gradients import load_unfolding_test_objects


test_numbers = {
    'lep': 2100.8794474850197,  # 2101.034800054623,
    'uep': 4697.200645884314  # 4682.662766066206
}


@pytest.fixture
def temp_file(tmp_path):
    """ creates a temporary file to use for hash tables in below tests """
    file_path = tmp_path / "adjoint_lookup_TEMP.pkl"
    file_path.touch()
    return file_path


def test_run_admm_lep(temp_file):
    """
    The verification number in here is correct as of May 31, 2023.
    """

    # obtain objects to run tess
    K, L_inv, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    # directory for adjoint eval hash table
    ADJOINT_EVAL_HT_FP = temp_file

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
    c_sp = np.zeros(p)
    lambda_sp = np.zeros(p)

    # operational parameter
    LEP_OPT = True
    MAX_ITERS = 20  # number of ADMM iterations
    SUBOPT_ITERS = 12
    MU = 1e3

    # run optimization
    output_dict = run_admm(
        y=L_inv @ y, A=A, b=b, h=h,
        w_start=w_sp, c_start=c_sp, lambda_start=lambda_sp,
        mu=MU, psi_alpha=np.sqrt(psi_sq),
        forward_eval=forward_eval, adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par, lep=LEP_OPT,
        max_iters=MAX_ITERS, subopt_iters=SUBOPT_ITERS,
        adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )

    assert output_dict['objective_evals'][-1] == test_numbers['lep']


def test_run_admm_uep(temp_file):
    """
    The verification number in here is correct as of May 31, 2023.
    """

    # obtain objects to run tess
    K, L_inv, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    # directory for adjoint eval hash table
    ADJOINT_EVAL_HT_FP = temp_file

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
    c_sp = np.zeros(p)
    lambda_sp = np.zeros(p)

    # operational parameter
    LEP_OPT = False
    MAX_ITERS = 20  # number of ADMM iterations
    SUBOPT_ITERS = 12
    MU = 1e3

    # run optimization
    output_dict = run_admm(
        y=L_inv @ y, A=A, b=b, h=h,
        w_start=w_sp, c_start=c_sp, lambda_start=lambda_sp,
        mu=MU, psi_alpha=np.sqrt(psi_sq),
        forward_eval=forward_eval, adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par, lep=LEP_OPT,
        max_iters=MAX_ITERS, subopt_iters=SUBOPT_ITERS,
        adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )

    assert output_dict['objective_evals'][-1] == test_numbers['uep']
