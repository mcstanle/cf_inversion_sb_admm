"""
Test the objective and gradient function evaluations against the results
from unfolding_with_admm_rank_deficient.ipynb.
===============================================================================
Operation instructions:
$ pytest ./tests
===============================================================================
Author        : Mike Stanley
Created       : May 30, 2023
Last Modified : May 30, 2023
===============================================================================
"""
import sys
sys.path.append('../')
from admm_optimizer import (
    w_update_obj, c_update_obj,
    w_update_obj_grad, c_update_obj_grad
)
from create_unfolding_objects import gen_poisson_data
import cvxpy as cp
import numpy as np
from scipy import stats


def load_unfolding_test_objects():
    """
    Generates all objects necessary to run unfolding tests.
    """
    # means, forward model, and functional of interest
    BASE_DIR_DECONV = '/Users/mikestanley/Research/strict_bounds'
    BASE_DIR_DECONV += '/prior_optimized_paper/osb_po_uq'
    with open(BASE_DIR_DECONV + '/bin_means/gmm_rd.npz', 'rb') as f:
        bin_means = np.load(f)
        true_means = bin_means['t_means_rd']
        smeared_means = bin_means['s_means_rd']

    with open(
        BASE_DIR_DECONV + '/smearing_matrices/K_rank_deficient_mats.npz', 'rb'
    ) as f:
        smearing_mats = np.load(f)
        K = smearing_mats['K_rd']

    with open(
        BASE_DIR_DECONV + '/functionals/H_80_deconvolution.npy', 'rb'
    ) as f:
        H = np.load(f)

    # set the functional of interest
    h = H[6]

    # set variable dimensions
    m, p = K.shape

    # generate data
    DATA_IDX = 42
    np.random.seed(DATA_IDX)
    y = gen_poisson_data(mu=smeared_means)

    assert np.dot(h, true_means) == 3598.2487934782844

    # change of basis
    Sigma_data = np.diag(smeared_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)
    K_tilde = L_data_inv @ K
    y_tilde = L_data_inv @ y

    # slack factor
    x = cp.Variable(p)
    cost = cp.sum_squares(y_tilde - K_tilde @ x)
    prob = cp.Problem(
        objective=cp.Minimize(cost),
        constraints=[
            x >= 0
        ]
    )
    s2 = prob.solve(solver='ECOS')
    assert prob.status == 'optimal'

    # find the constraint bound
    psi_sq = np.square(
        stats.norm(loc=0, scale=1).ppf(1 - (0.05 / 2))
    ) + s2

    return K_tilde, y_tilde, psi_sq, h


def test_w_update_obj():
    """
    Test w_update_obj.
    """
    # create necessary objects
    K, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    m, p = K.shape
    w_test = np.ones(m)
    lambda_test = np.ones(p)
    mu = 1e3
    c_test = np.ones(p)
    LEP = True
    psi_alpha = np.sqrt(psi_sq)

    def forward_eval(x):
        return K @ x

    def adjoint_eval(w):
        return K.T @ w

    res = w_update_obj(
        w=w_test, lambda_k=lambda_test, c_k=c_test, mu_k=mu,
        lep=LEP, psi_alpha=psi_alpha, forward_eval=forward_eval,
        adjoint_eval=adjoint_eval, y=y, A=A, b=b, h=h
    )

    assert res == 118140.96370323896


def test_c_update_obj():

    # create necessary objects
    K, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    c_test = np.ones(p)
    w_test = np.ones(m)
    lambda_test = np.ones(p)
    mu = 1e3
    LEP = True

    res = c_update_obj(
        c=c_test, KTwk1=K.T @ w_test, lambda_k=lambda_test,
        mu_k=mu, lep=LEP, A=A, b=b, h=h
    )

    assert res == 86794.32523475224


def test_w_update_obj_grad(file_loc='./files/w_vec_grad.npy'):
    """
    Reads in a vector file created in unfolding_with_admm_rank_deficient.ipynb
    """
    with open(file_loc, 'rb') as f:
        test_gradient = np.load(f)

    # create necessary objects
    K, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    m, p = K.shape
    w_test = np.ones(m)
    lambda_test = np.ones(p)
    mu = 1e3
    c_test = np.ones(p)
    LEP = True
    psi_alpha = np.sqrt(psi_sq)

    def forward_eval(x):
        return K @ x

    def adjoint_eval(w):
        return K.T @ w

    res_vec = w_update_obj_grad(
        w=w_test, lambda_k=lambda_test, c_k=c_test, mu_k=mu, 
        lep=LEP, psi_alpha=psi_alpha,
        forward_eval=forward_eval, adjoint_eval=adjoint_eval,
        y=y, A=A, b=b, h=h
    )

    assert np.allclose(test_gradient, res_vec)


def test_c_update_obj_grad(file_loc='./files/c_vec_grad.npy'):
    """
    Reads in a vector file created in unfolding_with_admm_rank_deficient.ipynb
    """
    with open(file_loc, 'rb') as f:
        test_gradient = np.load(f)

    # create necessary objects
    K, y, psi_sq, h = load_unfolding_test_objects()

    # dimensions
    m, p = K.shape

    # constraint objects
    A = - np.identity(p)
    b = np.zeros(p)

    m, p = K.shape
    w_test = np.ones(m)
    lambda_test = np.ones(p)
    mu = 1e3
    c_test = np.ones(p)
    LEP = True

    res_vec = c_update_obj_grad(
        c=c_test, KTwk1=K.T @ w_test, lambda_k=lambda_test,
        mu_k=mu, lep=LEP, A=A, b=b, h=h
    )

    # print('-- res vec --')
    # print(res_vec)

    # print('-- file vec --')
    # print(test_gradient)

    assert np.allclose(test_gradient, res_vec)
