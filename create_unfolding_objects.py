"""
Script to create the input elements to run the ADMM algorithm for the
unfolding problem.

Generates data from the correct poisson process.
===============================================================================
Creates a dictionary with the following elements
1. y            (observation vector -- transformed by cholesky decomp)
2. K            (smearning matrix -- transformed by cholesky decomp)
3. A            (constraint matrix)
4. b            (constraint vector)
5. psi_alpha_sq (slack factor)
6. h            (functional of interest)
7. OSB_int      (osb interval for comparison)

NOTE: See /Research/Carbon_Flux/optimization/ADMM_dual_ascent/
           unfolding_with_admm_rank_deficient.ipynb for inspired code.
===============================================================================
Author        : Mike Stanley
Created       : May 26, 2023
Last Modified : May 26, 2023
===============================================================================
"""
import cvxpy as cp
import numpy as np
import os
import pickle
from scipy import stats
from tqdm import tqdm


def gen_poisson_data(mu):
    """
    Create one realization of data given vector of bin means

    Parameters
    ----------
        mu (np arr) : p x 1

    Returns
    -------
        data (np arr) : m x 1
    """
    data = np.zeros_like(mu)
    for i in range(mu.shape[0]):
        data[i] = stats.poisson(mu=mu[i]).rvs()

    return data


def osb_interval(
    y, K, h, A, alpha=0.05, verbose=False, options={}
):
    """
    Copied from ~/Research/strict_bounds/prior_optimized_paper/
                  osb_po_uq/interval_estimators.py

    NOTE: dimension keys have been udpated from n x m to m x p

    Compute OSB interval.

    Dimension key:
    - m : number of smear bins
    - p : number of true bins

    Parameters:
    -----------
        y         (np arr) : cholesky transformed data -- m x 1
        K         (np arr) : cholesky transformed matrix -- m x p
        h         (np arr) : functional for parameter transform -- p x 1
        A         (np arr) : Matrix to enforce non-trivial constraints
        alpha     (float)  : type 1 error threshold -- (1 - confidence level)
        options   (dict)   : ECOS options for cvxpy

    Returns:
    --------
        opt_lb, -opt_ub (tup) : lower and upper interval bounds
        sq_err_constr (float) : slack factor for optimization
    """
    p = K.shape[1]

    # find the slack factor
    x = cp.Variable(p)
    cost = cp.sum_squares(y - K @ x)
    prob = cp.Problem(
        objective=cp.Minimize(cost),
        constraints=[
            A @ x <= 0
        ]
    )
    s2 = prob.solve(solver='ECOS', verbose=verbose, **options)

    # find the constraint bound
    sq_err_constr = np.square(
        stats.norm(loc=0, scale=1).ppf(1 - (alpha / 2))
    ) + s2

    # define a variables to solve the problem
    x_lb = cp.Variable(p)
    x_ub = cp.Variable(p)

    # define the problem
    prob_lb = cp.Problem(
        objective=cp.Minimize(h.T @ x_lb),
        constraints=[
            cp.square(cp.norm2(y - K @ x_lb)) <= sq_err_constr,
            A @ x_lb <= 0
        ]
    )
    prob_ub = cp.Problem(
        objective=cp.Minimize(-h.T @ x_ub),
        constraints=[
            cp.square(cp.norm2(y - K @ x_ub)) <= sq_err_constr,
            A @ x_ub <= 0
        ]
    )

    # solve the problem
    opt_lb = prob_lb.solve(solver='ECOS', verbose=verbose, **options)
    opt_ub = prob_ub.solve(solver='ECOS', verbose=verbose, **options)

    # check for convergence
    assert 'optimal' in prob_lb.status
    assert 'optimal' in prob_ub.status

    return opt_lb, -opt_ub, sq_err_constr


if __name__ == '__main__':

    # file locations
    BASE_PATH = './data'
    BASE_DIR_DECONV = '/Users/mikestanley/Research/strict_bounds'
    BASE_DIR_DECONV += '/prior_optimized_paper/osb_po_uq'

    # operational parameters
    ENS_SIZE = 50  # determines number of data points
    FUNCTIONAL_IDX = 6  # index of functional of interest
    DATA_IDX = 42  # index of data with which to build comparison interval
    ALPHA = 0.05  # 1 - confidence level

    # acquire bin means and smearing matrix
    with open(BASE_DIR_DECONV + '/bin_means/gmm_rd.npz', 'rb') as f:
        bin_means = np.load(f)
        true_means = bin_means['t_means_rd']
        smeared_means = bin_means['s_means_rd']

    p = true_means.shape[0]
    m = smeared_means.shape[0]
    print(f'True space dimension: {p}')
    print(f'Data space dimension: {m}')

    S_MAT_FP = BASE_DIR_DECONV + '/smearing_matrices/K_rank_deficient_mats.npz'
    with open(S_MAT_FP, 'rb') as f:
        smearing_mats = np.load(f)
        K = smearing_mats['K_rd']

    print(f'Smearing matrix dimension: {K.shape}')

    # generate data
    data = np.zeros(shape=(ENS_SIZE, m))

    for i in tqdm(range(ENS_SIZE)):

        np.random.seed(i)
        data[i, :] = gen_poisson_data(mu=smeared_means)

    # read in the functional of interest
    with open(
        BASE_DIR_DECONV + '/functionals/H_80_deconvolution.npy', 'rb'
    ) as f:
        H = np.load(f)
    h = H[FUNCTIONAL_IDX]

    print(f'True functional value: {np.dot(h, true_means)}')

    # define the constraints
    A = - np.identity(p)
    b = np.zeros(p)

    # optimize comparison interval
    y = data[DATA_IDX]

    # define the change of basis to move to identity covariance
    Sigma_data = np.diag(smeared_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)

    # transform the matrix
    K_tilde = L_data_inv @ K

    # transform the data
    y_tilde = L_data_inv @ y

    # find the interval
    osb_lep, osb_uep, psi_alpha_sq = osb_interval(
        y=y_tilde,
        K=K_tilde,
        h=h,
        A=A,
        alpha=ALPHA
    )
    print(f'CVXPY Interval: {(osb_lep, osb_uep)}')

    # write data to an npz file
    with open(BASE_PATH + '/unfolding_data.pkl', 'wb') as f:
        pickle.dump({
            'y': y_tilde,
            'K': K_tilde,
            'A': A,
            'b': b,
            'psi_alpha_sq': psi_alpha_sq,
            'h': h,
            'osb_int': (osb_lep, osb_uep)
        }, f)
    assert os.path.isfile(BASE_PATH + '/unfolding_data.pkl')
    print(f'Output written to: {BASE_PATH + "/unfolding_data.pkl"}')
