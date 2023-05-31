"""
Plotting functions for output. Namely,
1. objective function values over ADMM iterations
2. feasibility criterion over ADMM iterations
3. optimality criterion over iterations
===============================================================================
Author        : Mike Stanley
Created       : May 31, 2023
Last Modified : May 31, 2023
===============================================================================
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_objective(
    results_dict, save_path, true_endpoint,
    obj_plot_label, ylim=None
):
    """
    Plot objective function over ADMM iterations.

    From results_dict, use key 'objective_evals' for list

    Parameters
    ----------
        results_dict   (dict)  : contains plotting data
        save_path      (str)   : path to image save loc
        true_endpoint  (float) : true endpoint (if known as in unfolding)
        obj_plot_label (str)   : legend label for objective function values
        ylim           (tup)   : axis bounds for the y axis

    Returns
    -------
        writes image to save_path if defined
    """
    obj_func_evals = results_dict['objective_evals'].copy()
    x_vals = np.arange(1, len(obj_func_evals) + 1)
    plt.figure(figsize=(10, 5))

    if true_endpoint:
        plt.axhline(
            true_endpoint, linestyle='--', color='gray',
            label=f'True endpoint: {true_endpoint:.3f}'
        )

    plt.plot(x_vals, obj_func_evals, label=obj_plot_label)
    plt.xticks(x_vals)
    plt.xlabel('Outer Loop Iteration')
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_feasibility(
    results_dict, save_path, lep, h, A, cutoff=None
):
    """
    Plot feasiblity criterion over ADMM iterations.
    E.g., for LEP, this is h + A^Tc - K^T w = 0.

    In results_dict, use key 'KTw_vecs' for K^Tw evaluations and
    results_dict['c_opt_output']['vectors'] for the c values

    NOTE: plots log || h + A^T c - K^T w ||_2

    Parameters
    ----------
        results_dict   (dict)   : contains plotting data
        save_path      (str)    : path to image save loc
        lep            (bool)   : flag for lower endpoint
        h              (np arr) : functional of interest
        A              (np arr) : constraint matrix
        cutoff         (float)  : cut off of the above log norm to point out

    Returns
    -------
        writes image to save_path if defined
    """
    # label depending on the interval type
    if lep:
        line_label = r'$\log ||h + A^T c - K^T w||_2$'
    else:
        line_label = r'$\log ||h - A^T c - K^T w||_2$'

    # extract plotting objects from results_dict
    KTw_vecs = results_dict['KTw_vecs'].copy()
    c_vecs = results_dict['c_opt_output']['vectors'].copy()

    # look at the norm of the constraint term
    num_iters = KTw_vecs.shape[0]
    constr_mat = np.array([
        h + A.T @ c_vecs[i, :] - KTw_vecs[i, :]
        for i in range(num_iters)
    ])
    feasibility = np.sqrt(np.diagonal(constr_mat @ constr_mat.T))

    x_vals = np.arange(1, num_iters + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, np.log10(feasibility), label=line_label)

    if cutoff:

        # when does the norm go below some numerical tolerance?
        feas_idx = np.where(feasibility < cutoff)[0][0] + 1

        plt.axvline(
            feas_idx, linestyle=':', color='green',
            label=f'First feasible point (within a tol. of {cutoff})'
        )

    plt.xlabel('Outer Loop Iteration')
    plt.ylabel(line_label)
    plt.xticks(x_vals)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_optimality(
    results_dict, save_path, lep, y, K, psi_alpha
):
    """
    Plot gradient criterion of the Lagrangian to assess optimality.

    From results_dict, use 'w_opt_output' and 'vectors'

    Plots nabla L(w, c, lambda) to see how close to 0

    NOTE: this is not going to be possible for the carbon flux problem unless
    we somewhere get the K lambda_k evaluations.

    Parameters
    ----------
        results_dict (dict)   :
        save_path    (str)    :
        lep          (bool)   :
        y            (np arr) :
        K            (np arr) :
        psi_alpha    (float)  :

    Returns
    -------
        writes image to save_path if defined
    """
    lep_switch = -1 if lep else 1

    def lagrang_grad_w(w, l_vec):
        """ Lagrangian gradient with respect to w """
        return (psi_alpha / np.linalg.norm(w)) * w - lep_switch * y - K @ l_vec

    # extract plotting objects from results_dict
    w_vecs = results_dict['w_opt_output']['vectors'].copy()
    lambda_vecs = results_dict['lambda_opt_output']['vectors'].copy()
    num_iters = w_vecs.shape[0]

    lgrad_w = np.array([
        lagrang_grad_w(
            w=w_vecs[i, :],
            l_vec=lambda_vecs[i, :]) for i in range(num_iters)
    ])

    x_s = np.arange(num_iters) + 1
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_s, np.log10(np.linalg.norm(lgrad_w, axis=1)),
        label=r'$\log ||\nabla f(w^{k + 1}) - K \lambda^{k + 1} ||_2$'
    )
    plt.ylabel(r'$\log(|| \nabla \mathcal{L} ||_2)$')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
