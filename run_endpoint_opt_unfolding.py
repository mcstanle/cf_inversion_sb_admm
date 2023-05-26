"""
Script to kick off optimization jobs for the unfolding problem.
===============================================================================
Operation instructions:
$ python run_endpoint_opt_unfolding.py > /fp/to/stdout
===============================================================================
Author        : Mike Stanley
Created       : May 26, 2023
Last Modified : May 26, 2023
===============================================================================
"""
from admm_optimizer import run_admm
from forward_adjoint_evaluators import forward_eval_unfold, adjoint_eval_unfold


def get_KTwk1(w, K):
    """
    Obtain the last K^Tw_{k + 1} for the next c sub-opt.

    Parameters
    ----------
        w (np arr) : m x 1
        K (np arr) : m x p - forward model

    Returns
    -------
        K^T w (np arr) : p x 1
    """
    pass


def run_optimizer(forward_eval, adjoint_eval, lep):
    """
    Kicks off ADMM optimization. Supports both LEP and UEP optimization.

    Parameters
    ----------
        forward_eval (func) :
        adjoint_eval (func) :
        lep          (bool) : flag to run lower endpoint optimization

    Returns
    -------
        None
    """
    # perform checks
    # check f(0) vector is available

    print('RUN OPTIMIZER')


if __name__ == "__main__":

    # LEP switch
    LEP_OPT = True

    # filepath file

    # run optimization
    run_optimizer()
