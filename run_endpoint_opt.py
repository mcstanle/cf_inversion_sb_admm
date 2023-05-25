"""
Script to kick off optimization jobs.
===============================================================================
Operation instructions:
$ python run_endpoint_opt.py > /fp/to/stdout
===============================================================================
Author        : Mike Stanley
Created       : May 24, 2023
Last Modified : May 24, 2023
===============================================================================
"""


def run_optimizer(lep=True):
    """
    Kicks off ADMM optimization. Supports both LEP and UEP optimization.

    Parameters
    ----------
        lep (bool) : flag to run lower endpoint optimization

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
