"""
Script to kick off optimization jobs for the carbon flux problem. This script
is called for both the LEP and UEP optimization
===============================================================================
Author        : Mike Stanley
Created       : Jun 13, 2023
Last Modified : Jun 13, 2023
===============================================================================
"""
import os

if __name__ == "__main__":

    # operational parameters
    LEP_OPT = os.environ['LEP']
    MAX_ITERS = os.environ['MAX_ITERS']
    SUBOPT_ITERS = os.environ['SUBOPT_ITERS']

    print(LEP_OPT)
    print(MAX_ITERS)
    print(SUBOPT_ITERS)
