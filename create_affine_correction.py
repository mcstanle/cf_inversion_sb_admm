"""
Creates the affine correction equivalent to f(0).

By default, this script uses the GEOS-Chem directory on Cheyenne defined for
the LEP calculations.

Saves the correction as a npy file.
===============================================================================
Author        : Mike Stanley
Created       : Jun 12, 2023
Last Modified : Jun 12, 2023
===============================================================================
"""
from forward_adjoint_evaluators import forward_eval_cf
import numpy as np
import os

if __name__ == '__main__':

    # operational parameters
    HOME = "/glade/u/home/mcstanley"
    WORK_DIR = "/glade/work/mcstanley"
    SF_BASE_LOC = "/glade/work/mcstanley/Data/osb_endpoint_sfs"
    NUM_OBS = 26496
    GC_DIR = HOME + "/gc_adj_runs/forward_model_osb_lep"
    SF_F_FP = SF_BASE_LOC + "/lep_sfs_forward.txt"
    FM_RUN_FP = HOME + "/pbs_run_scripts/run_forward_model_osb_lep"
    TIME_2_WAIT = 15  # seconds
    SAVE_DIR = WORK_DIR + "/admm_objects/fixed_optimization_inputs"

    # run the forward model
    f_c = forward_eval_cf(
        c=np.zeros(NUM_OBS),
        gc_dir=GC_DIR,
        sf_fp=SF_F_FP,
        fm_run_fp=FM_RUN_FP,
        time_wait=TIME_2_WAIT,
        max_et=28800  # default 8 hr
    )
    print('--- Head of Output ---')
    print(f_c[:10])

    # save the output
    SAVE_FP = SAVE_DIR + "/affine_correction.npy"
    with open(SAVE_FP, "wb") as f:
        np.save(file=f, arr=f_c)
    assert os.path.exists(SAVE_FP)
