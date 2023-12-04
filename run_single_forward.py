"""
Script to run the forward model at one particular input.

This script can be used for:
1. obtaining K x for a given x

NOTE: we use the foward_linear_eval_cf function, which requires being mindful
of the affine correction.
===============================================================================
Author        : Mike Stanley
Created       : Dec 04, 2023
Last Modified : Dec 04, 2023
===============================================================================
"""
from forward_adjoint_evaluators import forward_linear_eval_cf
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # i/o
    HOME = '/glade/u/home/mcstanley'
    WORK = '/glade/work/mcstanley'
    WORK_P_FIX = WORK + '/admm_objects/fixed_optimization_inputs'
    AFFINE_CORR_FP = WORK_P_FIX + '/affine_correction.npy'
    GC_DIR = HOME + '/gc_adj_runs/forward_model_osb_lep'
    SF_F_FP = WORK + '/Data/osb_endpoint_sfs/lep_sfs_forward.txt'
    FM_RUN_FP = HOME + '/pbs_run_scripts/run_forward_model_osb_lep'
    GOSAT_DF_FP = WORK_P_FIX + '/gosat_df_jan1_aug31_2010.csv'
    TIME_2_WAIT = 15

    # define the necessary inputs
    gosat_df = pd.read_csv(GOSAT_DF_FP)
    L_inv_vec = 1 / gosat_df.xco2_unc.values

    # read in a input vector
    with open(
        WORK + '/admm_objects/lambda_vecs/lambda_06.npy',
        'rb'
    ) as f:
        x = np.load(f)

    # call the adjoint model
    output_forward_vec = forward_linear_eval_cf(
        c=x,
        L_inv=L_inv_vec,
        aff_corr_fp=AFFINE_CORR_FP,
        gc_dir=GC_DIR,
        sf_fp=SF_F_FP,
        fm_run_fp=FM_RUN_FP,
        time_wait=TIME_2_WAIT
    )

    # write out results
    OUTPUT_FP = WORK + '/admm_objects/results/misc/opt_check_K_lambda_06.npy'
    with open(OUTPUT_FP, 'wb') as f:
        np.save(file=f, arr=output_forward_vec)
