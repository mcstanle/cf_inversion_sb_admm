"""
Script to run the adjoint model at one particular input.

This script can be used for:
1. obtaining K^Tw to obtain the starting c
===============================================================================
Author        : Mike Stanley
Created       : Sep 12, 2023
Last Modified : Sep 12, 2023
===============================================================================
"""
from forward_adjoint_evaluators import adjoint_eval_cf
from generate_opt_objects import starting_point_generation
import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":

    # i/o
    HOME = '/glade/u/home/mcstanley'
    WORK = '/glade/work/mcstanley'
    HOME_RUN = HOME + '/gc_adj_runs/adjoint_model_osb_admm_lep'
    SUB_DIR_FILL = '/runs/v8-02-01/geos5'
    WORK_P_FIX = WORK + '/admm_objects/fixed_optimization_inputs'
    GOSAT_DF_FP = WORK_P_FIX + '/gosat_df_jan1_aug31_2010.csv'
    W_SAVE_FP = WORK + '/admm_objects/w_vecs/w_vec_lep.npy'
    SAT_OBS = WORK + '/Data/OSSE_OBS'
    W_DIR = WORK + '/admm_objects/w_gen_dir_lep'
    ADJOINT_EVAL_HT_FP = WORK + '/admm_objects/h_table_lep_TMP.pkl'
    COST_FUNC_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/cfn.01'
    GDT_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/gctm.gdt.01'
    ADJ_RUN_FP = HOME + '/pbs_run_scripts/run_adjoint_model_osb_admm_lep'
    YEAR = 2010
    MONTH_IDX = 9

    # define the necessary inputs
    gosat_df = pd.read_csv(GOSAT_DF_FP)
    L_inv_vec = 1 / gosat_df.xco2_unc.values

    # generate the starting w value
    m, d, p = 28267, 11120, 26496
    w_sp, c_sp, lambda_sp = starting_point_generation(
        m=m, d=d, p=p,
        random_seed=12345
    )

    # create dummy hash table so there are no issues
    with open(ADJOINT_EVAL_HT_FP, 'wb') as f:
        pickle.dump({0: None}, f)

    # call the adjoint model
    adj_val_flat, adj_cost = adjoint_eval_cf(
        w=w_sp,
        L_inv=L_inv_vec,
        w_save_fp=W_SAVE_FP,
        gosat_dir=SAT_OBS,
        w_dir=W_DIR,
        h_tabl_fp=ADJOINT_EVAL_HT_FP,
        cost_func_fp=COST_FUNC_FP,
        gdt_fp=GDT_FP,
        adj_run_fp=ADJ_RUN_FP,
        mnth_idx_bnd=MONTH_IDX,
        year=YEAR
    )

    # write out results
    OUTPUT_FP = WORK + '/admm_objects/misc/KTw_for_w_start.npy'
    with open(OUTPUT_FP, 'rb') as f:
        np.save(file=f, arr=adj_val_flat)
