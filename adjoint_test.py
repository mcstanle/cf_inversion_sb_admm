"""
Testing the functionality of adjoint_eval_cf on the LEP.
===============================================================================
Author        : Mike Stanley
Created       : Jun 14, 2023
Last Modified : Jun 14, 2023
===============================================================================
"""
from forward_adjoint_evaluators import adjoint_eval_cf
import numpy as np
import pickle


if __name__ == "__main__":

    # define operational constraints
    HOME = '/glade/work/mcstanley'
    HOME_RUN = '/glade/u/home/mcstanley/gc_adj_runs/adjoint_model_osb_admm_lep'
    SUB_DIR_FILL = '/runs/v8-02-01/geos5'
    W_SAVE_FP = HOME + '/admm_objects/w_vec.npy'
    GOSAT_DIR = HOME + '/Data/OSSE_OBS'
    W_DIR = HOME + '/admm_objects/w_gen_dir_lep'
    COST_FUNC_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/cfn.01'
    GDT_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/gctm.gdt.01'
    ADJ_RUN_FP = '/glade/u/home/mcstanley/pbs_run_scripts'
    ADJ_RUN_FP += '/run_adjoint_model_osb_admm_lep'
    MNTH_IDX_BND = 8
    YEAR = 2010
    TIME_WAIT = 15

    # hash table creation
    print('Creating hash table with h -> K^T w associations')
    H_TABL_FP = HOME + '/admm_objects/h_table_lep.pkl'
    starting_dict = {}
    with open(H_TABL_FP, 'wb') as f:
        pickle.dump(starting_dict, f)

    # generate a test w vector
    print('Generating test w vector')
    W_INP_FP = HOME + '/admm_objects/fixed_optimization_inputs'
    with open(W_INP_FP + '/affine_correction.npy', 'rb') as f:
        w = np.load(f)

    # execute the adjoint run
    print('Running adjoint model')
    adj_val, adj_cost = adjoint_eval_cf(
        w=w,
        w_save_fp=W_SAVE_FP,
        gosat_dir=GOSAT_DIR,
        w_dir=W_DIR,
        h_tabl_fp=H_TABL_FP,
        cost_func_fp=COST_FUNC_FP,
        gdt_fp=GDT_FP,
        adj_run_fp=ADJ_RUN_FP,
        mnth_idx_bnd=MNTH_IDX_BND,
        year=YEAR,
        time_wait=TIME_WAIT, max_eval_time=28800  # default 8 hrs
    )
    print('--- Results ---')
    print(f'Computed Cost: {adj_cost}')
    print('First 100 elements of K^T w vector')
    print(adj_val[:100])

    print('--- Look in the following locations ---')
    print(f'Hash table: {H_TABL_FP}')
