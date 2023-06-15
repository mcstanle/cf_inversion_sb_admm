"""
Script to kick off optimization jobs for the carbon flux problem. This script
is called for LEP optimization.
===============================================================================
Author        : Mike Stanley
Created       : Jun 15, 2023
Last Modified : Jun 15, 2023
===============================================================================
"""
from admm_optimizer import run_admm
from forward_adjoint_evaluators import forward_linear_eval_cf, adjoint_eval_cf
from functools import partial
from io_opt import get_KTwk1
import os


def check_directories(sat_obs):
    """
    Checks for the existence of necessary directories.

    Parameters
    ----------
        sat_obs (str) : satellite observation directory

    Returns
    -------
        None -- halts code if one directory does not exist.
    """
    assert os.path.isdir(sat_obs)


if __name__ == "__main__":

    # operational parameters
    LEP_OPT = True
    MAX_ITERS = 1
    SUBOPT_ITERS = 1
    TIME_2_WAIT = 15  # seconds between each check for file existence
    YEAR = 2010
    MONTH_IDX = 8

    # define necessary directories
    HOME = '/glade/u/home/mcstanley'
    HOME_RUN = HOME + '/gc_adj_runs/adjoint_model_osb_admm_lep'
    SUB_DIR_FILL = '/runs/v8-02-01/geos5'
    WORK = '/glade/work/mcstanley'
    SAT_OBS = WORK + '/Data/OSSE_OBS'
    GC_DIR = HOME + "/gc_adj_runs/forward_model_osb_lep"
    W_DIR = WORK + '/admm_objects/w_gen_dir_lep'

    # define necessary file paths
    AFFINE_CORR_FP = WORK + '/admm_objects/fixed_optimization_inputs'
    AFFINE_CORR_FP += '/affine_correction.npy'
    SF_F_FP = WORK + "/Data/osb_endpoint_sfs/lep_sfs_forward.txt"
    FM_RUN_FP = HOME + '/pbs_run_scripts/run_forward_model_osb_lep'
    ADJ_RUN_FP = HOME + '/pbs_run_scripts/run_adjoint_model_osb_admm_lep'
    W_SAVE_FP = WORK + '/admm_objects/w_vec.npy'
    COST_FUNC_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/cfn.01'
    GDT_FP = HOME_RUN + SUB_DIR_FILL + '/OptData/gctm.gdt.01'

    # check if necessary directories exist
    check_directories(
        sat_obs=SAT_OBS
    )

    # directory for adjoint eval hash table
    ADJOINT_EVAL_HT_FP = WORK + '/admm_objects/h_table_lep.pkl'

    # create wrappers around fuctions involving K matrix
    forward_eval = partial(
        forward_linear_eval_cf,
        aff_corr_fp=AFFINE_CORR_FP,
        gc_dir=GC_DIR,
        sf_fp=SF_F_FP,
        fm_run_fp=FM_RUN_FP,
        time_wait=TIME_2_WAIT,
        max_et=28800
    )
    adjoint_eval = partial(
        adjoint_eval_cf,
        w_save_fp=W_SAVE_FP,
        gosat_dir=SAT_OBS,
        w_dir=W_DIR,
        h_tabl_fp=ADJOINT_EVAL_HT_FP,
        cost_func_fp=COST_FUNC_FP,
        gdt_fp=GDT_FP,
        adj_run_fp=ADJ_RUN_FP,
        mnth_idx_bnd=MONTH_IDX,
        year=YEAR,
        time_wait=TIME_2_WAIT,
        max_eval_time=28800
    )
    get_KTwk1_par = partial(
        get_KTwk1, adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )

    # run optimization
    print('Running ADMM...')
    res_dict = run_admm(
        y=y_tilde,
        A=A,
        b=b,
        h=h,
        w_start=w_sp,
        c_start=c_sp,
        lambda_start=lambda_sp,
        mu=MU,
        psi_alpha=np.sqrt(psi_alpha_sq),
        forward_eval=forward_eval,
        adjoint_eval=adjoint_eval,
        get_KTwk1=get_KTwk1_par,
        lep=LEP_OPT,
        max_iters=MAX_ITERS,
        subopt_iters=SUBOPT_ITERS,
        adjoint_ht_fp=ADJOINT_EVAL_HT_FP
    )