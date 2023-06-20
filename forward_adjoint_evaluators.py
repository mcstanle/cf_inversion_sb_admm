"""
Defining forward and adjoint evaluation wrappers for different applications.
===============================================================================
Currently supported applications:
1. Unfolding
    -> forward function: forward_eval_unfold()
    -> adjoint function: adjoint_eval_unfold()
2. Carbon Flux inversion
    -> forward function: forward_eval_cf()
    -> adjoint function: adjoint_eval_cf()
===============================================================================
Author        : Mike Stanley
Created       : May 26, 2023
Last Modified : Jun 20, 2023
===============================================================================
"""
from io_opt import read_cfn_file, write_sfs_to_file
import numpy as np
from os.path import exists
import pickle
import PseudoNetCDF as pnc
import subprocess
import time
from w_gen_utils import create_gosat_files


def forward_eval_unfold(x, K, L_inv):
    """
    Performs a forward evaluation for the unfolding problem.

    Parameters
    ----------
        x     (np arr) : p x 1 - input vector
        K     (np arr) : m x p - forward model matrix
        L_inv (np arr) : m x m - inverse of l-tri cholesky factor

    Returns
    -------
        Kx (np arr) : m x 1
    """
    return L_inv @ K @ x


def adjoint_eval_unfold(w, K, L_inv, h_tabl_fp):
    """
    Performs an adjoint evaluation for the unfolding problem.

    Saves the K^Tw evaluation to a dictionary key'ed by hash(w.tobytes()).

    Parameters
    ----------
        w         (np arr) : m x 1 - input vector
        K         (np arr) : m x p - forward model matrix
        L_inv     (np arr) : m x m - inverse of l-tri cholesky factor
        h_tabl_fp (str)    : filepath to pickled dictionary hash table

    Returns
    -------
        K^Tw (np arr) : p x 1
    """
    # evaluate the adjoint
    KTw = K.T @ L_inv @ w

    # read in file
    with open(h_tabl_fp, 'rb') as f:
        adjoint_ht = pickle.load(f)

    # add new key value pair
    key = hash(w.tobytes())
    adjoint_ht[key] = KTw

    # write out new updated hash table
    with open(h_tabl_fp, 'wb') as f:
        pickle.dump(adjoint_ht, f)

    return KTw


def forward_eval_cf(
    c, gc_dir, sf_fp, fm_run_fp, time_wait,
    max_et=28800  # default 8 hr
):
    """
    Calls GEOS-Chem Forward model.

    Inspired from ~/Research/Carbon_Flux/optimization/src/optimizer.py

    NOTE:
    1. SF file is /OptData/gctm.sf.01
    2. model file is /xco2_modeled.txt

    NOTE: f(c) is scaled up by 10^6 and flipped such that the indices move from
    oldest to newest.

    Parameters
    ----------
        c         (np arr) : input scaling factor vector
        gc_dir    (str)    : directory where GEOS-Chem is run and where
                             gctm.sf.01 is found
        sf_fp     (str)    : file path to where scaling factor is saved
        fm_run_fp (str)    : forward model run script file path
        time_wait (float)  : time to wait between each check for the
                             existence of gctm.sf.01
        max_et    (int)    : maximum time (seconds) the code will wait

    Returns
    -------
        f_c (np arr) : forward model evaluated at c -- f(c)
    """
    SF_FILE = gc_dir + '/runs/v8-02-01/geos5/OptData/gctm.sf.01'
    MODEL_FILE = gc_dir + '/runs/v8-02-01/geos5/xco2_modeled.txt'

    # delete any currently existing scaling factor/model files
    subprocess.run([
        "rm",
        SF_FILE
    ])
    subprocess.run([
        "rm",
        MODEL_FILE
    ])

    # write the input scaling factors to file
    files_succ_written = write_sfs_to_file(
        c=c,
        fp=sf_fp,
        n=26496, nprime=298080, nnems_idx=2
    )
    assert files_succ_written  # verify that the scaling factor file exists

    # call the forward model
    subprocess.run([
        "qsub",
        fm_run_fp
    ])

    # wait to make sure gctm.sf.01 file exists
    total_sleep = 0
    while not exists(SF_FILE):
        time.sleep(time_wait)
        total_sleep += time_wait

        if total_sleep > max_et:
            raise RuntimeError

    # read in xco2_modeled.txt file
    with open(MODEL_FILE, 'rb') as f:
        f_c = np.loadtxt(f)

    # scale appropriately
    f_c = f_c * 1e6

    # flip so going from oldest to newest observations
    f_c = np.flip(f_c)

    return f_c


def forward_linear_eval_cf(
        c, aff_corr_fp, gc_dir, sf_fp, fm_run_fp, time_wait,
        max_et=28800  # default 8 hr
):
    """
    Returns only the linear component of the affine forward model, i.e.,
        Kc = f(c) - b.

    NOTE: expects b to be contained in npy file.

    Parameters
    ----------
        c           (np arr) : input scaling factor vector
        aff_corr_fp (str)    :
        gc_dir      (str)    : directory where GEOS-Chem is run and where
                               gctm.sf.01 is found
        sf_fp       (str)    : file path to where scaling factor is saved
        fm_run_fp   (str)    : forward model run script file path
        time_wait   (float)  : time to wait between each check for the
                               existence of gctm.sf.01
        max_et      (int)    : maximum time (seconds) the code will wait

    Returns
    -------
        f_c - b (np arr) : forward model minus affine term
    """
    # call the forward model
    f_c = forward_eval_cf(
        c=c,
        gc_dir=gc_dir,
        sf_fp=sf_fp,
        fm_run_fp=fm_run_fp,
        time_wait=time_wait,
        max_et=max_et
    )

    # read in the affine correction
    with open(aff_corr_fp, 'rb') as f:
        b = np.load(f)

    return f_c - b


def adjoint_eval_cf(
        w, w_save_fp, gosat_dir, w_dir, h_tabl_fp,
        cost_func_fp, gdt_fp, adj_run_fp, mnth_idx_bnd, year,
        time_wait, max_eval_time=28800  # default 8 hrs
):
    """
    Adjoint model evaluation for carbon flux inversion.

    Input is a vector w, and output is the GDT file from GEOS-Chem Adjoint.

    Inputs  -- w vector
    Outputs --
        - .npy file of w vector
        - GOSAT directory containing w vector
        - key value pair in hash table with w: K^Tw

    Parameters
    ----------
        w             (np arr) : opt variable in first ADMM subopt
        w_save_fp     (str)    : where to save w vector as npy file
        gosat_dir     (str)    : directory containing template GOSAT files
        w_dir         (str)    : directory where w is saved as faux gosat files
        h_tabl_fp     (str)    : file path to hashtable for w:k^Tw pairs
        cost_func_fp  (str)    : file path to cfn.01
        gdt_fp        (str)    : path to the gradient file (gctm.gdt.01)
        adj_run_fp    (str)    : adjoint model run script file path
        mnth_idx_bnd  (int)    : upper bound (inclusive) of month index
                                 (i.e. 8 == aug)
        year          (int)    : year of interest
        time_wait     (float)  : time to wait between each check for the
                                 existence of cfn.01
        max_eval_time (int)    : maximum time (seconds) the code will wait

    Returns
    -------
        adj_val_flat (np arr) : K^T w
        adj_cost     (float)  : cost function evaluation

    """
    # TODO - test out if we need a multiplier on w
    # create file containing w vector
    with open(w_save_fp, 'wb') as f:
        np.save(file=f, arr=w)

    # generate gosat files containing new w vector
    files_created = create_gosat_files(
        xco2_fp=w_save_fp,
        origin_dir=gosat_dir,
        save_dir=w_dir,
        year=year,
        month=mnth_idx_bnd
    )
    assert files_created

    # delete any currently existing cost/adj file
    subprocess.run([
        "rm",
        cost_func_fp
    ])
    subprocess.run([
        "rm",
        gdt_fp
    ])

    # call the adjoint model
    subprocess.run([
        "qsub",
        adj_run_fp
    ])

    # wait to make sure cfn.01 and gctm.gdt.01 files exists
    total_sleep = 0
    while (not exists(cost_func_fp)) or (not exists(gdt_fp)):
        time.sleep(time_wait)
        total_sleep += time_wait

        if total_sleep > max_eval_time:
            raise RuntimeError

    # read in cfn.01
    adj_cost = read_cfn_file(cfn_fp=cost_func_fp)
    adj_cost *= 2  # the above returns 1/2 the appropriate value

    # obtain the gradient file
    gdt = pnc.pncopen(gdt_fp, format='bpch')
    adj_val = gdt.variables['IJ-GDE-$_CO2bal'].array()[
        0, :(mnth_idx_bnd - 1), :, :
    ]

    # flatten in C-style (i.e., longitude expanding first)
    adj_val_flat = adj_val.flatten(order='C')

    # read in file
    with open(h_tabl_fp, 'rb') as f:
        adjoint_ht = pickle.load(f)

    # add new key value pair
    key = hash(w.tobytes())
    adjoint_ht[key] = adj_val

    # write out new updated hash table
    with open(h_tabl_fp, 'wb') as f:
        pickle.dump(adjoint_ht, f)

    return adj_val_flat, adj_cost
