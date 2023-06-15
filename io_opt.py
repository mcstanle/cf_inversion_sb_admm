"""
Script containing io tools necessary for optimizing with scipy's L-BFGS-B on
GEOS-Chem/Adj.

NOTE: copied from ~/Research/Carbon_Flux/optimization/src/computation

Author        : Mike Stanley
Created       : Apr 6, 2022
Last Modified : Jun 15, 2023
===============================================================================
"""
from copyreg import pickle
import numpy as np
from os.path import exists
import PseudoNetCDF as pnc
import pandas as pd
import pickle as pkl


def write_sfs_to_file(c, fp, n=26496, nprime=298080, nnems_idx=2):
    """
    Write the scaling factor vector coming out of scipy to a scaling factor
    file that can be read in by GEOS-Chem.

    Dimension keys:
    1. n = 8 * 46 * 72 = 26496 (dimension of scipy vector)
    2. n' = 72 * 46 * 9 * 10 = 298080 (dimension of GEOS-Chem vector)

    NOTE:
    1. Scipy is working with 8 x 46 x 72 length vector looping from slowest to
       fastest, months, latitude, longitude.
       - see gpp_file_explore_functional_creation.ipynb
    2. GEOS-Chem needs it 72 x 46 x 9 x 10 looping from slowest to fastest:
       longitude, latitude, months, number of emissions
       - by convention, we put our scaling factors in index 2 [python indexing!
         index 3 in fortran...]
        - see /notebookes/monte_carlo_method_reproduction/
              creating_starting_posititon_from_baseline_flux.ipynb

    TODO:
    1. have a better way to increment the name of file

    Parameters:
    -----------
        c         (np arr) : sf vector coming out of scipy optimizer
        fp        (str)    : file path out
        n         (int)    : expected dimension of incoming vector
        nprime    (int)    : expected dimension of outgoing vector
        nnems_idx (int)    : index of emission of modified scaling factors

    Returns:
    --------
        file_exits (bool) : flag that file exits
    """
    assert c.shape[0] == n

    # put the scipy array into a non-flat vector
    sp_sfs = np.zeros(shape=(8, 46, 72))
    count = 0
    for t in range(8):
        for jjpar in range(46):
            for iipar in range(72):
                sp_sfs[t, jjpar, iipar] = c[count]
                count += 1

    # create the output array
    output_array = np.ones(shape=(72, 46, 9, 10))
    for iipar in range(72):
        for jjpar in range(46):
            for t in range(8):
                output_array[iipar, jjpar, t, nnems_idx] = sp_sfs[
                    t, jjpar, iipar
                ]

    # write the file
    np.savetxt(fname=fp, X=output_array.flatten(order='C'))

    # check that the file exists
    file_exists = exists(fp)

    return file_exists


def create_test_sf_file(val, fp):
    """
    To test the bash scripts that launch the forward and adjoint models,
    this function can create a scaling factor txt file with uniform value.

    NOTE: we assume the same dimensions as those in write_sfs_to_file

    Parameters:
    -----------
        val (float) : uniform value for the array
        fp  (str)   : output location

    Returns:
    --------
        file_exists (bool) : flag that file exists
    """
    # create uniform value array
    unf_val_arr = np.ones(26496) * val

    # write the file
    file_exists = write_sfs_to_file(
        c=unf_val_arr,
        fp=fp,
        n=26496, nprime=298080, nnems_idx=2
    )

    return file_exists


def read_cfn_file(cfn_fp):
    """
    Reads the cost function evaluation from the cfn.01 file.

    NOTE: only expects one iteration.

    Parameters:
    -----------
        cfn_fp (str) : file path to the cost function output

    Returns:
    --------
        cf_val (float)
    """
    with open(cfn_fp, 'r') as f:
        cfn_ls = f.readlines()

    assert len(cfn_ls) == 1

    return float(
        cfn_ls[0].replace(' 1 ', '').replace(' ', '').replace('\n', '')
    )


def compute_forward_cost(model_fp, gosat_df_fp):
    """
    Once the forward model has been run, this function
    1. retrieves the modeled file gctm.model.01
    2. matches the values in the above to the true observations
    3. computes the forward cost function - (y - Kc)^T R^{-1} (y - Kc)

    NOTE: we assume that the variable name in the bpch file is IJ-AVG-$_CO2

    NOTE: assume that modeled_obs has the form [@ , day idx, lat idx, lon idx]

    Parameters:
    -----------
        model_fp    (str) : file path to gctm.model.01
        gosat_df_fp (str) : file path to gosat_df.csv

    Returns:
    --------
        y      (np arr) : true gosat observations
        Kc     (np arr) : modeled observations
        f_cost (float)  : quadratic cost defined above
    """
    # read in the gctm.model.01 file
    model = pnc.pncopen(model_fp, format='bpch')
    modeled_obs = model.variables["IJ-AVG-$_CO2"].array()

    # read in the gosat df
    gosat_df = pd.read_csv(gosat_df_fp)
    y = gosat_df['xco2'].values / 1e6  # this is the scaling done in GEOS-Chem

    # match the modeled obs with the true
    Kc = gosat_df.apply(
        lambda x: modeled_obs[0, x['date_idx'], x['lat_idx'], x['lon_idx']],
        axis=1
    ).values / 1e9

    # multiply by the inverse std for each observation
    R_OER_INV = (1 / (gosat_df['xco2_unc'] * 1e-6) ** 2)

    # compute the quadratic cost
    y_Kc = y - Kc
    force = R_OER_INV * y_Kc
    f_cost = np.dot(y_Kc, force)

    return y, Kc, f_cost


def flatten_gdt(gdt_arr):
    """
    Correctly flattens the gdt file (i.e., month, lat, lon, from slowest)
    to fastest.

    Parameters:
    -----------
        gdt_arr (np arr) : month x latitude x longitude

    Returns:
    --------
        gdt_flat (np arr) : month * latitude * longitude
    """
    gdt_shape = gdt_arr.shape
    gdt_flat = np.zeros(shape=(gdt_shape[0] * gdt_shape[1] * gdt_shape[2]))

    count = 0
    for t in range(gdt_shape[0]):  # month
        for i in range(gdt_shape[1]):  # latitude
            for j in range(gdt_shape[2]):  # longitude
                gdt_flat[count] = gdt_arr[t, i, j]
                count += 1

    return gdt_flat


def move_opt_c_to_start_pos(res_pos, c_write_loc):
    """
    Take optimized scaling factor vector from an iteration and move it to a
    directory where starting positions can be read.

    Parameters:
    -----------
        res_pos     (str) : location of the optimization results
        c_write_loc (str) : location to write the starting position numpy arr

    Returns:
    --------
        file_created (bool) : indicator that file was created
    """
    # read in the optimization results
    with open(res_pos, 'rb') as f:
        res = pickle.load(f)

    # write the optimized value to the starting position location
    np.save(file=c_write_loc, arr=res[0])

    return exists(c_write_loc)


def get_KTwk1(w, adjoint_ht_fp):
    """
    Obtain the last K^Tw_{k + 1} for the next c sub-opt.

    Parameters
    ----------
        w             (np arr) : w for which we want to find K^Tw
        adjoint_ht_fp (str)    : file path to the adjoint eval hash table

    Returns
    -------
        K^T w (np arr) : p x 1
    """
    # read in the hash table
    with open(adjoint_ht_fp, 'rb') as f:
        adjoint_ht = pkl.load(f)

    w_hash = hash(w.tobytes())
    return adjoint_ht[w_hash]
