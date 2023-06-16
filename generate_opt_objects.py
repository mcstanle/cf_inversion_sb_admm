"""
This script can generate the following objects which are inputs to the
endpoint optimizations.
1. y_generation --> GOSAT - affine correction observations to use in algo
2. A_b_generation --> polyhedra constraints for scaling factors
3. h_generation --> functional generation
4. starting_point_generation --> starting vectors for w, c, and lambda
===============================================================================
Author        : Mike Stanley
Created       : Jun 16, 2023
Last Modified : Jun 16, 2023
===============================================================================
"""
from carbonfluxtools.io_utils import create_gosat_df_year
import numpy as np
import os


def y_generation(aff_corr_fp, gosat_dir, write_fp):
    """
    GOSAT Observation minus affine correction.

    NOTE: expects affine correction to be npy file

    Parameters
    ----------
        aff_corr_fp (str) : file location of affine correction
        gosat_dir   (str) : location of GOSAT observation files
        write_fp    (str) : file destination

    Returns
    -------
        file exists (bool)
    """
    assert os.path.exists(aff_corr_fp)
    assert os.path.isdir(gosat_dir)
    assert not os.path.exists(write_fp)

    # read in affine correction
    with open(aff_corr_fp, 'rb') as f:
        b = np.load(f)

    # read in gosat observations
    gosat_df = create_gosat_df_year(obs_dir=gosat_dir, year=2010)

    # corrected observations
    y = gosat_df['xco2'].values - b

    # write to file
    with open(write_fp, 'rb') as f:
        np.save(file=f, arr=y)

    return os.path.exists(write_fp)


def A_b_generation():
    """
    Polyhedra constraints for scaling factors.
    """
    pass


def h_generation():
    """
    Generate functional of interest.
    """
    pass


def starting_point_generation():
    """
    Starting vectors for w, c, lambda.
    """


if __name__ == '__main__':

    # base directories
    BASE_DIR = '/glade/work/mcstanley'
    OBJ_DEST_DIR = BASE_DIR + '/admm_objects/fixed_optimization_inputs'

    # affine corrected observation
    AFF_CORR_FP = OBJ_DEST_DIR + '/affine_correction.npy'
    GOSAT_DIR = BASE_DIR + '/Data/OSSE_OBS'
    OBS_WRITE_FP = OBJ_DEST_DIR + '/y_affine_corrected.npy'

    print('Observation generation...')
    y_generation(
        aff_corr_fp=AFF_CORR_FP,
        gosat_dir=GOSAT_DIR,
        write_fp=OBS_WRITE_FP
    )

    # constraints
    A_b_generation()

    # functional
    h_generation()

    # starting points
    starting_point_generation()
