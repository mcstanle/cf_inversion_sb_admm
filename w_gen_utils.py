"""
Utility functions to support the generation of w vector.
===============================================================================
Author        : Mike Stanley
Created       : Jun 14, 2023
Last Modified : Aug 17, 2023
===============================================================================
"""
from carbonfluxtools.io_utils import create_gosat_df_year, get_ij
from datetime import datetime
from glob import glob
import numpy as np
import os
import pandas as pd
import pathlib
from tqdm import tqdm


def read_gosat_data(fp):
    """
    Read in and process GOSAT Data.

    Parameters
    ----------
        fp (str) : file path to GOSAT Obs

    Returns
    -------
        List of lists with satellite observations
    """
    # read in the file
    gs_data = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            gs_data.append(line.replace('\n', '').split(', '))

    # convert strings to floats
    gs_data = [[float(num) for num in line] for line in gs_data]

    return gs_data


def alter_xco2(gosat_obs, new_xco2):
    """
    Expects a GOSAT file consistent with the 7-part format

    Parameters
    ----------
        gosat_obs (list)
        new_xco2  (float)

    Returns
    -------
        list with same format as gosat_obs but with the new xco2 value

    NOTE:
    - make sure to send in copy of gosat_obs
    """
    assert len(gosat_obs) == 7

    # create new list for altered observation
    altered_obs = [sub_obs[:] for sub_obs in gosat_obs]

    # alter the XCO2 value
    altered_obs[0][4] = new_xco2

    return altered_obs


def create_gosat_day(obs_list, modeled_obs):
    """
    Creates a new GOSAT observation day file. These are organized
    such that one observation occurs every 7 elements in the list.

    NOTE: copied from generate_gosat_data.py with some modification

    Parameters
    ----------
        obs_list    (list)      : all observations
        modeled_obs (numpy arr) : sequence of xco2 values for day

    Returns
    -------
        New list of the form obs_list -- new meaning new XCO2 values
    """
    # create indices to loop over
    start_idx = np.arange(0, len(obs_list), step=7)
    end_idx = np.arange(7, len(obs_list) + 1, step=7)

    assert len(start_idx) == len(end_idx)
    assert len(modeled_obs) == len(start_idx)

    obs_idx = zip(start_idx, end_idx)

    altered_obs = []
    count = 0
    for start, end in obs_idx:

        # pull out the GOSAT observation
        gosat_obs = obs_list[start:end]

        altered_obs.extend(
            alter_xco2(
                gosat_obs=gosat_obs,
                new_xco2=modeled_obs[count]
            )
        )
        count += 1

    assert len(altered_obs) == len(obs_list)
    return altered_obs


def write_gosat_day(obs_list, write_path):
    """
    Writes a gosat observation list to file

    NOTE: copied from /dev/utils/generate_gosat_data.py

    Parameters:
    -----------
        obs_list   (list) : list of observations as created by
                            create_gosat_day()
        write_path (str)  : file path to file
        precision  (int)  : number of decimal points that the value is carried
                            until

    Returns:
    --------
        write_path

    NOTES:
    - for each row written, the files should be comma delimited
    - hard-code 5 decimal point precision
    """
    path = pathlib.Path(write_path)

    assert path.parents[0].is_dir()   # make sure the given directory exists

    with open(path, mode='w') as f:
        for line in obs_list:
            f.write(
                ', '.join([f"{i:.14f}" for i in line]) + '\n'
            )

    return write_path


# @profile
def create_gosat_files(
    xco2_fp,
    origin_dir, save_dir,
    year, month,
    gosat_file_form='GOSAT_OSSE_',
    disable_tqdm=True
):
    """
    With the input xco2 values stored in npy file (xco2_fp), create
    a GOSAT file with the desired "xco2" values. The idea here is that one can
    substitute in an arbitrary vector depending on what kind of xco2
    values one wants to make.

    NOTE: this function is copied from ../src/conjugacy/generate_data.py.

    NOTE: Uses dates from 1/1/{year} to {month - 1}/31/{year}

    Parameters
    ----------
        xco2_fp         (str)  : func to create new xco2 values in gosat files
        gosat_df_fp     (str)  : file path to gosat observations
        origin_dir      (str)  : template GOSAT observation files to be changed
        save_dir        (str)  : location of directory to be constructed
        year            (int)  : GOSAT obs year of interest
        month           (int)  : least upper bound (not inclusive) month
        gosat_file_form (str)  : prefix for gosat observation files
        disable_tqdm    (bool) : toggle progress bar for file generation

    Returns
    -------
        bool if directory and all files made successfully.
    """
    assert os.path.isdir(origin_dir)
    assert os.path.isdir(save_dir)

    # create gosat observation dataframe
    gosat_raw = create_gosat_df_year(
        obs_dir=origin_dir,
        year=year
    )

    # only use up until the desired month
    gosat_df = gosat_raw.loc[gosat_raw.omonth < month].copy()

    # drop the useless columns
    gosat_df.drop(
        [
            'date_field1', 'date_field2', 'date_field3',
            'date_field4', 'date_field5', 'date_field6'
        ],
        axis=1,
        inplace=True
    )

    # make the date more usable
    gosat_df.rename(
        columns={'oyear': 'year', 'omonth': 'month', 'oday': 'day'},
        inplace=True
    )

    # find lon/lat -- using GET_IJ function
    lon_lat_idx_ser = gosat_df.apply(
        lambda x: get_ij(lon=x['lon'], lat=x['lat']), axis=1
    )
    lon_lat_idx_df = pd.DataFrame(
        lon_lat_idx_ser.tolist(), columns=['lon_idx', 'lat_idx']
    )

    # join back with original data
    gosat_df = gosat_df.join(lon_lat_idx_df)

    # make date index
    dates = pd.date_range(f"1/1/{year}", f"{month - 1}/31/{year}")
    date_idx = pd.Series(np.arange(len(dates)))
    date_idx.index = dates

    # make datetime
    gosat_df['date'] = pd.to_datetime(gosat_df[['year', 'month', 'day']])
    gosat_df['date_str'] = gosat_df['date'].apply(
        lambda i: datetime.strftime(i, '%Y%m%d')
    )

    # find the day index for each GOSAT observation
    gosat_df['date_idx'] = gosat_df.apply(
        lambda row: date_idx[row['date']], axis=1
    )

    # capture the unique date strings
    dates = list(gosat_df['date_str'].unique())

    # write the gosat file to csv
    gosat_fp = save_dir + '/gosat_df.csv'
    gosat_df.to_csv(gosat_fp)
    assert os.path.exists(gosat_fp)

    # read in new xco2 values
    with open(xco2_fp, 'rb') as f:
        new_xco2 = np.load(f)

    # for each existing file, generate a new one
    all_files_exist = True
    for idx, date in tqdm(enumerate(dates), disable=disable_tqdm):

        # create input and output file names
        input_fp = origin_dir + '/' + gosat_file_form + date + '.txt'
        output_fp = save_dir + '/' + gosat_file_form + date + '.txt'

        # check to see if generated date has origin file
        if not os.path.exists(input_fp):
            continue

        # read in the original observation
        gos_orig = read_gosat_data(fp=input_fp)

        # get the desired indices for the day
        day_idx = gosat_df['date_str'] == date

        # create a new observation list with the modeled XCO2
        # loops through all observations on that day
        gos_new = create_gosat_day(
            obs_list=gos_orig,
            # modeled_obs=new_xco2.loc[day_idx].values  # expects np arr
            modeled_obs=new_xco2[day_idx]
        )

        # write the new file
        write_gosat_day(
            obs_list=gos_new,
            write_path=output_fp
        )

        # check that the new file exists
        if not os.path.exists(output_fp):
            all_files_exist = False

    # remove the saved gosat file
    os.remove(save_dir + '/gosat_df.csv')

    return all_files_exist

@profile
def create_gosat_files_lite(
    xco2_fp,
    origin_dir, save_dir,
    year, month,
    disable_tqdm=True
):
    """
    Lighter version of create_gosat_files() above. Does not use pandas.

    With the input xco2 values stored in npy file (xco2_fp), create
    a GOSAT file with the desired "xco2" values. The idea here is that one can
    substitute in an arbitrary vector depending on what kind of xco2
    values one wants to make.

    NOTE: this function simply copies the directory structure of the given on
          specified by origin_dir.

    NOTE: this implementation was developed and verified in Research/
          Carbon_Flux/optimization/
          new_gosat_file_creation_dev_and_test.ipynb

    Parameters
    ----------
        xco2_fp         (str)  : func to create new xco2 values in gosat files
        origin_dir      (str)  : template GOSAT observation files to be changed
        save_dir        (str)  : location of directory to be constructed
        year            (int)  : GOSAT obs year of interest
        month           (int)  : least upper bound (not inclusive) month
        disable_tqdm    (bool) : toggle progress bar for file generation

    Returns
    -------
        bool if directory and all files made successfully.
    """
    assert os.path.isdir(origin_dir)
    assert os.path.isdir(save_dir)

    # obtain list of sorted file names up until the end of month of interest
    fps_year = [i for i in glob(origin_dir + '/*') if int(i[-12:-8]) == year]
    dates = [i.split('/')[-1].split('.')[0][-8:] for i in fps_year]
    fps_sorted = [
        file_i for file_i, date_i in sorted(
            zip(fps_year, dates), key=lambda x: x[1]
        )
    ]
    end_date_idx = [
        idx for idx, date in enumerate(sorted(dates))
        if date[4:6] == str(month).zfill(2)
    ][0]

    fps = fps_sorted[:end_date_idx]

    # read in the new values to plug into new GOSAT files
    with open(xco2_fp, 'rb') as f:
        new_vals = np.load(f)

    # make new files
    start_idx = 0
    end_idx = 0
    all_files_exist = True
    for fp_i in tqdm(fps, disable=disable_tqdm):

        # import current data
        gos_orig_i = read_gosat_data(fp=fp_i)

        # determine the number of observations
        end_idx += len(gos_orig_i) // 7

        # create new gosat data
        gos_new = create_gosat_day(
            obs_list=gos_orig_i,
            modeled_obs=new_vals[start_idx:end_idx]
        )

        # write out
        output_fp = save_dir + '/' + fp_i.split('/')[-1]
        write_gosat_day(
            obs_list=gos_new,
            write_path=output_fp
        )

        # check that the new file exists
        if not os.path.exists(output_fp):
            all_files_exist = False

        # reset indices
        start_idx = end_idx

    return all_files_exist


if __name__ == "__main__":

    # define some testing values
    HOME = '/glade/work/mcstanley'
    XCO2_FP = HOME + "/admm_objects/test_w_file.npy"
    ORIGIN_DIR = HOME + "/Data/OSSE_OBS"
    SAVE_DIR = HOME + "/admm_objects/w_gen_directory"
    YEAR = 2010
    MONTH = 9

    # dimension of test observation -- how many expected xco2 observations
    TEST_DIM = 28267

    # create test w file
    with open(XCO2_FP, 'wb') as f:
        np.save(file=f, arr=np.ones(28267))

    # create files
    # files_created = create_gosat_files(
    #     xco2_fp=XCO2_FP,
    #     origin_dir=ORIGIN_DIR,
    #     save_dir=SAVE_DIR,
    #     year=YEAR,
    #     month=MONTH,
    #     disable_tqdm=False
    # )

    # create files using the "lite" framework
    files_created = create_gosat_files_lite(
        xco2_fp=XCO2_FP,
        origin_dir=ORIGIN_DIR,
        save_dir=SAVE_DIR,
        year=YEAR,
        month=MONTH,
        disable_tqdm=False
    )

    print("Files created successfully: %r" % files_created)
