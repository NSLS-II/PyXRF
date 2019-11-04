import os
import re

from .load_data_from_db import make_hdf
from .command_tools import pyxrf_batch
from .fileio import read_hdf_APS

import logging
logger = logging.getLogger()


def build_energy_map(*args, **kwargs):
    """
    A wrapper for the function ``build_energy_map_api`` that catches exceptions
    and prints the error message. Use this wrapper to run processing manually from
    iPython and use ``build_energy_map_api`` for custom scripts.

    For description of the function parameters see the docstring for
    ``build_energy_map_api``
    """
    try:
        build_energy_map_api(*args, **kwargs)
    except BaseException as ex:
        msg = f"Processing is incomplete! Exception was raised during execution:\n   {ex}"
        logger.error(msg)
    else:
        logger.info("Processing was completed successfully.")


def build_energy_map_api(start_id=None, end_id=None, *, param_file_name,
                         scaler_name=None, wd=None, sequence="build_energy_map",
                         processed_channel="sum"):

    """

    sequence : str
        the sequence of operations performed by the function:

        -- ``load_and_process`` - loading data and full processing,

        -- ``process`` - full processing of data, including xrf mapping
        and building the energy map,

        -- ``build_energy_map`` - build the energy map using xrf mapping data,
        all data must be processed.

    """

    if wd is None:
        wd = '.'
    else:
        wd = os.path.expanduser(wd)

    wd = os.path.abspath(wd)

    param_file_name = os.path.expanduser(param_file_name)
    param_file_name = os.path.abspath(param_file_name)

    seq_load_data = True
    seq_process_xrf_data = True
    seq_build_energy_map = True
    if sequence == "load_and_process":
        pass
    elif sequence == "process":
        seq_load_data = False
    elif sequence == "build_energy_map":
        seq_load_data = False
        seq_process_xrf_data = False
    else:
        ValueError(f"Unknown sequence name '{sequence}' is passed as a parameter "
                   "to the function 'build_energy_map_api'.")

    # XRF data will be placed in the subdirectory 'xrf_data' of the directory 'wd'
    wd_xrf = os.path.join(wd, "xrf_data")

    if seq_load_data:
        logger.info("Loading data from databroker ...")
        # Try to create the directory (does nothing if the directory exists)
        os.makedirs(wd_xrf, exist_ok=True)
        files_h5 = [fl.path for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
        if files_h5:
            # TODO: may be an option to enable/disable automatic removal should be added
            logger.warning(f"The temporary directory '{wd_xrf}' is not empty. "
                           f"Deleting {len(files_h5)} files (.h5) ...")
            for fln in files_h5:
                logger.info(f"Removing raw xrf data file: '{fln}'.")
                os.remove(path=fln)
        make_hdf(start_id, end_id, wd=wd_xrf,
                 completed_scans_only=True, file_overwrite_existing=True,
                 create_each_det=False, save_scaler=True)
    else:
        logger.info("Skipping loading of data from databroker ...")

    if seq_process_xrf_data:
        # Make sure that the directory with xrf data exists
        if not os.path.isdir(wd_xrf):
            # Unfortunately there is no way to continue if there is no directory with data
            raise IOError(f"XRF data directory '{wd_xrf}' does not exist.")
        # Process ALL .h5 files in the directory 'wd_xrf'. Processing results are saved
        #   as additional datasets in the original .h5 files.
        pyxrf_batch(param_file_name=param_file_name, wd=wd_xrf, save_tiff=False)

    if seq_build_energy_map:
        logger.info("Building energy map ...")

        # Load the datasets from file located in 'wd_xrf'

        # File names in the directory 'wd_xrf'
        files_h5 = [fl.name for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
        scan_ids = []
        scan_energy = []
        scan_img_dict = []
        scan_dataset = []
        for fln in files_h5:
            img_dict, dataset, mdata = \
                read_hdf_APS(working_directory=wd_xrf, file_name=fln, load_summed_data=True,
                             load_each_channel=False, load_processed_each_channel=False,
                             load_raw_data=False, load_fit_results=True,
                             load_roi_results=False)
            scan_img_dict.append(img_dict)
            scan_dataset.append(dataset)
            if "scan_id" in mdata:
                scan_ids.append(mdata["scan_id"])
            else:
                scan_ids.append("")
            if "instrument_mono_incident_energy" in mdata:
                scan_energy.append(mdata["instrument_mono_incident_energy"])
            else:
                scan_energy.append(-1.0)

        # Check if the set of scans is valid
        #   TODO: finish this part and move it to a separate function
        index_no_energy = []  # Indices of scans that contain no energy data
        index_no_processed_data = []
        index_no_scaler = []
        index_no_positions = []

        for n, e in enumerate(scan_energy):
            if e < 0:
                index_no_energy.append(n)
        if index_no_energy:
            msg = "Incident beam energy is not present in some HDF5 files. Check scans with IDs: "\
                  f"{[scan_ids[n] for n in index_no_energy]}"
            raise(RuntimeError(msg))

        print(f"Collected scan IDs: {scan_ids}")
        print(f"Collected scan energies: {scan_energy}")

        # Sort the lists based on energy
        #print(f"{sorted(zip(scan_energy, range(len(scan_energy))))}")
        scan_energy, sorted_indexes = list(zip(*sorted(zip(scan_energy, range(len(scan_energy))))))
        files_h5 = [files_h5[n] for n in sorted_indexes]
        scan_ids = [scan_ids[n] for n in sorted_indexes]
        scan_img_dict = [scan_img_dict for n in sorted_indexes]
        scan_dataset = [scan_dataset for n in sorted_indexes]

        print(f"scan_energy={scan_energy}")
        print(f"sorted_indexes={sorted_indexes}")


        print(f"scan_ids={scan_ids}")
        print(f"files_h5={files_h5}")

        eline_list = get_eline_list(files_h5[0], scan_img_dict[0])
        print(f"elines: {eline_list}")
        print(f"scan_ids={scan_ids}")
        print(f"files_h5={files_h5}")



        # Create common position grid

        # Interpolate all the scans to common grid


def get_dataset_name(file_name, detector=None):
    if detector is None:
        # The dataset is obtained by processing the sum of all channels
        suffix = "fit"
    else:
        raise ValueError(f"Unknown detector {detector} (function 'get_dataset_name')")
    base_fln = os.path.splitext(os.path.basename(file_name))[0]
    return f"{base_fln}_{suffix}"


def get_eline_list(file_name, img_dict):

    dataset_name = get_dataset_name(file_name)
    all_keys = img_dict[dataset_name].keys()
    # The following pattern is based on assumption that elines names
    #        are of the form Si_K, La_L, P_K etc.
    re_pattern = "^[A-Z][a-z]?_[KLM]$"
    return [key for key in all_keys if re.search(re_pattern, key)]
