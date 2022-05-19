from __future__ import absolute_import, division, print_function, unicode_literals

import h5py
import numpy as np
import os
import json
import multiprocessing
import pandas as pd
import platform
import math
import time as ttime
import copy
import re
from distutils.version import LooseVersion

import logging
import warnings

try:
    import databroker
except ImportError:
    pass

from ..core.utils import convert_time_to_nexus_string
from .scan_metadata import ScanMetadataXRF

import pyxrf

pyxrf_version = pyxrf.__version__

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

sep_v = os.sep

try:
    beamline_name = None

    # Attempt to find the configuration file first
    config_path = "/etc/pyxrf/pyxrf.json"
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as beamline_pyxrf:
                beamline_config_pyxrf = json.load(beamline_pyxrf)
                beamline_name = beamline_config_pyxrf["beamline_name"]
        except Exception as ex:
            raise IOError(f"Error while opening configuration file {config_path!r}") from ex

    else:
        # Otherwise try to identify the beamline using host name
        hostname = platform.node()
        beamline_names = {
            "xf03id": "HXN",
            "xf05id": "SRX",
            "xf08bm": "TES",
            "xf04bm": "XFM",
        }

        for k, v in beamline_names.items():
            if hostname.startswith(k):
                beamline_name = v

    if beamline_name is None:
        raise Exception("Beamline is not identified")

    if beamline_name == "HXN":
        from pyxrf.db_config.hxn_db_config import db
    elif beamline_name == "SRX":
        from pyxrf.db_config.srx_db_config import db
    elif beamline_name == "XFM":
        from pyxrf.db_config.xfm_db_config import db
    elif beamline_name == "TES":
        from pyxrf.db_config.tes_db_config import db
    else:
        db = None
        db_analysis = None
        print(f"Beamline Database is not used in pyxrf: unknown beamline {beamline_name!r}")

except Exception as ex:
    db = None
    print(f"Beamline Database is not used in pyxrf: {ex}")


def flip_data(input_data, subscan_dims=None):
    """
    Flip 2D or 3D array. The flip happens on the second index of shape.
    .. warning :: This function mutates the input values.

    Parameters
    ----------
    input_data : 2D or 3D array.

    Returns
    -------
    flipped data
    """
    new_data = np.asarray(input_data)
    data_shape = input_data.shape
    if len(data_shape) == 2:
        if subscan_dims is None:
            new_data[1::2, :] = new_data[1::2, ::-1]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :] = new_data[start:end:2, ::-1]
                i += ny

    if len(data_shape) == 3:
        if subscan_dims is None:
            new_data[1::2, :, :] = new_data[1::2, ::-1, :]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :, :] = new_data[start:end:2, ::-1, :]
                i += ny
    return new_data


def fetch_run_info(run_id_uid):
    """
    Fetches key data from start document of the selected run

    Parameters
    ----------
    run_id_uid: int or str
        Run ID (positive or negative int) or UID (str, full or short) of the run.

    Returns
    -------
    int or str
        Run ID (always positive int) or Run UID (str, always full UID). Returns
        `run_id=-1` and `run_uid=""` in case of failure.

    Raises
    ------
    RuntimeError
        failed to fetch the run from Databroker
    """
    try:
        hdr = db[run_id_uid]
        run_id = hdr.start["scan_id"]
        run_uid = hdr.start["uid"]
    except Exception:
        if isinstance(run_id_uid, int):
            msg = f"ID {run_id_uid}"
        else:
            msg = f"UID '{run_id_uid}'"
        raise RuntimeError(f"Failed to find run with {msg}.")
    return run_id, run_uid


def fetch_data_from_db(
    run_id_uid,
    fpath=None,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=False,
    save_scaler=True,
    num_end_lines_excluded=None,
):
    """
    Read data from databroker.
    This is the place where new beamlines can be easily added
    to pyxrf GUI.
    Save the data from databroker to hdf file if needed.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        id number for given run
    fpath: str, optional
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered, an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        will be raised (loading the single scan) or the scan will be skipped (loading the range
        of scans).
    output_to_file : bool, optional
        save data to hdf5 file if True
    save_scaler : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[-1]
    print("Loading data from database.")

    if hdr.start.beamline_id == "HXN":
        data = map_data2D_hxn(
            run_id_uid,
            fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
        )
    elif hdr.start.beamline_id == "xf05id" or hdr.start.beamline_id == "SRX":
        data = map_data2D_srx(
            run_id_uid,
            fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
            save_scaler=save_scaler,
            num_end_lines_excluded=num_end_lines_excluded,
        )
    elif hdr.start.beamline_id == "XFM":
        data = map_data2D_xfm(
            run_id_uid,
            fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
        )
    elif hdr.start.beamline_id == "TES":
        data = map_data2D_tes(
            run_id_uid,
            fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
        )
    else:
        print("Databroker is not setup for this beamline")
        return
    free_memory_from_handler()
    return data


def make_hdf(
    start,
    end=None,
    *,
    fname=None,
    wd=None,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    prefix="scan2D_",
    create_each_det=False,
    save_scaler=True,
    num_end_lines_excluded=None,
):
    """
    Load data from database and save it in HDF5 files.

    Parameters
    ----------

    start : int
        Run ID (positive or negative int) or  of the first scan to convert or Run UID
        (str, full or short). If `start` is UID, then `end` must not be provided or set to None.
    end : int, optional
        scan ID of the last scan to convert. If ``end`` is not specified or None, then
        only the scan with ID ``start`` is converted and an exception is raised if an
        error occurs during the conversion. If ``end`` is specified, then scans in the
        range ``scan``..``end`` are converted and a scan in the sequence is skipped
        if there is an issue during the conversion. For example:

        .. code-block:: python

            make_hdf(2342)

        will process scan #2342 and throw an exception if error occurs. On the other hand

        .. code-block:: python

            make_hdf(2342, 2342)

        will process scan #2342 and write data to file if conversion is successful, otherwise
        no file will be created. The scans with IDs in the range 2342..2441 can be processed by
        calling

        .. code-block:: python

            make_hdf(2342, 2441)

        Scans with IDs in specified range, but not existing in the database, or scans causing errors
        during conversion will be skipped.

    fname : string, optional keyword parameter
        path to save data file when ``end`` is ``None`` (only one scan is processed).
        File name is created automatically if ``fname`` is not specified.
    wd : str
        working directory, the file(s) will be created in this directory. The directory
        will be created if it does not exist. If ``wd`` is not specified, then the file(s)
        will be saved to the current directory.
    fname_add_version : bool, keyword parameter
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails. If ``end`` is ``None``, then
        the exception is raised. If ``end`` is specified, the scan is skipped
        and the next scan in the range is processed.
    completed_scans_only : bool, keyword parameter
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown (``end`` is not specified) or the scan
        is skipped (``end`` is specified). This feature allows to use
        ``make_hdf`` as part of the script for real time data analysis:

        .. code-block:: python

            # Wait time between retires in seconds. Select the value appropriate
            #   for the workflow type.
            wait_time = 600  # Wait for 10 minuts between retries.
            for scan_id in range(n_start, n_start + n_scans):
                while True:
                    try:
                        # Load scan if it is available
                        make_hdf(scan_id, completed_scans_only=True)
                        # Process the file using the prepared parameter file
                        pyxrf_batch(scan_id, param_file_name="some_parameter_file.json")
                        break
                    except Exception:
                        time.sleep(wait_time)

        Such scripts are currently used at HXN and SRX beamlines of NSLS-II, so this feature
        supports the existing workflows.
        False: the feature is disabled, incomplete scan will be processed.
    successful_scans_only : bool, keyword parameter
        Similar to ``complete_scans_only``. The file is created only if the stop document
        exists and ``exit_status=='success'``.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        will be raised (loading the single scan) or the scan will be skipped (loading the range
        of scans).
    prefix : str, optional
        prefix name of the created data file. If ``fname`` is not specified, it is generated
        automatically in the form ``<prefix>_<scanID>_<some_additional_data>.h5``
    create_each_det: bool, optional
        True: save data for each available detector channel into a file. Enabling this
        feature leads to larger data files. Inspection of data from individual channels
        of the detector may be helpful in evaluation of quality of the detector calibration
        and adds flexibility to data analysis. This feature may be disabled if large number
        of routine scans recorded by well tested system are processed and disk space
        is an issue.
        False: disable the feature. Only the sum of all detector channels is saved
        to disk.
    save_scaler : bool, optional
        True: save scaler data in the data file
        False: do not save scaler data
    num_end_lines_excluded : int, optional
        The number of lines at the end of the scan that will not be saved to the data file.
    """

    if wd:
        # Create the directory
        wd = os.path.expanduser(wd)
        wd = os.path.abspath(wd)  # 'make_dirs' does not accept paths that contain '..'
        os.makedirs(wd, exist_ok=True)  # Does nothing if the directory already exists

    if isinstance(start, str) or (end is None):
        # Two cases: only one Run ID ('start') is provided or 'start' is Run UID.
        #   In both cases only one run is loaded.
        if end is not None:
            raise ValueError(r"Parameter 'end' must be None if run is loaded by UID")

        run_id, run_uid = fetch_run_info(start)  # This may raise RuntimeException

        # Load one scan with ID specified by ``start``
        #   If there is a problem while reading the scan, the exception is raised.
        if fname is None:
            fname = prefix + str(run_id) + ".h5"
            if wd:
                fname = os.path.join(wd, fname)
        fetch_data_from_db(
            run_uid,
            fpath=fname,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=True,
            save_scaler=save_scaler,
            num_end_lines_excluded=num_end_lines_excluded,
        )
    else:
        # Both ``start`` and ``end`` are specified. Convert the scans in the range
        #   ``start`` .. ``end``. If there is a problem reading the scan,
        #   then the scan is skipped and the next scan is processed
        datalist = range(start, end + 1)
        for v in datalist:
            fname = prefix + str(v) + ".h5"
            if wd:
                fname = os.path.join(wd, fname)
            try:
                fetch_data_from_db(
                    v,
                    fpath=fname,
                    create_each_det=create_each_det,
                    fname_add_version=fname_add_version,
                    completed_scans_only=completed_scans_only,
                    successful_scans_only=successful_scans_only,
                    file_overwrite_existing=file_overwrite_existing,
                    output_to_file=True,
                    save_scaler=save_scaler,
                    num_end_lines_excluded=num_end_lines_excluded,
                )
                print(f"Scan #{v}: Conversion completed.\n")
            except Exception as ex:
                print(f"Scan #{v}: Can not complete the conversion")
                print(f"    ({ex})\n")


def _is_scan_complete(hdr):
    """Checks if the scan is complete ('stop' document exists)

    Parameters
    ----------

    hdr : databroker.core.Header
        header of the run
        hdr = db[scan_id]
        The header must be reloaded each time before the function is called.

    Returns
    -------

    True: scan is complete
    False: scan is incomplete (still running)
    """

    # hdr.stop is an empty dictionary if the scan is incomplete
    return bool(hdr.stop)


def _is_scan_successful(hdr):
    """
    Checks if the scan is successful
    """
    return bool(hdr.stop) and hdr.stop["exit_status"] == "success"


def _extract_metadata_from_header(hdr):
    """
    Extract metadata from start and stop document. Metadata extracted from other document
    in the scan are beamline specific and added to dictionary at later time.
    """
    start_document = hdr.start

    mdata = ScanMetadataXRF()

    data_locations = {
        "scan_id": ["scan_id"],
        "scan_uid": ["uid"],
        "scan_instrument_id": ["beamline_id"],
        "scan_instrument_name": [],
        "scan_end_station": [],
        "scan_time_start": ["time"],
        "scan_time_start_utc": ["time"],
        "instrument_mono_incident_energy": ["beamline_status/energy", "scan/energy"],
        "instrument_beam_current": [],
        "instrument_detectors": ["detectors", "scan/detectors"],
        "sample_name": ["sample/name", "sample", "scan/sample_name"],
        "experiment_plan_name": ["plan_name"],
        "experiment_plan_type": ["plan_type"],
        "proposal_num": ["proposal/proposal_num"],
        "proposal_title": ["proposal/proposal_title"],
        "proposal_PI_lastname": ["proposal/PI_lastname"],
        "proposal_saf_num": ["proposal/saf_num"],
        "proposal_cycle": ["proposal/cycle"],
        # Scan parameters
        "param_type": ["scan/type"],
        "param_input": ["scan/scan_input"],
        "param_dwell": ["scan/dwell", "exposure_time"],
        "param_snake": ["scan/snake"],
        "param_shape": ["scan/shape", "shape"],
        "param_theta": ["scan/theta/val"],
        "param_theta_units": ["scan/theta/units"],
        "param_delta": ["scan/delta/val"],
        "param_delta_units": ["scan/delta/units"],
        "param_fast_axis": ["scaninfo/fast_axis", "scan/fast_axis/motor_name"],
        "param_fast_axis_units": ["scan/fast_axis/units"],
        "param_slow_axis": ["scaninfo/slow_axis", "scan/slow_axis/motor_name"],
        "param_slow_axis_units": ["scan/slow_axis/units"],
    }

    for key, locations in data_locations.items():
        # Go to the next key if no location is defined for the current key.
        #   No locations means that the data is not yet defined in start document on any beamline
        #   Multiple locations point to locations at different beamlines
        if not locations:
            continue

        # For each metadata key there could be none, one or multiple locations in the start document
        for loc in locations:
            path = loc.split("/")  #
            ref = start_document
            for n, p in enumerate(path):
                if n >= len(path) - 1:
                    break
                # 'ref' must always point to dictionary
                if not isinstance(ref, dict):
                    ref = None
                    break
                if p in ref:
                    ref = ref[p]
                else:
                    ref = None
                    break
            # At this point 'ref' must be a dictionary
            value = None
            if ref is not None and isinstance(ref, dict):
                if path[-1] in ref:
                    value = ref[path[-1]]
            # Now we finally arrived to the end of the path: the 'value' must be a scalar or a list
            if value is not None and not isinstance(value, dict):
                if path[-1] == "time":
                    if key.endswith("_utc"):
                        value = convert_time_to_nexus_string(ttime.gmtime(value))
                    else:
                        value = convert_time_to_nexus_string(ttime.localtime(value))
                mdata[key] = value
                break

    stop_document = hdr.stop

    if stop_document:

        if "time" in stop_document:
            t = stop_document["time"]
            mdata["scan_time_stop"] = convert_time_to_nexus_string(ttime.localtime(t))
            mdata["scan_time_stop_utc"] = convert_time_to_nexus_string(ttime.gmtime(t))

        if "exit_status" in stop_document:
            mdata["scan_exit_status"] = stop_document["exit_status"]

    else:

        mdata["scan_exit_status"] = "incomplete"

    # Add full beamline name (if available, otherwise don't create the entry).
    #   Also, don't overwrite the existing name if it was read from the start document
    if "scan_instrument_id" in mdata and "scan_instrument_name" not in mdata:
        instruments = {
            "srx": "Submicron Resolution X-ray Spectroscopy",
            "hxn": "Hard X-ray Nanoprobe",
            "tes": "Tender Energy X-ray Absorption Spectroscopy",
            "xfm": "X-ray Fluorescence Microprobe",
        }
        iname = instruments.get(mdata["scan_instrument_id"].lower(), "")
        if iname:
            mdata["scan_instrument_name"] = iname

    return mdata


def _get_metadata_value_from_descriptor_document(hdr, *, data_key, stream_name="baseline"):
    """
    Returns the first occurrence of the variable with the name ``data_key`` in
    specified document stream. Returns ``None`` if the variable is not found
    """
    value = None
    docs = hdr.documents(stream_name=stream_name)
    for name, doc in docs:
        if (name != "event") or ("descriptor" not in doc):
            continue
        try:
            value = doc["data"][data_key]
            break  # Don't go through the rest of the documents
        except Exception:
            pass

    return value


def _get_metadata_all_from_descriptor_document(hdr, *, data_key, stream_name="baseline"):
    """
    Returns the list of the recorded values of variables with the name ``data_key`` in
    specified document stream. Returns ``None`` if the variable is not found
    """
    value = []
    docs = hdr.documents(stream_name=stream_name)
    for name, doc in docs:
        if (name != "event") or ("descriptor" not in doc):
            continue
        try:
            value.append(doc["data"][data_key])
        except Exception:
            pass

    value = value or None  # Replace [] with None

    return value


def map_data2D_hxn(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
):
    """
    Save the data from databroker to hdf file.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    run_id_uid : int
        ID or UID of a run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        is raised.
    output_to_file : bool, optional
        save data to hdf5 file if True
    """
    hdr = db[run_id_uid]
    runid = hdr.start["scan_id"]  # Replace with the true value (runid may be relative, such as -2)

    logger.info(f"Loading scan #{runid}")
    if completed_scans_only and not _is_scan_complete(hdr):
        raise Exception("Scan is incomplete. Only completed scans are currently processed.")
    if successful_scans_only and not _is_scan_successful(hdr):
        raise Exception(
            "Scan is not successfully completed. Only successfully completed scans are currently processed."
        )

    # Generate the default file name for the scan
    if fpath is None:
        fpath = f"scan2D_{runid}.h5"

    # Output data is the list of data structures for all available detectors
    data_output = []

    start_doc = hdr["start"]

    # Exclude certain types of plans based on data from the start document
    if start_doc["plan_type"] in ("FlyPlan1D",):
        raise RuntimeError(f"Failed to load the plan: plan {start_doc['plan_type']!r} is not supported")

    # The dictionary holding scan metadata
    mdata = _extract_metadata_from_header(hdr)
    # Some metadata is located at specific places in the descriptor documents
    # Search through the descriptor documents for the metadata
    v = _get_metadata_value_from_descriptor_document(
        hdr, data_key="beamline_status_beam_current", stream_name="baseline"
    )
    if v is not None:
        mdata["instrument_beam_current"] = v

    v = _get_metadata_value_from_descriptor_document(hdr, data_key="energy", stream_name="baseline")
    if v is not None:
        mdata["instrument_mono_incident_energy"] = v

    # --------------------------------------------------------------------------------------------
    #            IDENTIFY END STATION AND SELECT THE APPROPRIATE THETA ANGLE FROM BASELINE
    # Identify endstation
    end_station = ""
    es_motors = hdr.start["motors"]
    motors_mll, motors_zp = ("dssx", "dssy", "dssz"), ("zpssx", "zpssy", "zpssz")
    if es_motors[0] in motors_mll:
        end_station = "MLL"
    elif es_motors[0] in motors_zp:
        end_station = "ZP"
    else:
        logger.warning("Failed to identify end station from data found in start document.")
    if end_station:
        mdata["scan_end_station"] = end_station

    logger.info(f"Identified beamline end station: {end_station!r}")

    # Get theta angles (each scan has the angles for both endstations, but we need to pick one)
    v = _get_metadata_value_from_descriptor_document(
        hdr, data_key="beamline_status_beam_current", stream_name="baseline"
    )
    if end_station == "MLL":
        theta = _get_metadata_value_from_descriptor_document(hdr, data_key="dsth", stream_name="baseline")  # MLL
    elif end_station == "ZP":
        theta = _get_metadata_value_from_descriptor_document(hdr, data_key="zpsth", stream_name="baseline")  # ZP
    else:
        theta = None
    # Add theta to the the metadata
    if theta is not None:
        mdata["param_theta"] = round(theta * 1000)  # Convert to mdeg (same as SRX)
        mdata["param_theta_units"] = "mdeg"
        theta = round(theta, 3)  # Better presentation
    else:
        logger.warning("Angle 'theta' is not found and is not included in the HDF file metadata")
    # -----------------------------------------------------------------------------------------------
    # Determine fast axis and slow axis
    fast_axis, slow_axis = start_doc.get("fast_axis", None), None
    motors = start_doc.get("motors", None)
    if motors and isinstance(motors, (list, tuple)) and len(motors) == 2:
        fast_axis = fast_axis if fast_axis else motors[0]
        fast_axis_index = motors.index(fast_axis, 0)
        slow_axis_index = 0 if (fast_axis_index == 1) else 1
        slow_axis = motors[slow_axis_index]
    if fast_axis:
        mdata["param_fast_axis"] = fast_axis
    if slow_axis:
        mdata["param_slow_axis"] = slow_axis
    # -----------------------------------------------------------------------------------------------
    # Reconstruct scan input
    try:
        plan_args = start_doc["plan_args"]
        # px_motor = plan_args["motor1"]
        px_start, px_end, px_step = plan_args["scan_start1"], plan_args["scan_end1"], plan_args["num1"]
        # py_motor = plan_args["motor2"]
        py_start, py_end, py_step = plan_args["scan_start2"], plan_args["scan_end2"], plan_args["num2"]
        dwell_time = plan_args["exposure_time"]
        param_input = [px_start, px_end, px_step, py_start, py_end, py_step, dwell_time]
        mdata["param_input"] = param_input
    except Exception as ex:
        logger.warning(
            "Failed to reconstruct scan input: %s. Scan input is not saved as part of metadata to HDF5 file",
            str(ex),
        )
    # -------------------------------------------------------------------------------------------------

    if "dimensions" in start_doc:
        datashape = start_doc.dimensions
    elif "shape" in start_doc:
        datashape = start_doc.shape
    else:
        logger.error("No dimension/shape is defined in hdr.start.")

    datashape = [datashape[1], datashape[0]]  # vertical first, then horizontal
    fly_type = start_doc.get("fly_type", None)
    subscan_dims = start_doc.get("subscan_dims", None)

    if "motors" in hdr.start:
        pos_list = hdr.start.motors
    elif "axes" in hdr.start:
        pos_list = hdr.start.axes
    else:
        pos_list = ["zpssx[um]", "zpssy[um]"]

    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = "hxn_pv_config.json"
    config_path = sep_v.join(current_dir.split(sep_v)[:-2] + ["configs", config_file])
    with open(config_path, "r") as json_data:
        config_data = json.load(json_data)

    keylist = hdr.descriptors[0].data_keys.keys()
    det_list = [v for v in keylist if "xspress3" in v]  # find xspress3 det with key word matching

    scaler_list_all = config_data["scaler_list"]

    all_keys = hdr.descriptors[0].data_keys.keys()
    scaler_list = [v for v in scaler_list_all if v in all_keys]

    fields = det_list + scaler_list + pos_list
    data = hdr.table(fields=fields, fill=True)

    data_out = map_data2D(
        data,
        datashape,
        det_list=det_list,
        pos_list=pos_list,
        scaler_list=scaler_list,
        create_each_det=create_each_det,
        fly_type=fly_type,
        subscan_dims=subscan_dims,
        spectrum_len=4096,
    )

    # Transform coordinates for the fast axis if necessary:
    #   Flip the direction of the fast axis for certain angles
    if (theta is not None) and fast_axis.lower().endswith("z") and (theta < 0):
        logger.info(f"Fast axis: {fast_axis!r}. Angle 'theta': {theta}. Flipping data along the fast axis ...")
        data_out["pos_data"][fast_axis_index, :, :] = np.fliplr(data_out["pos_data"][fast_axis_index, :, :])
        data_out["scaler_data"] = np.flip(data_out["scaler_data"], axis=1)
        data_out["det_sum"] = np.flip(data_out["det_sum"], axis=1)
        for k in data.keys():
            if re.search(r"^det[\d]+$", k):  # Individual detectors such as 'det1', 'det2', etc.
                data_out[k] = np.flip(data_out[k], axis=1)
    else:
        logger.info(
            f"Fast axis: {fast_axis!r}. Angle 'theta': {theta}. Data along the fast axis is not reordered."
        )

    #   Correct positions for distortions due to rotation of the stage
    if theta is not None:
        if fast_axis.lower().endswith("x"):
            logger.info(f"Scaling the positions along fast X-axis ({fast_axis!r}: 'theta'={theta}) ...")
            data_out["pos_data"][fast_axis_index, :, :] *= np.cos(theta * np.pi / 180.0)
        elif fast_axis.lower().endswith("z"):
            logger.info(f"Scaling the positions along fast Z-axis ({fast_axis!r}: 'theta'={theta}) ...")
            data_out["pos_data"][fast_axis_index, :, :] *= np.sin(theta * np.pi / 180.0)
        else:
            logger.info(f"No scaling is applied to the positions along fast axis ({fast_axis!r})")

    if output_to_file:
        # output to file
        print("Saving data to hdf file.")
        fpath = save_data_to_hdf5(
            fpath,
            data_out,
            metadata=mdata,
            fname_add_version=fname_add_version,
            file_overwrite_existing=file_overwrite_existing,
            create_each_det=create_each_det,
        )

    detector_name = "xpress3"
    d_dict = {"dataset": data_out, "file_name": fpath, "detector_name": detector_name, "metadata": mdata}
    data_output.append(d_dict)

    return data_output

    # write_db_to_hdf(fpath, data, datashape,
    #                 det_list=det_list, pos_list=pos_list,
    #                 scaler_list=scaler_list,
    #                 fly_type=fly_type, subscan_dims=subscan_dims)
    #
    # # use suitcase to save baseline data, and scaler data from primary
    # tmp = set()
    # for descriptor in hdr.descriptors:
    #     # no 3D vector data
    #     xs3 = [key for key in descriptor.data_keys.keys() if 'xspress3' in key]
    #     tmp.update(xs3)
    #     tmp.add('merlin1')
    # fds = sc.filter_fields(hdr, tmp)
    # if full_data == True:
    #     sc.export(hdr, fpath, db.mds, fields=fds, use_uid=False)


def get_total_scan_point(hdr):
    """
    Find the how many data points are recorded. This number may not equal to the total number
    defined at the start of the scan due to scan stop or abort.
    """
    evs = hdr.events()
    n = 0
    try:
        for e in evs:
            n = n + 1
    except IndexError:
        pass
    return n


def map_data2D_srx(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
    save_scaler=True,
    num_end_lines_excluded=None,
):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at SRX beamline for both fly scan and step scan.
    Save to hdf file if needed.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    run_id_uid : int
        ID or UID of a run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        is raised.
    output_to_file : bool, optional
        save data to hdf5 file if True
    save_scaler : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[run_id_uid]
    start_doc = hdr["start"]
    use_new_format = "md_version" in start_doc

    if use_new_format:
        return map_data2D_srx_new(
            run_id_uid=run_id_uid,
            fpath=fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
            save_scaler=save_scaler,
            num_end_lines_excluded=num_end_lines_excluded,
        )
    else:
        return map_data2D_srx_old(
            run_id_uid=run_id_uid,
            fpath=fpath,
            create_each_det=create_each_det,
            fname_add_version=fname_add_version,
            completed_scans_only=completed_scans_only,
            successful_scans_only=successful_scans_only,
            file_overwrite_existing=file_overwrite_existing,
            output_to_file=output_to_file,
            save_scaler=save_scaler,
            num_end_lines_excluded=num_end_lines_excluded,
        )


def map_data2D_srx_old(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
    save_scaler=True,
    num_end_lines_excluded=None,
):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at SRX beamline for both fly scan and step scan.
    Save to hdf file if needed.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    run_id_uid : int
        ID or UID of a run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        is raised.
    output_to_file : bool, optional
        save data to hdf5 file if True
    save_scaler : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[run_id_uid]
    runid = hdr.start["scan_id"]  # Replace with the true value (runid may be relative, such as -2)

    print("Scan metadata format: old SRX specification.")

    if completed_scans_only and not _is_scan_complete(hdr):
        raise Exception("Scan is incomplete. Only completed scans are currently processed.")
    if successful_scans_only and not _is_scan_successful(hdr):
        raise Exception(
            "Scan is not successfully completed. Only successfully completed scans are currently processed."
        )

    spectrum_len = 4096
    start_doc = hdr["start"]

    # The dictionary holding scan metadata
    mdata = _extract_metadata_from_header(hdr)
    plan_n = start_doc.get("plan_name")

    # Load configuration file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = "srx_pv_config.json"
    config_path = sep_v.join(current_dir.split(sep_v)[:-2] + ["configs", config_file])
    with open(config_path, "r") as json_data:
        config_data = json.load(json_data)

    # Generate the default file name for the scan
    if fpath is None:
        fpath = f"scan2D_{runid}.h5"

    # Output data is the list of data structures for all available detectors
    data_output = []

    # There may be no 'plan_name' key in the old stepscans
    if (plan_n is None) or ("fly" not in plan_n):  # not fly scan

        print()
        print("****************************************")
        print("        Loading SRX step scan           ")
        print("****************************************")

        # Examples for testing on SRX beamline:
        #    good 'old-style' step scan ID: 2357 UID: e063146b-103a-40c5-9266-2201f157e950
        #    good 'new-style' step scan ID: 18015 UID: 6ae30aa1-5834-4641-8e68-5eaad4669ce0

        fly_type = None

        if num_end_lines_excluded is None:
            # It seems like the 'shape' in plan is in the form of [y, x], where
            #    y - is the vertical and x is horizontal axis. This matches the
            #    shape of the matrix that is used for storage of the maps.
            #    In step scan, the results are represented as 1D array, not 2D array,
            #    so it needs to be reshaped before processing. So the datashape
            #    needs to be determined correctly.
            # We also assume that scanning is performed along the x-axis first
            #    before stepping along y-axis. Snaking may be on or off.
            #    Different order (along y-axis first, then along x-axis) will require
            #    some additional parameter in the start document to indicate this.
            #    And the 'datashape' will need to be set the opposite way. Also
            #    the map representation will be transposed.
            datashape = [start_doc["shape"][0], start_doc["shape"][1]]
        else:
            datashape = [start_doc["shape"][0] - num_end_lines_excluded, start_doc["shape"][1]]

        snake_scan = start_doc.get("snaking")
        if snake_scan[1] is True:
            fly_type = "pyramid"

        if hdr.start.get("plan_type") == "OuterProductAbsScanPlan":
            # This is 'old-style' step scan
            detector_list = ["xs_settings_ch1", "xs_settings_ch2", "xs_settings_ch3"]
            scaler_list = ["current_preamp_ch2"]
        else:
            # This is 'new-style' step scan
            detector_list = config_data["xrf_detector"]
            scaler_list = config_data["scaler_list"]

        try:
            data = hdr.table(fill=True, convert_times=False)

        except IndexError:
            total_len = get_total_scan_point(hdr) - 2
            evs, _ = zip(*zip(hdr.events(fill=True), range(total_len)))
            namelist = detector_list + hdr.start.motors + scaler_list
            dictv = {v: [] for v in namelist}
            for e in evs:
                for k, v in dictv.items():
                    dictv[k].append(e.data[k])
            data = pd.DataFrame(dictv, index=np.arange(1, total_len + 1))  # need to start with 1

        #  Commented by DG: Just use the detector names from .json configuration file. Do not delete the code.
        # express3 detector name changes in databroker
        # if xrf_detector_names[0] not in data.keys():
        #     xrf_detector_names = ['xs_channel'+str(i) for i in range(1,4)]
        #     config_data['xrf_detector'] = xrf_detector_names

        if output_to_file:
            if "xs" in hdr.start.detectors:
                logger.info("Saving data to hdf file: Xpress3 detector #1 (three channels).")
                root, ext = os.path.splitext(fpath)
                fpath_out = f"{root + '_xs'}{ext}"
                data_out = assemble_data_SRX_stepscan(
                    data,
                    datashape,
                    det_list=detector_list,
                    pos_list=hdr.start.motors,
                    scaler_list=scaler_list,
                    fname_add_version=fname_add_version,
                    create_each_det=create_each_det,
                    fly_type=fly_type,
                    base_val=config_data["base_value"],
                )  # base value shift for ic
                fpath_out = save_data_to_hdf5(
                    fpath_out,
                    data_out,
                    metadata=mdata,
                    fname_add_version=fname_add_version,
                    file_overwrite_existing=file_overwrite_existing,
                    create_each_det=create_each_det,
                )
                d_dict = {"dataset": data_out, "file_name": fpath_out, "detector_name": "xs", "metadata": mdata}
                data_output.append(d_dict)

            if "xs2" in hdr.start.detectors:
                logger.info("Saving data to hdf file: Xpress3 detector #2 (single channel).")
                root, ext = os.path.splitext(fpath)
                fpath_out = f"{root}_xs2{ext}"
                data_out = assemble_data_SRX_stepscan(
                    data,
                    datashape,
                    # The following must be XS2 detectors (not present in 'old' step scans)
                    det_list=config_data["xrf_detector2"],
                    pos_list=hdr.start.motors,
                    scaler_list=scaler_list,
                    fname_add_version=fname_add_version,
                    create_each_det=create_each_det,
                    fly_type=fly_type,
                    base_val=config_data["base_value"],
                )  # base value shift for ic
                fpath_out = save_data_to_hdf5(
                    fpath_out,
                    data_out,
                    metadata=mdata,
                    fname_add_version=fname_add_version,
                    file_overwrite_existing=file_overwrite_existing,
                    create_each_det=create_each_det,
                )
                d_dict = {"dataset": data_out, "file_name": fpath_out, "detector_name": "xs", "metadata": mdata}
                data_output.append(d_dict)

            fln_list = [_["file_name"] for _ in data_output]
            logger.debug(f"Step scan data was saved to the following files: {fln_list}")

        return data_output

    else:

        print()
        print("****************************************")
        print("         Loading SRX fly scan           ")
        print("****************************************")

        if save_scaler is True:
            scaler_list = ["i0", "time", "i0_time", "time_diff"]
            xpos_name = "enc1"
            ypos_name = "hf_stage_y"  # 'hf_stage_x' if fast axis is vertical

        # The dictionary of fields that are used to store data from different detectors (for fly scan only)
        #   key - the name of the field used to store data read from the detector
        #   value - the detector name (probably short abbreviation, attached to the created file name so that
        #           the detector could be identified)
        # A separate data file is created for each detector
        detector_field_dict = config_data["xrf_flyscan_detector_fields"]

        num_det = 0  # Some default value (should never be used)

        #  Added by AMK to allow flying of single element on xs2
        # if 'E_tomo' in start_doc['scaninfo']['type']:
        #     num_det = 1
        #     ypos_name = 'e_tomo_y'
        # else:
        #     num_det = 3
        vertical_fast = False  # assuming fast on x as default
        if num_end_lines_excluded is None:
            # vertical first then horizontal, assuming fast scan on x
            datashape = [start_doc["shape"][1], start_doc["shape"][0]]
        else:
            datashape = [start_doc["shape"][1] - num_end_lines_excluded, start_doc["shape"][0]]

        using_nanostage = "nanoZebra" in hdr.start.detectors

        if using_nanostage:
            # There should also be a source of 'z' positions
            xpos_name, ypos_name = "enc1", "enc2"
            # Note: the following block doesn't make sence for the setup with nanostage
            # The following condition will be more complicated when 'slow_axis' is
            #   added to the metadata.
            # if hdr.start.scaninfo['fast_axis'] == "NANOVER":
            #    xpos_name, ypos_name = ypos_name, xpos_name
            #    vertical_fast = True
        else:
            if "fast_axis" in hdr.start.scaninfo:
                # fast scan along vertical, y is fast scan, x is slow
                if hdr.start.scaninfo["fast_axis"] in ("VER", "DET2VER"):
                    xpos_name = "enc1"
                    ypos_name = "hf_stage_x"
                    if "E_tomo" in start_doc["scaninfo"]["type"]:
                        ypos_name = "e_tomo_x"
                    vertical_fast = True
                    #   fast vertical scan put shape[0] as vertical direction
                    # datashape = [start_doc['shape'][0], start_doc['shape'][1]]

        new_shape = datashape + [spectrum_len]
        # total_points = datashape[0]*datashape[1]

        des = [d for d in hdr.descriptors if d.name == "stream0"][0]
        #   merlin data doesn't need to be saved.
        # un_used_det = ['merlin', 'im'] # data not to be transfered for pyxrf
        # data_list_used = [v for v in des.data_keys.keys() if 'merlin' not in v.lower()]

        # The number of found detectors for which data exists in the database
        n_detectors_found = 0

        # Try each data field listed in the config file
        for detector_field, detector_name in detector_field_dict.items():

            # Assume that Databroker caches the tables locally, so that data will not be reloaded
            e = hdr.events(fill=True, stream_name=des.name)

            new_data = {}
            data = {}

            if save_scaler is True:
                new_data["scaler_names"] = scaler_list
                scaler_tmp = np.zeros([datashape[0], datashape[1], len(scaler_list)])
                if vertical_fast is True:  # data shape only has impact on scaler data
                    scaler_tmp = np.zeros([datashape[1], datashape[0], len(scaler_list)])
                key_list = scaler_list + [xpos_name]
                if using_nanostage:
                    key_list += [ypos_name]
                for v in key_list:
                    data[v] = np.zeros([datashape[0], datashape[1]])

            # Total number of lines in fly scan
            n_scan_lines_total = new_shape[0]

            detector_field_exists = True

            # This 'try' block was added in response to the request to retrieve data after
            #   detector failure (empty files were saved by Xpress3). The program is supposed
            #   to retrieve 'good' data from the scan.
            try:
                for m, v in enumerate(e):
                    if m == 0:

                        # Check if detector field does not exist. If not, then the file should not be created.
                        if detector_field not in v.data:
                            detector_field_exists = False
                            break

                        print()
                        print(f"Collecting data from detector '{detector_name}' (field '{detector_field}')")

                        # Determine the number of channels from the size of the table with fluorescence data
                        num_det = v.data[detector_field].shape[1]

                        # Now allocate space for fluorescence data
                        if create_each_det is False:
                            new_data["det_sum"] = np.zeros(new_shape, dtype=np.float32)
                        else:
                            for i in range(num_det):
                                new_data[f"det{i + 1}"] = np.zeros(new_shape, dtype=np.float32)

                        print(f"Number of the detector channels: {num_det}")

                    if m < datashape[0]:  # scan is not finished
                        if save_scaler is True:
                            for n in scaler_list[:-1] + [xpos_name]:
                                min_len = min(v.data[n].size, datashape[1])
                                data[n][m, :min_len] = v.data[n][:min_len]
                                # position data or i0 has shorter length than fluor data
                                if min_len < datashape[1]:
                                    len_diff = datashape[1] - min_len
                                    # interpolation on scaler data
                                    interp_list = (v.data[n][-1] - v.data[n][-3]) / 2 * np.arange(
                                        1, len_diff + 1
                                    ) + v.data[n][-1]
                                    data[n][m, min_len : datashape[1]] = interp_list
                        fluor_len = v.data[detector_field].shape[0]
                        if m > 0 and not (m % 10):
                            print(f"Processed {m} of {n_scan_lines_total} lines ...")
                        # print(f"m = {m} Data shape {v.data['fluor'].shape} - {v.data['fluor'].shape[1] }")
                        # print(f"Data keys: {v.data.keys()}")
                        if create_each_det is False:
                            for i in range(num_det):
                                # in case the data length in each line is different
                                new_data["det_sum"][m, :fluor_len, :] += v.data[detector_field][:, i, :]
                        else:
                            for i in range(num_det):
                                # in case the data length in each line is different
                                new_data["det" + str(i + 1)][m, :fluor_len, :] = v.data[detector_field][:, i, :]

            except Exception as ex:
                logger.error(f"Error occurred while reading data: {ex}. Trying to retrieve available data ...")

            # If the detector field does not exist, then try the next one from the list
            if not detector_field_exists:
                continue

            # Modify file name (path) to include data on how many channels are included in the file and how many
            #    channels are used for sum calculation
            root, ext = os.path.splitext(fpath)
            s = f"_{detector_name}_sum{num_det}ch"
            if create_each_det:
                s += f"+{num_det}ch"
            fpath_out = f"{root}{s}{ext}"

            if vertical_fast is True:  # need to transpose the data, as we scan y first
                if create_each_det is False:
                    new_data["det_sum"] = np.transpose(new_data["det_sum"], axes=(1, 0, 2))
                else:
                    for i in range(num_det):
                        new_data["det" + str(i + 1)] = np.transpose(new_data["det" + str(i + 1)], axes=(1, 0, 2))

            if save_scaler is True:
                if vertical_fast is False:
                    for i, v in enumerate(scaler_list[:-1]):
                        scaler_tmp[:, :, i] = data[v]
                    scaler_tmp[:, :-1, -1] = np.diff(data["time"], axis=1)
                    scaler_tmp[:, -1, -1] = data["time"][:, -1] - data["time"][:, -2]
                else:
                    for i, v in enumerate(scaler_list[:-1]):
                        scaler_tmp[:, :, i] = data[v].T
                    data_t = data["time"].T
                    scaler_tmp[:-1, :, -1] = np.diff(data_t, axis=0)
                    scaler_tmp[-1, :, -1] = data_t[-1, :] - data_t[-2, :]
                new_data["scaler_data"] = scaler_tmp
                x_pos = np.vstack(data[xpos_name])

                if using_nanostage:
                    y_pos0 = np.vstack(data[ypos_name])
                else:
                    # get y position data, from differet stream name primary
                    data1 = hdr.table(fill=True, stream_name="primary")
                    if num_end_lines_excluded is not None:
                        data1 = data1[: datashape[0]]
                    # if ypos_name not in data1.keys() and 'E_tomo' not in start_doc['scaninfo']['type']:
                    # print(f"data1 keys: {data1.keys()}")
                    if ypos_name not in data1.keys():
                        ypos_name = "hf_stage_z"  # vertical along z
                    y_pos0 = np.hstack(data1[ypos_name])

                # Original comment (from the previous authors):
                #      y position is more than actual x pos, scan not finished?
                #
                # The following (temporary) fix assumes that incomplete scan contains
                #   at least two completed lines. The step between the scanned lines
                #   may be used to recreate y-coordinates for the lines that were not
                #   scanned: data for those lines will be filled with zeros.
                #   Having some reasonable y-coordinate data for the missed lines
                #   will allow to plot and process the data even if the scan is incomplete.
                #   In the case if scan contain only one line, there is no reliable way
                #   to to generate coordinates, use the same step as for x coordinates
                #   or 1 if the first scannned line contains only one point.

                # First check if the scan of the last line was completed. If not,
                #   then x-coordinates of the scan points are all ZERO
                last_scan_line_no_data = False
                if math.isclose(np.sum(x_pos[x_pos.shape[0] - 1, :]), 0.0, abs_tol=1e-20):
                    last_scan_line_no_data = True

                no_position_data = False
                if len(y_pos0) == 0 or (len(y_pos0) == 1 and last_scan_line_no_data):
                    no_position_data = True
                    print("WARNING: The scan contains no completed scan lines")

                if len(y_pos0) < x_pos.shape[0] and len(y_pos0) > 1:
                    # The number of the lines for which the scan was initiated
                    #   Unfortunately this is not the number of scanned lines,
                    #   so x-axis values need to be restored for the line #'n_scanned_lines - 1' !!!
                    n_scanned_lines = len(y_pos0)
                    print(f"WARNING: The scan is not completed: {n_scanned_lines} out of {x_pos.shape[0]} lines")
                    y_step = 1
                    if n_scanned_lines > 1:
                        y_step = (y_pos0[-1] - y_pos0[0]) / (n_scanned_lines - 1)
                    elif x_pos.shape[1] > 1:
                        # Set 'y_step' equal to the absolute value of 'x_step'
                        #    this is just to select some reasonable scale and happens if
                        #    only one line was completed in the unfinished flyscan.
                        #    This is questionable decision, but it should be rarely applied
                        y_step = math.fabs((x_pos[0, -1] - x_pos[0, 0]) / (x_pos.shape[1] - 1))
                    # Now use 'y_step' to generate the remaining points
                    n_pts = x_pos.shape[0] - n_scanned_lines
                    v_start = y_pos0[-1] + y_step
                    v_stop = v_start + (n_pts - 1) * y_step
                    y_pos_filled = np.linspace(v_start, v_stop, n_pts)
                    y_pos0 = np.append(y_pos0, y_pos_filled)
                    # Now duplicate x-coordinate values from the last scanned line to
                    #   all the unscanned lines, otherwise they all will be zeros
                    for n in range(n_scanned_lines - 1, x_pos.shape[0]):
                        x_pos[n, :] = x_pos[n_scanned_lines - 2, :]

                elif x_pos.shape[0] > 1 and last_scan_line_no_data:
                    # One possible scenario is that if the scan was interrupted while scanning
                    #   the last line. In this case the condition
                    #                 len(y_pos0) >= x_pos.shape[0]
                    #   will hold, but the last line of x-coordinates will be filleds with
                    #   zeros, which will create a mess if data is plotted with PyXRF
                    #   To fix the problem, fill the last line with values from the previous line
                    x_pos[-1, :] = x_pos[-2, :]

                # The following condition check is left from the existing code. It is still checking
                #   for the case if 0 lines were scanned.
                if len(y_pos0) >= x_pos.shape[0] and not no_position_data:
                    if using_nanostage:
                        yv = y_pos0
                    else:
                        y_pos = y_pos0[: x_pos.shape[0]]
                        x_tmp = np.ones(x_pos.shape[1])
                        xv, yv = np.meshgrid(x_tmp, y_pos)
                    # need to change shape to sth like [2, 100, 100]
                    data_tmp = np.zeros([2, x_pos.shape[0], x_pos.shape[1]])
                    data_tmp[0, :, :] = x_pos
                    data_tmp[1, :, :] = yv
                    new_data["pos_data"] = data_tmp
                    new_data["pos_names"] = ["x_pos", "y_pos"]
                    if vertical_fast is True:  # need to transpose the data, as we scan y first
                        # fast scan on y has impact for scaler data
                        data_tmp = np.zeros([2, x_pos.shape[1], x_pos.shape[0]])
                        data_tmp[1, :, :] = x_pos.T
                        data_tmp[0, :, :] = yv.T
                        new_data["pos_data"] = data_tmp

                else:
                    print("WARNING: Scan was interrupted: x,y positions are not saved")

            n_detectors_found += 1

            if output_to_file:
                # output to file
                print(f"Saving data to hdf file #{n_detectors_found}: Detector: {detector_name}.")
                fpath_out = save_data_to_hdf5(
                    fpath_out,
                    new_data,
                    metadata=mdata,
                    fname_add_version=fname_add_version,
                    file_overwrite_existing=file_overwrite_existing,
                    create_each_det=create_each_det,
                )

            # Preparing data for the detector ``detector_name`` for output
            d_dict = {
                "dataset": new_data,
                "file_name": fpath_out,
                "detector_name": detector_name,
                "metadata": mdata,
            }
            data_output.append(d_dict)

        print()
        if n_detectors_found == 0:
            print("ERROR: no data from known detectors were found in the database:")
            print("     Check that appropriate fields are included in 'xrf_fly_scan_detector_fields'")
            print(f"         of configuration file: {config_path}")
        else:
            print(f"Total of {n_detectors_found} detectors were found", end="")
            if output_to_file:
                print(f", {n_detectors_found} data files were created", end="")
            print(".")

        fln_list = [_["file_name"] for _ in data_output]
        logger.debug(f"Fly scan data was saved to the following files: {fln_list}")

        return data_output


def map_data2D_srx_new(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
    save_scaler=True,
    num_end_lines_excluded=None,
):

    if num_end_lines_excluded:
        logger.warning(
            "The data loading function for new SRX format does not support the parameter "
            "'num_end_lines_excluded' ({num_end_lines_excluded}). All available data will "
            "be included in the output file."
        )

    hdr = db[run_id_uid]
    start_doc = hdr.start
    runid = start_doc["scan_id"]  # Replace with the true value (runid may be relative, such as -2)

    print("**********************************************************")
    print(f"Loading scan #{runid}")
    print(f"Scan metadata format: version {start_doc['md_version']}")

    if completed_scans_only and not _is_scan_complete(hdr):
        raise Exception("Scan is incomplete. Only completed scans are currently processed.")
    if successful_scans_only and not _is_scan_successful(hdr):
        raise Exception(
            "Scan is not successfully completed. Only successfully completed scans are currently processed."
        )

    scan_doc = start_doc["scan"]
    stop_doc = hdr.stop

    print(f"Scan type: {scan_doc['type']}")

    # Check for detectors
    dets = []
    try:
        md_dets = hdr.start["scan"]["detectors"]
        for d in md_dets:
            if d in ("xs", "xs2", "xs4"):
                dets.append(d)
    except KeyError:
        # AMK forgot to add detectors to step scans
        # This is fixed, but left in for those scans
        if scan_doc["type"] == "XRF_STEP":
            dets.append("xs")

    if not (dets):
        raise IOError("No detectors found!")

    # Get metadata
    mdata = _extract_metadata_from_header(hdr)

    v = _get_metadata_value_from_descriptor_document(hdr, data_key="ring_current", stream_name="baseline")
    if v is not None:
        mdata["instrument_beam_current"] = v

    for ax in ["X", "Y", "Z"]:
        v = _get_metadata_all_from_descriptor_document(
            hdr, data_key=f"nanoKB_interferometer_pos{ax}", stream_name="baseline"
        )
        if v is not None:
            mdata[f"param_interferometer_pos{ax}"] = v

    # Get position data from scan
    n_scan_fast, n_scan_slow = hdr.start["scan"]["shape"]

    # ===================================================================
    #                     NEW SRX FLY SCAN
    # ===================================================================
    if scan_doc["type"] == "XRF_FLY":
        fast_motor = scan_doc["fast_axis"]["motor_name"]
        if fast_motor == "nano_stage_sx":
            fast_key = "enc1"
        elif fast_motor == "nano_stage_x":
            fast_key = "enc1"
        elif fast_motor == "nano_stage_sy":
            fast_key = "enc2"
        elif fast_motor == "nano_stage_y":
            fast_key = "enc2"
        elif fast_motor == "nano_stage_sz":
            fast_key = "enc3"
        else:
            raise IOError(f"{fast_motor} not found!")

        slow_motor = scan_doc["slow_axis"]["motor_name"]
        if slow_motor == "nano_stage_sx":
            slow_key = "enc1"
        elif slow_motor == "nano_stage_x":
            slow_key = "enc1"
        elif slow_motor == "nano_stage_sy":
            slow_key = "enc2"
        elif slow_motor == "nano_stage_y":
            slow_key = "enc2"
        elif slow_motor == "nano_stage_sz":
            slow_key = "enc3"
        else:
            slow_key = slow_motor

        # Let's get the data using the events! Yay!
        e = hdr.events("stream0", fill=True)
        ep = hdr.events("primary", fill=True)
        d_xs, d_xs_sum, N_xs = [], [], 0
        d_xs2, d_xs2_sum, N_xs2 = [], [], 0
        sclr_list = ["i0", "i0_time", "time", "im", "it"]
        sclr_dict = {}
        fast_pos, slow_pos = [], []

        n_recorded_events = 0

        try:
            for m, v in enumerate(e):
                if "xs" in dets or "xs4" in dets:
                    event_data = v["data"]["fluor"]
                    N_xs = max(N_xs, event_data.shape[1])
                    d_xs_sum.append(np.sum(event_data, axis=1))
                    if create_each_det:
                        d_xs.append(event_data)
                if "xs2" in dets:
                    event_data = v["data"]["fluor_xs2"]
                    N_xs2 = max(N_xs2, event_data.shape[1])
                    d_xs2_sum.append(np.sum(event_data, axis=1))
                    if create_each_det:
                        d_xs2.append(event_data)
                keys = v["data"].keys()
                for s in sclr_list:
                    if s in keys:
                        tmp = np.array(v["data"][s])
                        if s not in sclr_dict:
                            sclr_dict[s] = [tmp]
                        else:
                            sclr_dict[s].append(tmp)

                fast_pos.append(np.array(v["data"][fast_key]))
                if "enc" not in slow_key:
                    vp = next(ep)
                    tmp = np.array(vp["data"][slow_key])
                    tmp2 = [tmp] * n_scan_fast
                else:
                    tmp2 = v["data"][slow_key]
                slow_pos.append(np.array(tmp2))

                n_recorded_events = m + 1

                if m > 0 and not (m % 10):
                    print(f"Processed lines: {m}")

        except Exception as ex:
            logger.error(f"Error occurred while reading data: {ex}. Trying to retrieve available data ...")

        def repair_set(dset_list, n_row_pts):
            """
            Replaces corrupt rows (incorrect number of points) with closest 'good' row. This allows to load
            and use data from corrupt scans. The function will have no effect on 'good' scans.
            If there are no rows with correct number of points (unlikely case), then the array remains unchanged.
            """
            missed_rows = []
            n_last_good_row = -1
            for n in range(len(dset_list)):
                d = dset_list[n]
                n_pts = d.shape[0]
                if n_pts != n_row_pts:
                    print(f"WARNING: Row #{n + 1} has {n_pts} data points. {n_row_pts} points are expected.")
                    if n_last_good_row == -1:
                        missed_rows.append(n)
                    else:
                        dset_list[n] = np.array(dset_list[n_last_good_row])
                        print(f"Data in row #{n + 1} is replaced by data from row #{n_last_good_row}")
                else:
                    n_last_good_row = n
                    if missed_rows:
                        for nr in missed_rows:
                            dset_list[nr] = np.array(dset_list[n_last_good_row])
                            print(f"Data in row #{nr + 1} is replaced by data from row #{n_last_good_row}")
                        missed_rows = []

        sclr_name = list(sclr_dict.keys())

        repair_set(d_xs_sum, n_scan_fast)
        repair_set(d_xs, n_scan_fast)
        repair_set(d_xs2_sum, n_scan_fast)
        repair_set(d_xs2, n_scan_fast)
        repair_set(fast_pos, n_scan_fast)
        repair_set(slow_pos, n_scan_fast)
        for sc in sclr_dict.values():
            repair_set(sc, n_scan_fast)

        pos_pos = np.zeros((2, n_recorded_events, n_scan_fast))
        if "x" in slow_key:
            pos_pos[1, :, :] = fast_pos
            pos_pos[0, :, :] = slow_pos
        else:
            pos_pos[0, :, :] = fast_pos
            pos_pos[1, :, :] = slow_pos
        pos_name = ["x_pos", "y_pos"]

        if n_recorded_events != n_scan_slow:
            logger.error(
                "The number of recorded events (%d) is not equal to the expected number of events (%d): "
                "The scan is incomplete.",
                n_recorded_events,
                n_scan_slow,
            )

        # The following arrays may be empty if 'create_each_det == False' or the detector is not used.
        d_xs = np.asarray(d_xs)
        d_xs_sum = np.asarray(d_xs_sum)
        d_xs2 = np.asarray(d_xs2)
        d_xs2_sum = np.asarray(d_xs2_sum)

        sclr = np.zeros((n_recorded_events, n_scan_fast, len(sclr_name)))
        for n, sname in enumerate(sclr_name):
            sclr[:, :, n] = np.asarray(sclr_dict[sname])

    # ===================================================================
    #                     NEW SRX STEP SCAN
    # ===================================================================
    if scan_doc["type"] == "XRF_STEP":
        # Define keys for motor data
        fast_motor = scan_doc["fast_axis"]["motor_name"]
        fast_key = fast_motor + "_user_setpoint"
        slow_motor = scan_doc["slow_axis"]["motor_name"]
        slow_key = slow_motor + "_user_setpoint"

        # Collect motor positions
        fast_pos = hdr.data(fast_key, stream_name="primary", fill=True)
        fast_pos = np.array(list(fast_pos))
        slow_pos = hdr.data(slow_key, stream_name="primary", fill=True)
        slow_pos = np.array(list(slow_pos))

        # Reshape motor positions
        num_events = stop_doc["num_events"]["primary"]
        n_scan_slow, n_scan_fast = scan_doc["shape"]
        if num_events != (n_scan_slow * n_scan_fast):
            num_rows = num_events // n_scan_fast + 1  # number of rows
            fast_pos = np.zeros((num_rows, n_scan_fast))
            slow_pos = np.zeros((num_rows, n_scan_fast))
            for i in range(num_rows):
                for j in range(n_scan_fast):
                    fast_pos[i, j] = fast_pos[i * n_scan_fast + j]
                    slow_pos[i, j] = slow_pos[i * n_scan_fast + j]
        else:
            num_rows = n_scan_slow
            fast_pos = np.reshape(fast_pos, (n_scan_slow, n_scan_fast))
            slow_pos = np.reshape(slow_pos, (n_scan_slow, n_scan_fast))

        # Put into one array for h5 file
        pos_pos = np.zeros((2, num_rows, n_scan_fast))
        if "x" in slow_key:
            pos_pos[1, :, :] = fast_pos
            pos_pos[0, :, :] = slow_pos
        else:
            pos_pos[0, :, :] = fast_pos
            pos_pos[1, :, :] = slow_pos
        pos_name = ["x_pos", "y_pos"]

        # Get detector data
        keys = hdr.table().keys()
        MAX_DET_ELEMENTS = 8
        N_xs, det_name_prefix = None, None
        for i in np.arange(1, MAX_DET_ELEMENTS + 1):
            if f"xs_channel{i}" in keys:
                N_xs, det_name_prefix = i, "xs_channel"
            elif f"xs_channels_channel{i:02d}" in keys:
                N_xs, det_name_prefix = i, "xs_channels_channel"
            else:
                break
        N_pts = num_events
        N_bins = 4096
        if "xs" in dets or "xs4" in dets:
            d_xs = np.empty((N_xs, N_pts, N_bins))
            for i in np.arange(0, N_xs):
                if det_name_prefix == "xs_channel":
                    dname = det_name_prefix + f"{i + 1}"
                else:
                    dname = det_name_prefix + f"{i + 1:02d}"
                d = hdr.data(dname, fill=True)
                d = np.array(list(d))
                d_xs[i, :, :] = np.copy(d)
            del d
            # Reshape data
            if num_events != (n_scan_slow * n_scan_fast):
                tmp = np.zeros((N_xs, num_rows, n_scan_fast, N_bins))
                for i in range(num_rows):
                    for j in range(n_scan_fast):
                        tmp[:, i, j, :] = fast_pos[:, i * n_scan_fast + j, :]
                d_xs = np.copy(tmp)
                del tmp
            else:
                d_xs = np.reshape(d_xs, (N_xs, n_scan_slow, n_scan_fast, N_bins))
            # Sum data
            d_xs_sum = np.squeeze(np.sum(d_xs, axis=0))

        # Scaler list
        sclr_list = ["sclr_i0", "sclr_im", "sclr_it"]
        sclr_name = []
        for s in sclr_list:
            if s in keys:
                sclr_name.append(s)
        sclr = np.array(hdr.table()[sclr_name].values)
        # Reshape data
        if num_events != (n_scan_slow * n_scan_fast):
            tmp = np.zeros((num_rows, n_scan_fast))
            for i in range(num_rows):
                for j in range(n_scan_fast):
                    tmp[i, j] = fast_pos[i * n_scan_fast + j]
            sclr = np.copy(tmp)
            del tmp
        else:
            sclr = np.reshape(sclr, (n_scan_slow, n_scan_fast, len(sclr_name)))

    # Consider snake
    # pos_pos, d_xs, d_xs_sum, sclr
    if scan_doc["snake"] == 1:
        pos_pos[:, 1::2, :] = pos_pos[:, 1::2, ::-1]
        if "xs" in dets or "xs4" in dets:
            if d_xs.size:
                d_xs[:, 1::2, :, :] = d_xs[:, 1::2, ::-1, :]
            if d_xs_sum.size:
                d_xs_sum[1::2, :, :] = d_xs_sum[1::2, ::-1, :]
        if "xs2" in dets:
            if d_xs2.size:
                d_xs2[:, 1::2, :, :] = d_xs2[:, 1::2, ::-1, :]
            if d_xs2_sum.size:
                d_xs2_sum[1::2, :, :] = d_xs2_sum[1::2, ::-1, :]
        sclr[1::2, :, :] = sclr[1::2, ::-1, :]

    def swap_axes():
        nonlocal pos_name, pos_pos, d_xs, d_xs_sum, d_xs2, d_xs2_sum, sclr
        # Need to swapaxes on pos_pos, d_xs, d_xs_sum, sclr
        pos_name = pos_name[::-1]
        pos_pos = np.swapaxes(pos_pos, 1, 2)
        if "xs" in dets or "xs4" in dets:
            if d_xs.size:
                d_xs = np.swapaxes(d_xs, 0, 1)
            if d_xs_sum.size:
                d_xs_sum = np.swapaxes(d_xs_sum, 0, 1)
        if "xs2" in dets:
            if d_xs2.size:
                d_xs2 = np.swapaxes(d_xs2, 0, 1)
            if d_xs2_sum.size:
                d_xs2_sum = np.swapaxes(d_xs2_sum, 0, 1)
        sclr = np.swapaxes(sclr, 0, 1)

    if scan_doc["type"] == "XRF_FLY":
        if fast_motor in ("nano_stage_sy", "nano_stage_y"):
            swap_axes()
    elif scan_doc["type"] == "XRF_STEP":
        if "xs" in dets or "xs4" in dets:
            d_xs = np.swapaxes(d_xs, 0, 1)
            d_xs = np.swapaxes(d_xs, 1, 2)
        if "xs2" in dets:
            d_xs2 = np.swapaxes(d_xs2, 0, 1)
            d_xs2 = np.swapaxes(d_xs2, 1, 2)
        if fast_motor not in ("nano_stage_sy", "nano_stage_y"):
            swap_axes()
            pos_name = pos_name[::-1]  # Swap the positions back
        else:
            pos_name = pos_name[::-1]  # Swap the positions back

    print("Data is loaded successfully. Preparing to save data ...")

    data_output = []

    for detector_name in dets:
        if detector_name in ("xs", "xs4"):
            tmp_data = d_xs
            tmp_data_sum = d_xs_sum
            num_det = N_xs
        elif detector_name == "xs2":
            tmp_data = d_xs2
            tmp_data_sum = d_xs2_sum
            num_det = N_xs2

        loaded_data = {}
        loaded_data["det_sum"] = tmp_data_sum
        if create_each_det:
            for i in range(num_det):
                loaded_data["det" + str(i + 1)] = np.squeeze(tmp_data[:, :, i, :])

        if save_scaler:
            loaded_data["scaler_data"] = sclr
            loaded_data["scaler_names"] = sclr_name

        loaded_data["pos_data"] = pos_pos
        loaded_data["pos_names"] = pos_name

        # Generate the default file name for the scan
        if fpath is None:
            fpath = f"scan2D_{runid}.h5"

        # Modify file name (path) to include data on how many channels are included in the file and how many
        #    channels are used for sum calculation
        root, ext = os.path.splitext(fpath)
        s = f"_{detector_name}_sum{num_det}ch"
        if create_each_det:
            s += f"+{num_det}ch"
        fpath_out = f"{root}{s}{ext}"

        if output_to_file:
            # output to file
            print(f"Saving data to hdf file '{fpath_out}': Detector: {detector_name}.")
            fpath_out = save_data_to_hdf5(
                fpath_out,
                loaded_data,
                metadata=mdata,
                fname_add_version=fname_add_version,
                file_overwrite_existing=file_overwrite_existing,
                create_each_det=create_each_det,
            )

        d_dict = {
            "dataset": loaded_data,
            "file_name": fpath_out,
            "detector_name": detector_name,
            "metadata": mdata,
        }
        data_output.append(d_dict)

    return data_output


def map_data2D_tes(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
    save_scaler=True,
):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at TES beamline for step scan.
    Save the new data dictionary to hdf5 file if needed.

    .. note::

      It is recommended to read data from databroker into memory
    directly, instead of saving to files. This is ongoing work.

    Parameters
    ----------
    run_id_uid : int
        ID or UID of a run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector if data size is too large,
        if set as False. This will slow down the speed of creating an hdf5 file
        with large data size.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        is raised.
    output_to_file : bool, optional
        save data to hdf5 file if True

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """

    hdr = db[run_id_uid]
    runid = hdr.start["scan_id"]  # Replace with the true value (runid may be relative, such as -2)

    # The dictionary holding scan metadata
    mdata = _extract_metadata_from_header(hdr)
    # Some metadata is located at specific places in the descriptor documents
    # Search through the descriptor documents for the metadata
    v = _get_metadata_value_from_descriptor_document(hdr, data_key="mono_energy", stream_name="baseline")
    # Incident energy in the descriptor document is expected to be more accurate, so
    #   overwrite the value if it already exists
    if v is not None:
        mdata["instrument_mono_incident_energy"] = v / 1000.0  # eV to keV

    if completed_scans_only and not _is_scan_complete(hdr):
        raise Exception("Scan is incomplete. Only completed scans are currently processed.")
    if successful_scans_only and not _is_scan_successful(hdr):
        raise Exception(
            "Scan is not successfully completed. Only successfully completed scans are currently processed."
        )

    # Generate the default file name for the scan
    if fpath is None:
        fpath = f"scan2D_{runid}.h5"

    # Load configuration file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = "tes_pv_config.json"
    config_path = sep_v.join(current_dir.split(sep_v)[:-2] + ["configs", config_file])
    with open(config_path, "r") as json_data:
        config_data = json.load(json_data)

    # NOTE:
    #   Currently implemented algorithm will work only with the following flyscan:
    #     flyscanning along X-axis, stepping along Y-axis (to do otherwise or support both cases
    #     the function has to be modified).
    #   Each document will contain full data for a single line of N-point flyscan:
    #     N-element arrays with values for X and Y axis
    #     N-element arrays with values for each scaler
    #     N fluorescent spectra (each spectrum is 4096 points, saved by Xspress3 into
    #        separate file on GPFS, the document contains the path to file)

    print()
    print("****************************************")
    print("         Loading TES fly scan           ")
    print("****************************************")

    xpos_name = "x_centers"  # For now, we always fly on stage_x (fast axis)
    ypos_name = "y_centers"

    # The dictionary of fields that are used to store data from different detectors (for fly scan only)
    #   key - the name of the field used to store data read from the detector
    #   value - the detector name (probably short abbreviation, attached to the created file name so that
    #           the detector could be identified)
    # A separate data file is created for each detector

    # The following list will be used if the function is modified to work with multiple detectors
    # detector_field_dict = config_data['xrf_flyscan_detector_fields']

    spectrum_len = 4096  # It is typically fixed

    # Output data is the list of data structures for all available detectors
    data_output = []

    # The dictionary that will contain the data extracted from scan data
    #   This data will be saved to file and/or loaded into processing software
    new_data = {}

    def _is_row_missing(row_data):
        """
        Determine if the row is missing. Different versions of Databroker will return differnent value types.
        """
        if row_data is None:
            return True
        elif isinstance(row_data, np.ndarray) and (row_data.size == 1) and (row_data == np.array(None)):
            # This is returned by databroker.v0
            return True
        elif not len(row_data):
            return True
        else:
            return False

    def _get_row_len(row_data):
        if _is_row_missing(row_data):
            return 0
        else:
            return len(row_data)

    # Typically the scalers are saved
    if save_scaler is True:
        # Read the scalers
        scaler_names = config_data["scaler_list"]

        # Save all scaler names using lowercase letters
        scaler_names_lower = scaler_names.copy()
        for n in range(len(scaler_names)):
            scaler_names_lower[n] = scaler_names_lower[n].lower()
        new_data["scaler_names"] = scaler_names_lower

        n_scalers = len(config_data["scaler_list"])
        scaler_data = None
        data_shape = None
        for n, name in enumerate(scaler_names):
            s_data = hdr.table()[name]
            # Convert pandas dataframe to a list of ndarrays (.to_numpy())
            #   and then stack the arrays into a single 2D array
            s_data = s_data.to_numpy()

            # Find maximum number of points in a row.
            n_max_points = -1  # Maximum number of points in the row
            for row_data in s_data:
                n_max_points = max(n_max_points, _get_row_len(row_data))

            # Fix for the issue: 'empty' rows in scaler data. Fill 'empty' row
            #   with the nearest (preceding) row.
            # TODO: investigate the issue of 'empty' scaler ('dwell_time') rows at TES
            n_full = -1

            for _n in range(len(s_data)):
                if _is_row_missing(s_data[_n]) or (len(s_data[_n]) < n_max_points):
                    n_full = _n
                    break
            for _n in range(len(s_data)):
                # Data for the missing row is replaced by data from the previous 'good' row
                if _is_row_missing(s_data[_n]) or (len(s_data[_n]) < n_max_points):
                    s_data[_n] = np.copy(s_data[n_full])
                    logger.error(
                        f"Scaler '{name}': row #{_n} is corrupt or contains no data. "
                        f"Replaced by data from row #{n_full}"
                    )
                else:
                    n_full = _n

            s_data = np.vstack(s_data)
            if scaler_data is None:
                data_shape = s_data.shape
                scaler_data = np.zeros(shape=data_shape + (n_scalers,), dtype=float)
            scaler_data[:, :, n] = s_data
        new_data["scaler_data"] = scaler_data

    # Read x-y coordinates
    new_data["pos_names"] = ["x_pos", "y_pos"]
    pos_data = np.zeros(shape=(2,) + data_shape, dtype=float)
    # Convert pandas dataframes to 2D ndarrays
    pos_data[0, :, :] = np.vstack(hdr.table()[xpos_name].to_numpy())
    pos_data[1, :, :] = np.vstack(hdr.table()[ypos_name].to_numpy())
    new_data["pos_data"] = pos_data

    detector_field = "fluor"

    # Read detector values (for single detector)
    detector_data = np.zeros(shape=data_shape + (spectrum_len,), dtype=np.float32)
    n_events = data_shape[0]
    n_events_found = 0
    e = hdr.events(fill=True, stream_name="primary")

    n_pt_max = -1
    try:
        for n, v in enumerate(e):
            if n >= n_events:
                print("The number of lines is less than expected")
                break
            data = v.data[detector_field]
            data_det1 = np.array(data[:, 0, :], dtype=np.float32)

            # The following is the fix for the case when data has corrupt row (smaller number of data points).
            # It will not work if the first row is corrupt.
            n_pt_max = max(data_det1.shape[0], n_pt_max)
            data_det1_adjusted = np.zeros([n_pt_max, data_det1.shape[1]])
            data_det1_adjusted[: data_det1.shape[0], :] = data_det1

            detector_data[n, :, :] = data_det1_adjusted
            n_events_found = n + 1
    except Exception as ex:
        logger.error(f"Error occurred while reading data: {ex}. Trying to retrieve available data ...")

    if n_events_found < n_events:
        print("The number of lines is less than expected. The experiment may be incomplete")

    if n_events_found != n_events:
        # This will happen if data is corrupt, for example the experiment is interrupted prematurely.
        n_events_min = min(n_events_found, n_events)
        print(f"The map is resized: data for only {n_events_min} rows is available")
        detector_data = detector_data[:n_events_min, :, :]
        new_data["scaler_data"] = new_data["scaler_data"][:n_events_min, :, :]
        new_data["pos_data"] = new_data["pos_data"][:, :n_events_min, :]

    # Note: the following code assumes that the detector has only one channel.
    #   If the detector is upgraded, the following code will have to be rewritten, but
    #   the rest of the data loading procedure will have to be modified anyway.
    if create_each_det:
        new_data["det1"] = detector_data
    else:
        new_data["det_sum"] = detector_data

    num_det = 1
    detector_name = "xs"
    n_detectors_found = 1

    # Modify file name (path) to include data on how many channels are included in the file and how many
    #    channels are used for sum calculation
    root, ext = os.path.splitext(fpath)
    s = f"_{detector_name}_sum{num_det}ch"
    if create_each_det:
        s += f"+{num_det}ch"
    fpath_out = f"{root}{s}{ext}"

    if output_to_file:
        # output to file
        print(f"Saving data to hdf file #{n_detectors_found}: Detector: {detector_name}.")
        fpath_out = save_data_to_hdf5(
            fpath_out,
            new_data,
            metadata=mdata,
            fname_add_version=fname_add_version,
            file_overwrite_existing=file_overwrite_existing,
            create_each_det=create_each_det,
        )

    d_dict = {"dataset": new_data, "file_name": fpath_out, "detector_name": detector_name, "metadata": mdata}
    data_output.append(d_dict)

    return data_output


def map_data2D_xfm(
    run_id_uid,
    fpath,
    create_each_det=False,
    fname_add_version=False,
    completed_scans_only=False,
    successful_scans_only=False,
    file_overwrite_existing=False,
    output_to_file=True,
):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at XFM beamline for step scan.
    Save the new data dictionary to hdf file if needed.

    .. note:: It is recommended to read data from databroker into memory
    directly, instead of saving to files. This is ongoing work.

    Parameters
    ----------
    run_id_uid : int
        ID or UID of a run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size.
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: then conversion fails.
    completed_scans_only : bool
        True: process only completed scans (for which ``stop`` document exists in
        the database). Failed scan for which ``stop`` document exists are considered
        completed even if not the whole image was scanned. If incomplete scan is
        encountered: an exception is thrown.
        False: the feature is disabled, incomplete scan will be processed.
    file_overwrite_existing : bool, keyword parameter
        This option should be used if the existing file should be deleted and replaced
        with the new file with the same name. This option should be used with caution,
        since the existing file may contain processed data, which will be permanently deleted.
        True: overwrite existing files if needed. Note, that if ``fname_add_version`` is ``True``,
        then new versions of the existing file will always be created.
        False: do not overwrite existing files. If the file already exists, then the exception
        is raised.
    output_to_file : bool, optional
        save data to hdf5 file if True

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[run_id_uid]
    runid = hdr.start["scan_id"]  # Replace with the true value (runid may be relative, such as -2)

    if completed_scans_only and not _is_scan_complete(hdr):
        raise Exception("Scan is incomplete. Only completed scans are currently processed.")
    if successful_scans_only and not _is_scan_successful(hdr):
        raise Exception(
            "Scan is not successfully completed. Only successfully completed scans are currently processed."
        )

    # Generate the default file name for the scan
    if fpath is None:
        fpath = f"scan2D_{runid}.h5"

    # Output data is the list of data structures for all available detectors
    data_output = []

    # spectrum_len = 4096
    start_doc = hdr["start"]
    # The dictionary holding scan metadata
    mdata = _extract_metadata_from_header(hdr)
    plan_n = start_doc.get("plan_name")
    if "fly" not in plan_n:  # not fly scan
        datashape = start_doc["shape"]  # vertical first then horizontal
        fly_type = None

        snake_scan = start_doc.get("snaking")
        if snake_scan[1] is True:
            fly_type = "pyramid"

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = "xfm_pv_config.json"
        config_path = sep_v.join(current_dir.split(sep_v)[:-2] + ["configs", config_file])
        with open(config_path, "r") as json_data:
            config_data = json.load(json_data)

        # try except can be added later if scan is not completed.
        data = db.get_table(hdr, fill=True, convert_times=False)

        xrf_detector_names = config_data["xrf_detector"]
        data_out = map_data2D(
            data,
            datashape,
            det_list=xrf_detector_names,
            pos_list=hdr.start.motors,
            create_each_det=create_each_det,
            scaler_list=config_data["scaler_list"],
            fly_type=fly_type,
        )

        fpath_out = fpath

        if output_to_file:
            print("Saving data to hdf file.")
            fpath_out = save_data_to_hdf5(
                fpath_out,
                data_out,
                metadata=mdata,
                fname_add_version=fname_add_version,
                file_overwrite_existing=file_overwrite_existing,
                create_each_det=create_each_det,
            )

        detector_name = "xs"
        d_dict = {"dataset": data_out, "file_name": fpath_out, "detector_name": detector_name, "metadata": mdata}
        data_output.append(d_dict)

    return data_output


def write_db_to_hdf(
    fpath,
    data,
    datashape,
    det_list=("xspress3_ch1", "xspress3_ch2", "xspress3_ch3"),
    pos_list=("zpssx[um]", "zpssy[um]"),
    scaler_list=("sclr1_ch3", "sclr1_ch4"),
    fname_add_version=False,
    fly_type=None,
    subscan_dims=None,
    base_val=None,
):
    """
    Assume data is obained from databroker, and save the data to hdf file.
    This function can handle stopped/aborted scans.

    .. note:: This function should become part of suitcase

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: the exception is thrown if the file exists.
    """
    interpath = "xrfmap"

    if os.path.exists(fpath):
        if fname_add_version:
            fpath = _get_fpath_not_existing(fpath)
        else:
            raise IOError(f"'write_db_to_hdf': File '{fpath}' already exists.")

    with h5py.File(fpath, "a") as f:

        sum_data = None
        new_v_shape = datashape[0]  # to be updated if scan is not completed
        spectrum_len = 4096  # standard

        for n, c_name in enumerate(det_list):
            if c_name in data:
                detname = "det" + str(n + 1)
                dataGrp = f.create_group(interpath + "/" + detname)

                logger.info("read data from %s" % c_name)
                channel_data = data[c_name]

                # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
                new_v_shape = len(channel_data) // datashape[1]

                new_data = np.vstack(channel_data)
                new_data = new_data[: new_v_shape * datashape[1], :]

                new_data = new_data.reshape([new_v_shape, datashape[1], len(channel_data[1])])
                if new_data.shape[2] != spectrum_len:
                    # merlin detector has spectrum len 2048
                    # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                    new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                    new_tmp[:, :, : new_data.shape[2]] = new_data
                    new_data = new_tmp
                if fly_type in ("pyramid",):
                    new_data = flip_data(new_data, subscan_dims=subscan_dims)

                if sum_data is None:
                    sum_data = np.copy(new_data)
                else:
                    sum_data += new_data
                ds_data = dataGrp.create_dataset("counts", data=new_data, compression="gzip")
                ds_data.attrs["comments"] = "Experimental data from channel " + str(n)

        # summed data
        dataGrp = f.create_group(interpath + "/detsum")

        if sum_data is not None:
            sum_data = sum_data.reshape([new_v_shape, datashape[1], spectrum_len])
            ds_data = dataGrp.create_dataset("counts", data=sum_data, compression="gzip")
            ds_data.attrs["comments"] = "Experimental data from channel sum"

        # position data
        dataGrp = f.create_group(interpath + "/positions")

        pos_names, pos_data = get_name_value_from_db(pos_list, data, datashape)

        for i in range(len(pos_names)):
            if "x" in pos_names[i]:
                pos_names[i] = "x_pos"
            elif "y" in pos_names[i]:
                pos_names[i] = "y_pos"
        if "x_pos" not in pos_names or "y_pos" not in pos_names:
            pos_names = ["x_pos", "y_pos"]

        # need to change shape to sth like [2, 100, 100]
        data_temp = np.zeros([pos_data.shape[2], pos_data.shape[0], pos_data.shape[1]])
        for i in range(pos_data.shape[2]):
            data_temp[i, :, :] = pos_data[:, :, i]

        if fly_type in ("pyramid",):
            for i in range(data_temp.shape[0]):
                # flip position the same as data flip on det counts
                data_temp[i, :, :] = flip_data(data_temp[i, :, :], subscan_dims=subscan_dims)

        dataGrp.create_dataset("name", data=helper_encode_list(pos_names))
        dataGrp.create_dataset("pos", data=data_temp[:, :new_v_shape, :])

        # scaler data
        dataGrp = f.create_group(interpath + "/scalers")

        scaler_names, scaler_data = get_name_value_from_db(scaler_list, data, datashape)

        if fly_type in ("pyramid",):
            scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)

        dataGrp.create_dataset("name", data=helper_encode_list(scaler_names))

        if base_val is not None:  # base line shift for detector, for SRX
            base_val = np.array([base_val])
            if len(base_val) == 1:
                scaler_data = np.abs(scaler_data - base_val)
            else:
                for i in scaler_data.shape[2]:
                    scaler_data[:, :, i] = np.abs(scaler_data[:, :, i] - base_val[i])

        dataGrp.create_dataset("val", data=scaler_data[:new_v_shape, :])

    return fpath


def assemble_data_SRX_stepscan(
    data,
    datashape,
    det_list=("xspress3_ch1", "xspress3_ch2", "xspress3_ch3"),
    pos_list=("zpssx[um]", "zpssy[um]"),
    scaler_list=("sclr1_ch3", "sclr1_ch4"),
    fname_add_version=False,
    create_each_det=True,
    fly_type=None,
    subscan_dims=None,
    base_val=None,
):
    """
    Convert stepscan data from SRX beamline obtained from databroker into the for accepted
    by ``write_db_to_hdf_base`` function.
    This function can handle stopped/aborted scans.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    fname_add_version : bool
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_(1).h5, <fname>_(2).h5, etc.
        False: the exception is thrown if the file exists.
    create_each_det: bool
        True: output dataset contains data for individual detectors, False: output
        dataset contains only sum of all detectors.
    """

    data_assembled = {}

    sum_data = None
    new_v_shape = datashape[0]  # to be updated if scan is not completed
    spectrum_len = 4096  # standard

    for n, c_name in enumerate(det_list):
        if c_name in data:
            detname = "det" + str(n + 1)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]

            new_data = np.vstack(channel_data)
            new_data = new_data.astype(np.float32, copy=False)  # Change representation to np.float32
            new_data = new_data[: new_v_shape * datashape[1], :]

            new_data = new_data.reshape([new_v_shape, datashape[1], len(channel_data[1])])
            if new_data.shape[2] != spectrum_len:
                # merlin detector has spectrum len 2048
                # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                new_tmp[:, :, : new_data.shape[2]] = new_data
                new_data = new_tmp
            if fly_type in ("pyramid",):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)

            if sum_data is None:
                sum_data = np.copy(new_data)
            else:
                sum_data += new_data

            if create_each_det:
                data_assembled[detname] = new_data

    if sum_data is not None:
        data_assembled["det_sum"] = sum_data

    # position data
    pos_names, pos_data = get_name_value_from_db(pos_list, data, datashape)

    # I don't have knowledge of all possible scenarios to change the following algorithm for
    #   naming 'x_pos' and 'y_pos'. It definitely covers the basic cases of having x and y axis.
    #   It will also produce good dataset if the naming is inconsistent.
    for i in range(len(pos_names)):
        if "x" in pos_names[i]:
            pos_names[i] = "x_pos"
        elif "y" in pos_names[i]:
            pos_names[i] = "y_pos"
    if "x_pos" not in pos_names or "y_pos" not in pos_names:
        pos_names = ["x_pos", "y_pos"]

    # need to change shape to sth like [2, 100, 100]
    n_pos = min(pos_data.shape[2], len(pos_names))
    data_temp = np.zeros([n_pos, pos_data.shape[0], pos_data.shape[1]])

    for i in range(n_pos):
        data_temp[i, :, :] = pos_data[:, :, i]

    if fly_type in ("pyramid",):
        for i in range(data_temp.shape[0]):
            # flip position the same as data flip on det counts
            data_temp[i, :, :] = flip_data(data_temp[i, :, :], subscan_dims=subscan_dims)

    data_assembled["pos_names"] = pos_names
    data_assembled["pos_data"] = data_temp[:, :new_v_shape, :]

    # scaler data
    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data, datashape)

    if fly_type in ("pyramid",):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)

    if base_val is not None:  # base line shift for detector, for SRX
        base_val = np.array([base_val])
        if len(base_val) == 1:
            scaler_data = np.abs(scaler_data - base_val)
        else:
            for i in scaler_data.shape[2]:
                scaler_data[:, :, i] = np.abs(scaler_data[:, :, i] - base_val[i])

    data_assembled["scaler_names"] = scaler_names
    data_assembled["scaler_data"] = scaler_data[:new_v_shape, :]

    return data_assembled


def get_name_value_from_db(name_list, data, datashape):
    """
    Get name and data from db.
    """
    pos_names = []
    pos_data = np.zeros([datashape[0], datashape[1], len(name_list)])
    for i, v in enumerate(name_list):
        posv = np.zeros(
            datashape[0] * datashape[1]
        )  # keep shape unchanged, so stopped/aborted run can be handled.
        data[v] = np.asarray(data[v])  # in case data might be list
        posv[: data[v].shape[0]] = np.asarray(data[v])
        pos_data[:, :, i] = posv.reshape([datashape[0], datashape[1]])
        pos_names.append(str(v))
    return pos_names, pos_data


def map_data2D(
    data,
    datashape,
    det_list=("xspress3_ch1", "xspress3_ch2", "xspress3_ch3"),
    pos_list=("zpssx[um]", "zpssy[um]"),
    scaler_list=("sclr1_ch3", "sclr1_ch4"),
    create_each_det=False,
    fly_type=None,
    subscan_dims=None,
    spectrum_len=4096,
):
    """
    Data is obained from databroker. Transfer items from data to a dictionary of
    numpy array, which has 2D shape same as scanning area.

    This function can handle stopped/aborted scans. Raster scan (snake scan) is
    also considered.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    fly_type : string or optional
        raster scan (snake scan) or normal
    subscan_dims : 1D array or optional
        used at HXN, 2D of a large area is split into small area scans
    spectrum_len : int, optional
        standard spectrum length

    Returns
    -------
    dict of numpy array
    """
    data_output = {}
    new_v_shape = datashape[0]  # updated if scan is not completed
    sum_data = None

    for n, c_name in enumerate(det_list):
        if c_name in data:
            detname = "det" + str(n + 1)
            logger.info("read data from %s" % c_name)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]
            new_data = np.vstack(channel_data)
            new_data = new_data.astype(np.float32, copy=False)  # Change representation to np.float32
            new_data = new_data[: new_v_shape * datashape[1], :]
            new_data = new_data.reshape([new_v_shape, datashape[1], len(channel_data[1])])
            if new_data.shape[2] != spectrum_len:
                # merlin detector has spectrum len 2048
                # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len], dtype=np.float32)
                new_tmp[:, :, : new_data.shape[2]] = new_data
                new_data = new_tmp
            if fly_type in ("pyramid",):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)
            if create_each_det:
                data_output[detname] = new_data
            if sum_data is None:
                # Note: Here is the place where the error was found!!!
                #   The assignment in the next line used to be written as
                #      sum_data = new_data
                #   i.e. reference to data from 'det1' was assigned to 'sum_data'.
                #   After computation of the sum, both 'sum_data' and detector 'det1'
                #     were referencing the same ndarray, holding the sum of values
                #     from detector channels 'det1', 'det2' and 'det3'. In addition, the sum is
                #     computed again before data is saved into '.h5' file.
                #     The algorithm for computing of the second sum is working correctly,
                #     but since 'det1' already contains the true sum 'det1'+'det2'+'det3',
                #     the computed sum equals 'det1'+2*'det2'+2*'det3'.
                #   The problem was fixed by replacing assignment of reference during
                #   initalization of 'sum_data' by copying the array.
                # The error is documented because the code was used for a long time
                #   for initial processing of XRF imaging data at HXN beamline.
                sum_data = np.copy(new_data)
            else:
                sum_data += new_data
    data_output["det_sum"] = sum_data

    # scanning position data
    pos_names, pos_data = get_name_value_from_db(pos_list, data, datashape)
    for i in range(len(pos_names)):
        if "x" in pos_names[i]:
            pos_names[i] = "x_pos"
        elif "y" in pos_names[i]:
            pos_names[i] = "y_pos"
    if "x_pos" not in pos_names or "y_pos" not in pos_names:
        pos_names = ["x_pos", "y_pos"]

    if fly_type in ("pyramid",):
        for i in range(pos_data.shape[2]):
            # flip position the same as data flip on det counts
            pos_data[:, :, i] = flip_data(pos_data[:, :, i], subscan_dims=subscan_dims)
    new_p = np.zeros([len(pos_names), pos_data.shape[0], pos_data.shape[1]])
    for i in range(len(pos_names)):
        new_p[i, :, :] = pos_data[:, :, i]
    data_output["pos_names"] = pos_names
    data_output["pos_data"] = new_p

    # scaler data
    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data, datashape)
    if fly_type in ("pyramid",):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)
    data_output["scaler_names"] = scaler_names
    data_output["scaler_data"] = scaler_data
    return data_output


def _get_fpath_not_existing(fpath):
    # Returns path to the new file that is guaranteed to not exist
    # The function cycles through paths obtained by inserting
    #   version number between name and extension in the prototype path ``fpath``
    #  The version number is inserted in the form ``filename_v2.ext``

    if os.path.exists(fpath):
        p, e = os.path.splitext(fpath)
        n = 1
        while True:
            fpath = f"{p}_v{n}{e}"
            if not os.path.exists(fpath):
                break
            n += 1
    return fpath


def save_data_to_hdf5(
    fpath, data, *, metadata=None, fname_add_version=False, file_overwrite_existing=False, create_each_det=True
):
    """
    This is the function used to save raw experiment data into HDF5 file. The raw data is
    represented as a dictionary with the following keys:

      keys 'det1', 'det2' etc. - 3D ndarrays of size (N, M, K) where NxM are dimensions of the map
      and K is the number of spectrum points (4096) contain data from the detector channels 1, 2, 3 etc.

      key 'det_sum' - 3D ndarray with the same dimensions as 'det1' contains the sum of the channels

      key 'scaler_names' - the list of scaler names

      key 'scaler_data' - 3D ndarray of scaler values. The array shape is (N, M, P), where P is
      the number of scaler names.

      key 'pos_names' - the list of position (axis) names, must contain the names 'x_pos' and 'y_pos'
      in correct order.

      key 'pos_data' - 3D ndarray with position values. The array must have size (2, N, M). The first
          index is the number of the position name 'pos_names' list.

    Parameters
    ----------
    fpath: str
        Full path to the HDF5 file. The function creates an new HDF5 file. If file already exists
        and ``file_overwrite_existing=False``, then the IOError exception is raised.
    data : dict
        The dictionary of raw data.
    metadata : dict
        Metadata to be saved in the HDF5 file. The function will add or overwrite the existing
        metadata fields: ``file_type``, ``file_format``, ``file_format_version``, ``file_created_time``.
        User may define metadata fields ``file_software`` and ``file_software_version``. If ``file_software``
        is not defined, then the default values for ``file_software`` and ``file_software_version`` are added.
    fname_add_version : boolean
        True: if file already exists, then file version is added to the file name
        so that it becomes unique in the current directory. The version is
        added to <fname>.h5 in the form <fname>_v1.h5, <fname>_v2.h5, etc.
        False: the exception is raised if the file exists.
    file_overwrite_existing : boolean
        Overwrite the existing file or raise exception if the file exists.
    create_each_det : boolean
        Save data from individual detectors (``True``) or only the sum of fluorescence from
        all detectors (``False``).

    Raises
    ------
    IOError
        Failed to write data to HDF5 file.
    """

    fpath = os.path.expanduser(fpath)
    fpath = os.path.abspath(fpath)

    data = data.copy()  # Must be a shallow copy (avoid creating copies of data arrays)
    metadata = copy.deepcopy(metadata)  # Create deep copy (metadata is modified)

    interpath = "xrfmap"
    sum_data, sum_data_exists = None, False
    xrf_det_list = [n for n in data.keys() if "det" in n and "sum" not in n]
    xrf_det_list.sort()

    # Verify that raw fluorescence data is represented with np.float32 precision: print the warning message
    #   and convert the raw spectrum data to np.float32. Assume that data is represented as ndarray.
    def incorrect_type_msg(channel, data_type):
        logger.debug(
            f"Attemptying to save raw fluorescence data for the channel '{channel}' "
            f"as '{data_type}' numbers.\n    Memory may be used inefficiently. "
            f"The data is converted from '{data_type}' to 'np.float32' before saving to file."
        )

    if "det_sum" in data and isinstance(data["det_sum"], np.ndarray):
        if data["det_sum"].dtype != np.float32:
            incorrect_type_msg("det_sum", data["det_sum"].dtype)
            data["det_sum"] = data["det_sum"].astype(np.float32, copy=False)
        sum_data = data["det_sum"]
        sum_data_exists = True

    for detname in xrf_det_list:
        if detname in data and isinstance(data[detname], np.ndarray):
            if data[detname].dtype != np.float32:
                incorrect_type_msg(detname, data[detname].dtype)
                data[detname] = data[detname].astype(np.float32, copy=False)

            if not sum_data_exists:  # Don't compute it if it already exists
                if sum_data is None:
                    sum_data = np.copy(data[detname])
                else:
                    sum_data += data[detname]

    file_open_mode = "a"
    if os.path.exists(fpath):
        if fname_add_version:
            # Creates unique file name
            fpath = _get_fpath_not_existing(fpath)
        else:
            if file_overwrite_existing:
                # Overwrite the existing file. This completely deletes the HDF5 file,
                #   including all information (possibly processed results).
                file_open_mode = "w"
            else:
                raise IOError(f"Function 'save_data_to_hdf5': File '{fpath}' already exists")

    with h5py.File(fpath, file_open_mode) as f:

        # Create metadata group
        metadata_grp = f.create_group(f"{interpath}/scan_metadata")

        metadata_additional = {
            "file_type": "XRF-MAP",
            "file_format": "NSLS2-XRF-MAP",
            "file_format_version": "1.0",
            "file_created_time": ttime.strftime("%Y-%m-%dT%H:%M:%S+00:00", ttime.localtime()),
        }

        metadata_software_version = {
            "file_software": "PyXRF",
            "file_software_version": pyxrf_version,
        }

        metadata_prepared = metadata or {}
        metadata_prepared.update(metadata_additional)
        if "file_software" not in metadata_prepared:
            metadata_prepared.update(metadata_software_version)

        if metadata_prepared:
            # We assume, that metadata does not contain repeated keys. Otherwise the
            #   entry with the last occurrence of the key will override the previous ones.
            for key, value in metadata_prepared.items():
                metadata_grp.attrs[key] = value

        if create_each_det is True:
            for detname in xrf_det_list:
                new_data = data[detname]
                dataGrp = f.create_group(interpath + "/" + detname)
                ds_data = dataGrp.create_dataset("counts", data=new_data, compression="gzip")
                ds_data.attrs["comments"] = "Experimental data from {}".format(detname)

        # summed data
        if sum_data is not None:
            dataGrp = f.create_group(interpath + "/detsum")
            ds_data = dataGrp.create_dataset("counts", data=sum_data, compression="gzip")
            ds_data.attrs["comments"] = "Experimental data from channel sum"

        # add positions
        if "pos_names" in data:
            dataGrp = f.create_group(interpath + "/positions")
            pos_names = data["pos_names"]
            pos_data = data["pos_data"]
            dataGrp.create_dataset("name", data=helper_encode_list(pos_names))
            dataGrp.create_dataset("pos", data=pos_data)

        # scaler data
        if "scaler_data" in data:
            dataGrp = f.create_group(interpath + "/scalers")
            scaler_names = data["scaler_names"]
            scaler_data = data["scaler_data"]
            dataGrp.create_dataset("name", data=helper_encode_list(scaler_names))
            dataGrp.create_dataset("val", data=scaler_data)

    return fpath


write_db_to_hdf_base = save_data_to_hdf5  # Backward compatibility


'''
# This may not be needed, since hdr always goes out of scope
def clear_handler_cache(hdr):
    """
    Clear handler cache after loading data.

    Parameters
    ----------
    hdr
        reference to the handler
    """
    if LooseVersion(databroker.__version__) >= LooseVersion('1.0.0'):
        hdr._data_source.fillers['yes']._handler_cache.clear()
        hdr._data_source.fillers['delayed']._handler_cache.clear()
'''


# TODO: the following function may be deleted after Databroker 0.13 is forgotten
def free_memory_from_handler():
    """
    Quick way to set 3D dataset at handler to None to release memory.
    """
    # The following check is redundant: Data Broker prior to version 1.0.0 always has '_handler_cache'.
    #   In later versions of databroker the attribute may still be present if 'databroker.v0' is used.
    if (LooseVersion(databroker.__version__) < LooseVersion("1.0.0")) or hasattr(db.fs, "_handler_cache"):
        for h in db.fs._handler_cache.values():
            setattr(h, "_dataset", None)
        print("Memory is released.")


def export1d(runid, name=None):
    """
    Export all PVs to a file. Do not talk to filestore.

    Parameters
    ----------
    name : str or optional
        name for the file
    runid : int
        run number
    """
    t = db.get_table(db[runid], fill=False)
    if name is None:
        name = "scan_" + str(runid) + ".txt"
    t.to_csv(name)


def helper_encode_list(data, data_type="utf-8"):
    return [d.encode(data_type) for d in data]


def helper_decode_list(data, data_type="utf-8"):
    return [d.decode(data_type) for d in data]


def get_data_per_event(n, data, e, det_num):
    db.fill_event(e)
    min_len = e.data["fluor"].shape[0]
    for i in range(det_num):
        data[n, :min_len, :] += e.data["fluor"][:, i, :]


def get_data_parallel(data, elist, det_num):
    num_processors_to_use = multiprocessing.cpu_count() - 2

    print("cpu count: {}".format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    # result_pool = [
    #     pool.apply_async(get_data_per_event, (n, data, e, det_num))
    #     for n, e in enumerate(elist)]

    # results = [r.get() for r in result_pool]

    pool.terminate()
    pool.join()
