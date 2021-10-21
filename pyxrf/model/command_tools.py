# basic command line functions to use. Some of those functions can be implemented to
# scikit-beam later.
import os
import time
import json
import glob
import multiprocessing
import numpy as np
import h5py
import copy
from collections.abc import Iterable
import re

from ..core.quant_analysis import ParamQuantitativeAnalysis

from skbeam.core.fitting.xrf_model import linear_spectrum_fitting, define_range
from .fileio import output_data, read_hdf_APS, read_MAPS, sep_v
from .fit_spectrum import single_pixel_fitting_controller, save_fitdata_to_hdf

from ..core.map_processing import dask_client_create

import logging

logger = logging.getLogger(__name__)


def fit_pixel_data_and_save(
    working_directory,
    file_name,
    *,
    param_file_name,
    fit_channel_sum=True,
    fit_channel_each=False,
    param_channel_list=None,
    incident_energy=None,
    ignore_datafile_metadata=False,
    fln_quant_calib_data=None,
    quant_distance_to_sample=0,
    method="nnls",
    pixel_bin=0,
    raise_bg=0,
    comp_elastic_combine=False,
    linear_bg=False,
    use_snip=True,
    bin_energy=0,
    save_txt=False,
    save_tiff=True,
    scaler_name=None,
    use_average=False,
    interpolate_to_uniform_grid=False,
    data_from="NSLS-II",
    dask_client=None,
):
    """
    Do fitting for signle data set, and save data accordingly. Fitting can be performed on
    either summed data or each channel data, or both.

    Parameters
    ----------
    working_directory : str, required
        path folder
    file_names : str, required
        selected h5 file
    param_file_name : str, required
        param file name for summed data fitting
    fit_channel_sum : bool, optional
        fit summed data or not
    fit_channel_each : bool, optional
        fit each channel data or not
    param_channel_list : list, optional
        list of param file names for each channel
    incident_energy : float, optional
        use this energy as incident energy instead of the one in param file, i.e., XANES
        This value overrides the incident energy from metadata and from JSON parameter file
    ignore_datafile_metadata : bool
        tells whether to ignore metadata from the data file (if data file contains metadata).
        At the moment, only incident energy (monochromator energy) is used by the processing routine.
        If True, then the incident energy from the parameter JSON file is used; if False, then
        the energy from metadata is used. If the function parameter ``incident_energy`` is
        not None, then its value overrides the incident energy from metadata or JSON file.
    fln_quant_calib_data : str or list(str)
        file name or a list of file names that contain quantitative calibration data
    quant_distance_to_sample : float
        distance-to-sample used in quantitative calibration. If 0, then correction for
        distance is not applied (assumed that the standard and the sample were placed
        at the same distance from the detector).
    method : str, optional
        fitting method, default as nnls
    pixel_bin : int, optional
        bin pixel as 2by2, or 3by3
    raise_bg : int, optional
        add a constant value to each spectrum, better for fitting
    comp_elastic_combine : bool, optional
        combine elastic and compton as one component for fitting
    linear_bg : bool, optional
        use linear background instead of snip
    use_snip : bool, optional
        use snip method to remove background
    bin_energy : int, optional
        bin spectrum with given value
    save_txt : bool, optional
        save data to txt or not
    save_tiff : bool, optional
        save data to tiff or not
    scaler_name : str, optional
        name of the field representing the scaler (for example 'i0'), which must be present
        in the data files. If given, normalization will be performed before saving data
        as .txt and .tiff
    use_average : bool, optional
        if true, norm is performed as data/scaler*mean(scaler), otherwise just data/scaler
    interpolate_to_uniform_grid : bool
        interpolate the result to uniform grid before saving to tiff and txt files
        The grid dimensions match the dimensions of positional data for X and Y axes.
        The range of axes is chosen to fit the values of X and Y.
    data_from : str, optional
        where do data come from? Data format includes data from NSLS-II, or 2IDE-APS
    dask_client: dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time: `client = Client(processes=True, silence_logs=logging.ERROR)`
    """
    fpath = os.path.join(working_directory, file_name)

    # Load quantitative calibration files (if necessary)
    quant_norm = False  # Indicates if at least one calibration file is loaded
    param_quant_analysis = ParamQuantitativeAnalysis()
    if fln_quant_calib_data:
        if isinstance(fln_quant_calib_data, str):
            fln_quant_calib_data = [fln_quant_calib_data]
        for fln in fln_quant_calib_data:
            if os.path.isabs(fln):
                f = fln
            else:
                f = os.path.join(working_directory, fln)
            try:
                param_quant_analysis.load_entry(f)
                quant_norm = True
                logger.info(f"Quantitative calibration is loaded successfully from file '{f}'")
            except Exception as ex:
                logger.error(f"Error occurred while loading quantitative calibration from file '{f}': {ex}")

    t0 = time.time()
    prefix_fname = file_name.split(".")[0]
    if fit_channel_sum is True:
        if data_from == "NSLS-II":
            img_dict, data_sets, mdata = read_hdf_APS(working_directory, file_name, load_each_channel=False)
        elif data_from == "2IDE-APS":
            img_dict, data_sets, mdata = read_MAPS(working_directory, file_name, channel_num=1)
        else:
            print("Unknown data sets.")

        try:
            data_all_sum = data_sets[prefix_fname + "_sum"].raw_data
        except KeyError:
            data_all_sum = data_sets[prefix_fname].raw_data

        # load param file
        if not os.path.isabs(param_file_name):
            param_path = os.path.join(working_directory, param_file_name)
        else:
            param_path = param_file_name
        with open(param_path, "r") as json_data:
            param_sum = json.load(json_data)

        # update incident energy, required for XANES
        if incident_energy is not None:
            param_sum["coherent_sct_energy"]["value"] = incident_energy
            print("Using incident beam energy passed as the function parameter.")
        elif (
            mdata.is_metadata_available()
            and "instrument_mono_incident_energy" in mdata
            and not ignore_datafile_metadata
        ):
            param_sum["coherent_sct_energy"]["value"] = mdata["instrument_mono_incident_energy"]
            print(f"Using incident beam energy from the data file '{file_name}'.")
        else:
            print(f"Using incident beam energy from the parameter file '{param_path}'.")

        # The value of incident energy that is used for processing
        incident_energy_used = param_sum["coherent_sct_energy"]["value"]
        print(f"Incident beam energy: {incident_energy_used}.")

        result_map_sum, calculation_info = single_pixel_fitting_controller(
            data_all_sum,
            param_sum,
            incident_energy=incident_energy,
            method=method,
            pixel_bin=pixel_bin,
            raise_bg=raise_bg,
            comp_elastic_combine=comp_elastic_combine,
            linear_bg=linear_bg,
            use_snip=use_snip,
            bin_energy=bin_energy,
            dask_client=dask_client,
        )

        # output to .h5 file
        inner_path = "xrfmap/detsum"
        # fit_name = prefix_fname+'_fit'
        save_fitdata_to_hdf(fpath, result_map_sum, datapath=inner_path)

        def get_scaler_set(img_dict):
            sc_set_names = [_ for _ in img_dict if _.endswith("_scaler")]
            if sc_set_names:
                return img_dict[sc_set_names[0]]
            else:
                return {}

        def get_positions_set(img_dict):
            if "positions" in img_dict:
                return img_dict["positions"]
            else:
                return {}

        scaler_dict = get_scaler_set(img_dict)
        scaler_name_list = list(scaler_dict.keys())
        positions_dict = get_positions_set(img_dict)
        # Generate dataset
        dataset = copy.deepcopy(scaler_dict)
        dataset.update(result_map_sum)

        # Set parameters for quantitative normalization
        param_quant_analysis.experiment_incident_energy = incident_energy_used
        param_quant_analysis.experiment_distance_to_sample = quant_distance_to_sample
        param_quant_analysis.experiment_detector_channel = "sum"

        if save_txt is True:
            output_folder = "output_txt_" + prefix_fname
            output_path = os.path.join(working_directory, output_folder)
            output_data(
                output_dir=output_path,
                interpolate_to_uniform_grid=interpolate_to_uniform_grid,
                dataset_name="dataset_fit",  # Sum of all detectors: should end with '_fit'
                quant_norm=quant_norm,
                param_quant_analysis=param_quant_analysis,
                distance_to_sample=quant_distance_to_sample,
                dataset_dict=dataset,
                positions_dict=positions_dict,
                file_format="txt",
                scaler_name=scaler_name,
                scaler_name_list=scaler_name_list,
                use_average=use_average,
            )

        if save_tiff is True:
            output_folder = "output_tiff_" + prefix_fname
            output_path = os.path.join(working_directory, output_folder)
            output_data(
                output_dir=output_path,
                interpolate_to_uniform_grid=interpolate_to_uniform_grid,
                dataset_name="dataset_fit",  # Sum of all detectors: should end with '_fit'
                quant_norm=quant_norm,
                param_quant_analysis=param_quant_analysis,
                dataset_dict=dataset,
                positions_dict=positions_dict,
                file_format="tiff",
                scaler_name=scaler_name,
                scaler_name_list=scaler_name_list,
                use_average=use_average,
            )

    if fit_channel_each:
        img_dict, data_sets, mdata = read_hdf_APS(working_directory, file_name, load_each_channel=True)

        # Find the detector channels and the names of the channels
        det_channels = [_ for _ in data_sets.keys() if re.search(r"_det\d+$", _)]
        det_channel_names = [re.search(r"det\d+$", _)[0] for _ in det_channels]
        if param_channel_list is None:
            param_channel_list = [param_file_name] * 3
        else:
            if not isinstance(param_channel_list, list) and not isinstance(param_channel_list, tuple):
                raise RuntimeError("Parameter 'param_channel_list' must be a list or a tuple of strings")
            if len(param_channel_list) != len(det_channels):
                raise RuntimeError(
                    f"Parameter 'param_channel_list' must be 'None' or contain {len(det_channels)} file names."
                )

        channel_num = len(param_channel_list)
        for i in range(channel_num):
            inner_path = "xrfmap/" + det_channel_names[i]
            print(f"Processing data from detector channel {det_channel_names[i]} (#{i+1}) ...")

            # load param file
            param_file_name = param_channel_list[i]

            # load param file
            if not os.path.isabs(param_file_name):
                param_path = os.path.join(working_directory, param_file_name)
            else:
                param_path = param_file_name
            with open(param_path, "r") as json_data:
                param_det = json.load(json_data)

            # update incident energy, required for XANES
            if incident_energy is not None:
                param_det["coherent_sct_energy"]["value"] = incident_energy
                print("Using incident beam energy passed as the function parameter.")
            elif (
                mdata.is_metadata_available()
                and "instrument_mono_incident_energy" in mdata
                and not ignore_datafile_metadata
            ):
                param_det["coherent_sct_energy"]["value"] = mdata["instrument_mono_incident_energy"]
                print(f"Using incident beam energy from the data file '{file_name}'.")
            else:
                print(f"Using incident beam energy from the parameter file '{param_path}'.")

            # The value of incident energy that is used for processing
            incident_energy_used = param_det["coherent_sct_energy"]["value"]
            print(f"Incident beam energy: {incident_energy_used}.")

            data_all_det = data_sets[det_channels[i]].raw_data

            result_map_det, calculation_info = single_pixel_fitting_controller(
                data_all_det,
                param_det,
                incident_energy=incident_energy,
                method=method,
                pixel_bin=pixel_bin,
                raise_bg=raise_bg,
                comp_elastic_combine=comp_elastic_combine,
                linear_bg=linear_bg,
                use_snip=use_snip,
                bin_energy=bin_energy,
                dask_client=dask_client,
            )

            # output to .h5 file
            save_fitdata_to_hdf(fpath, result_map_det, datapath=inner_path)

            def get_scaler_set(img_dict):
                sc_set_names = [_ for _ in img_dict if _.endswith("_scaler")]
                if sc_set_names:
                    return img_dict[sc_set_names[0]]
                else:
                    return {}

            def get_positions_set(img_dict):
                if "positions" in img_dict:
                    return img_dict["positions"]
                else:
                    return {}

            scaler_dict = get_scaler_set(img_dict)
            scaler_name_list = list(scaler_dict.keys())
            positions_dict = get_positions_set(img_dict)
            # Generate dataset
            dataset = copy.deepcopy(scaler_dict)
            dataset.update(result_map_det)

            # Set parameters for quantitative normalization
            param_quant_analysis.experiment_incident_energy = incident_energy_used
            param_quant_analysis.experiment_distance_to_sample = quant_distance_to_sample
            param_quant_analysis.experiment_detector_channel = det_channel_names[i]

            if save_txt is True:
                output_folder = "output_txt_" + prefix_fname
                output_path = os.path.join(working_directory, output_folder)
                output_data(
                    output_dir=output_path,
                    interpolate_to_uniform_grid=interpolate_to_uniform_grid,
                    dataset_name=f"dataset_{det_channel_names[i]}_fit",  # ..._det1_fit, etc.
                    quant_norm=quant_norm,
                    param_quant_analysis=param_quant_analysis,
                    dataset_dict=dataset,
                    positions_dict=positions_dict,
                    file_format="txt",
                    scaler_name=scaler_name,
                    scaler_name_list=scaler_name_list,
                    use_average=use_average,
                )

            if save_tiff is True:
                output_folder = "output_tiff_" + prefix_fname
                output_path = os.path.join(working_directory, output_folder)
                output_data(
                    output_dir=output_path,
                    interpolate_to_uniform_grid=interpolate_to_uniform_grid,
                    dataset_name=f"dataset_{det_channel_names[i]}_fit",  # ..._det1_fit, etc.
                    quant_norm=quant_norm,
                    param_quant_analysis=param_quant_analysis,
                    dataset_dict=dataset,
                    positions_dict=positions_dict,
                    file_format="tiff",
                    scaler_name=scaler_name,
                    scaler_name_list=scaler_name_list,
                    use_average=use_average,
                )

    t1 = time.time()
    print(f"Processing time: {t1 - t0}")


def pyxrf_batch(
    start_id=None,
    end_id=None,
    *,
    param_file_name,
    data_files=None,
    wd=None,
    fit_channel_sum=True,
    fit_channel_each=False,
    param_channel_list=None,
    incident_energy=None,
    ignore_datafile_metadata=False,
    fln_quant_calib_data=None,
    quant_distance_to_sample=0,
    use_snip=True,
    save_txt=False,
    save_tiff=True,
    scaler_name=None,
    use_average=False,
    interpolate_to_uniform_grid=False,
    dask_client=None,
):
    """
    Perform fitting on a batch of data files. The results are saved as new datasets
    in the respective data files and may be viewed using PyXRF. Fitting can be performed on
    the sum of all detector channels, data from selected detector channels, or both.
    Internally, the function is calling the lower level function ``fit_pixel_data_and_save``.
    While it is possible to write processing scripts that call ``fit_pixel_data_and_save`` directly,
    but it is recommended, that ``pyxrf_batch`` is used instead.

    Parameters
    ----------
    start_id : int, optional
        starting run id
    end_id : int, optional
        ending run id
    param_file_name : str, required
        File name of the processing parameter file (JSON) used for data fitting.
        If the parameter file name in the list does not contain full
        path, then it is extended with the path in ``wd``.
    data_files : str, tuple (str,) or list [str], optional
        data file names: may be specified as a string for a single file,
        or iterable container (list, tuple) of strings for multiple files
        If ``data_files`` is specified (not None), then ``start_id`` and ``end_id``
        parameters are ignored. If a file name in the list does not contain full
        path, then it is extended with the path in ``wd``.
    wd : str, or optional
        path folder, default is the current folder
    fit_channel_sum : bool, optional
        fit summed data or not
    fit_channel_each : bool, optional
        fit each channel data or not
    param_channel_list : list, optional
        list of param file names for each channel
    incident_energy : float, optional
        use this energy as incident energy instead of the one in param file, i.e., XANES
        This value overrides the incident energy from metadata and from JSON parameter file.
    ignore_datafile_metadata : bool
        tells whether to ignore metadata from the data file (if data file contains metadata).
        At the moment, only incident energy (monochromator energy) is used by the processing routine.
        If True, then the incident energy from the parameter JSON file is used; if False, then
        the energy from metadata is used. If the function parameter ``incident_energy`` is
        not None, then its value overrides the incident energy from metadata or JSON file.
        Default: False.
    fln_quant_calib_data : str or list(str)
        file name or a list of file names that contain quantitative calibration data
    quant_distance_to_sample : float
        distance-to-sample used in quantitative calibration. If 0, then correction for
        distance is not applied (assumed that the standard and the sample were placed
        at the same distance from the detector).
    use_snip : bool, optional
        use snip method to remove background (`True`). If `False`, then do fitting
        without removing the background (runs faster)
    save_txt : bool, optional
        save data to txt or not
    save_tiff : bool, optional
        save data to tiff or not
    scaler_name : str, optional
        if given, normalization will be performed
    use_average : bool, optional
        if true, norm is performed as data/scaler*mean(scaler), otherwise just data/scaler
    interpolate_to_uniform_grid : bool
        interpolate the result to uniform grid before saving to tiff and txt files
        The grid dimensions match the dimensions of positional data for X and Y axes.
        The range of axes is chosen to fit the values of X and Y.
    dask_client: dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time:

        .. code:: python

            from pyxrf.api import dask_client_create
            client = dask_client_create()  # Create Dask client
            # <-- code that runs computations -->
            client.close()  # Close Dask client

    Returns
    -------

    flist : [str]
        list of file names of processed files: the list is empty
        if no files were processed. The file names contain relative path
        (combination of ``wd`` and file names).
        The file names are absolute if absolute paths are supplied in ``data_files``.
        If file list is created based on scan IDs, then the paths are always relative.

    Exceptions
    ----------

    ``ValueError`` is raised if the ``param_file_name" is not a string or is referencing
    non-existing file or ``data_files`` is specified, but not a string or iterable object
    containing strings.

    ``IOError`` is raised if data file does not exist. The exception is raised only if
    a single scan is to be processed (``end_id`` is not specied or None) or ``data_files``
    is set to a string with a single data file name. If the range of scan IDs that
    contains one scan (``start_id == end_id``) is specified, or ``data_files`` is
    a list containing a single file name, the exception will not be raised. This
    is consistent with the behavior of the ``make_hdf`` function.

    If an exception is raised by the processing code, it is reraised if
    processing of a single experiment is requested (the same conditions as for ``IOError``).
    Otherwise the error message is printed and processing of the next file or scan in the list
    is attempted.

    The data files and the processing parameter file may be in different directories. If the parameter
    file is in a different directory from the data files, then the full path to the parameter
    file should be specified:

        .. code: python

            param_file_name="/home/user/data_parameters/parameters.json"
            param_file_name="~/data_parameters/parameters.json"

    How does ``pyxrf-batch`` identify a file with a given Scan ID?
    In order for the batch mode processing to work correctly, the data file name must
    conform to the following structure requirements:

        <prefix>_<scanID>(any non-digit character)(any character sequence).h5

    <prefix> - any sequence of characters, not containing ``_``
    <scanID> - string version of Scan ID, obtained as ``str(ScanID)``
    <prefix> and <scanID> are separated by ``_``
    <scanID> is separated from the rest of the file name from the right by any non-number character.
    Files must have extension .h5. (Note, that ``.`` separating the extension is a non-number character
    that may terminate the Scan ID part of the file name

    For example, the following files will be recognized as Scan ID 28355:

        scan2D_28355.h5
        scanNewName_28355_some_comments.h5
        scan_2D_28355.h5  (incorrect, recognized as Scan ID 2, not 28355)
        scan2D28355.h5 (incorrect, no ``_``)

    Examples
    --------

    In the following examples it is assumed that all .h5 files and the .json parameter file are
    located in the current directory.

    -- Process data file with Scan ID 455.

        .. code: python

            pyxrf_batch(455, param_file_name="parameters.json")

    -- Process data files with Scan IDs in the range 455 .. 457:

        .. code: python

            pyxrf_batch(455, 457, param_file_name="parameters.json")

    -- Process all data files in the current directory:

        .. code: python

            pyxrf_batch(param_file_name="parameters.json")

    """

    print()

    if wd is None:
        wd = "."
    else:
        wd = os.path.expanduser(wd)
    param_file_name = os.path.expanduser(param_file_name)

    allow_raising_exceptions = False
    if isinstance(data_files, str):
        allow_raising_exceptions = True
    elif (start_id is not None) and (end_id is None):
        allow_raising_exceptions = True

    if fln_quant_calib_data is not None:
        if not isinstance(fln_quant_calib_data, str) and not isinstance(fln_quant_calib_data, list):
            raise ValueError("Parameter 'fln_quant_calib_data' must be a string or a list of strings")
        if isinstance(fln_quant_calib_data, list) and any([not isinstance(_, str) for _ in fln_quant_calib_data]):
            raise ValueError("List passed with the parameter 'fln_quant_calib_data' must contain only strings")

    if not isinstance(quant_distance_to_sample, float) and not isinstance(quant_distance_to_sample, int):
        raise ValueError("Value of the parameter 'quant_distance_to_sample' must be floating point number")

    if data_files is not None:

        # Check if ``data_files`` has valid value
        data_files_valid = False
        if isinstance(data_files, str):
            data_files = [data_files]  # Convert to list
            data_files_valid = True
        elif isinstance(data_files, Iterable):
            data_files = list(data_files)  # Convert to list
            data_files_valid = True
            for fln in data_files:
                if not isinstance(fln, str):
                    data_files_valid = False
                    break

        if not data_files_valid:
            raise ValueError(
                f"Function pyxrf_batch. Parameter 'data_files' has invalid format. \n"
                f"data_files = '{data_files}'"
            )

        # At this point ``data_files`` is a list of str.
        data_files = [os.path.expanduser(fln) for fln in data_files]

        # Working directory name is appended to each file name in the list,
        data_files = [fln if os.path.isabs(fln) else os.path.join(wd, fln) for fln in data_files]

        # Create the list of files. Include only existing files in the list.
        flist = []
        for fln in data_files:
            if os.path.exists(fln) and os.path.isfile(fln):
                flist.append(fln)
            else:
                if allow_raising_exceptions:
                    raise IOError(f"File '{fln}' does not exist")
                else:
                    print(f"WARNING: file '{fln}' does not exist.")

    else:

        # ``data_files`` parameter is None

        all_files = glob.glob(os.path.join(wd, "*.h5"))

        # Sort files (processing unsorted list of files is unsightly)
        all_files.sort()

        if start_id is None and end_id is None:
            # ``start_id`` and ``end_id`` are not specified:
            #   process all .h5 files in the current directory
            flist = all_files
        elif end_id is None:
            # only ``start_id`` is specified:
            #   process only one file that contains ``start_id`` in its name
            #   (only if such file exists)
            pattern = f"^[^_]*_{str(start_id)}\D+"  # noqa: W605
            flist = [fname for fname in all_files if re.search(pattern, os.path.basename(fname))]

            if len(flist) < 1:
                msg = f"File with Scan ID {start_id} was not found"
                if allow_raising_exceptions:
                    raise IOError(msg)
                else:
                    print(msg)
            else:
                print(f"Processing file with Scan ID {start_id}")
        else:
            # ``start_id`` and ``end_id`` are specified:
            #   select files, which contain the respective ID substring in their names
            flist = []
            for data_id in range(start_id, end_id + 1):
                pattern = f"^[^_]*_{str(data_id)}\D+"  # noqa: W605
                flist += [fname for fname in all_files if re.search(pattern, os.path.basename(fname))]
            if len(flist) < 1:
                print(f"No files with Scan IDs in the range {start_id} .. {end_id} were found.")
            else:
                print(f"Processing file with Scan IDs in the range {start_id} .. {end_id}")

    # Check if .json parameter file exists
    if not os.path.isabs(param_file_name):
        pname = os.path.join(wd, param_file_name)
    else:
        pname = param_file_name
    if not os.path.exists(pname) or not os.path.isfile(pname):
        raise IOError(f"Function 'pyxrf_batch'. Processing parameter file '{pname}' does not exist.")
    else:
        print(f"Processing parameter file: '{pname}'")

    if len(flist) > 0:
        # If no external Dask client is provided and we are processing a batch
        #   then create a local client that will be used to process the whole batch
        if (len(flist) > 1) and (dask_client is None):
            logger.info("Creating local Dask client for processing the batch of files ...")
            dask_client = dask_client_create()
            client_is_local = True
        else:
            client_is_local = False

        def _dask_client_close(is_local):
            # We don't want to close an externally provided client
            if is_local:
                logger.info("Closing the local Dask client ...")
                dask_client.close()

        print("The following files are scheduled for processing:")
        for fln in flist:
            print(f"    {fln}")
        print(f"Total number of selected files: {len(flist)}\n")

        for fpath in flist:
            print(f"Processing file '{fpath}' ...")
            fname = fpath.split(sep_v)[-1]
            working_directory = fpath[: -len(fname)]
            try:
                fit_pixel_data_and_save(
                    working_directory,
                    fname,
                    fit_channel_sum=fit_channel_sum,
                    param_file_name=param_file_name,
                    fit_channel_each=fit_channel_each,
                    param_channel_list=param_channel_list,
                    incident_energy=incident_energy,
                    ignore_datafile_metadata=ignore_datafile_metadata,
                    fln_quant_calib_data=fln_quant_calib_data,
                    quant_distance_to_sample=quant_distance_to_sample,
                    use_snip=use_snip,
                    save_txt=save_txt,
                    save_tiff=save_tiff,
                    scaler_name=scaler_name,
                    use_average=use_average,
                    interpolate_to_uniform_grid=interpolate_to_uniform_grid,
                    dask_client=dask_client,
                )
            except Exception as ex:
                if allow_raising_exceptions:
                    _dask_client_close(client_is_local)
                    raise Exception from ex
                else:
                    print(f"ERROR: could not process the file '{fname}'. No results are saved.")

        print("\nAll selected files were processed.")

        _dask_client_close(client_is_local)

    else:

        print("No files were selected for processing.")

    return flist


def fit_each_pixel_with_nnls(data, params, elemental_lines=None, incident_energy=None, weights=None):
    """
    Fit a spectrum with a linear model.

    Parameters
    ----------
    data : array
        spectrum intensity
    param : dict
        fitting parameters
    elemental_lines : list, optional
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum. If elemental_lines is set as None,
            all the possible lines activated at given energy will be used.
    """
    param = copy.deepcopy(params)
    if incident_energy is not None:
        param["coherent_sct_amplitude"]["value"] = incident_energy
    # cut data into proper range
    low = param["non_fitting_values"]["energy_bound_low"]["value"]
    high = param["non_fitting_values"]["energy_bound_high"]["value"]
    a0 = param["e_offset"]["value"]
    a1 = param["e_linear"]["value"]
    x, y = define_range(data, low, high, a0, a1)
    # pixel fitting
    _, result_dict, area_dict = linear_spectrum_fitting(x, y, elemental_lines=elemental_lines, weights=weights)
    return result_dict


def fit_pixel_per_file_no_multi(dir_path, file_prefix, fileID, param, interpath, save_spectrum=True):
    """
    Single pixel fit of experiment data. No multiprocess is applied.

    .. warning :: This function is not optimized as it calls
    linear_spectrum_fitting function, where lots of repeated
    calculation are processed.

    Parameters
    ----------
    data : array
        3D data of experiment spectrum
    param : dict
        fitting parameters

    Returns
    -------
    dict :
        fitting values for all the elements
    """

    num_str = "{:03d}".format(fileID)
    filename = file_prefix + num_str
    file_path = os.path.join(dir_path, filename)
    with h5py.File(file_path, "r") as f:
        data = f[interpath][:]
    datas = data.shape

    elist = param["non_fitting_values"]["element_list"].split(", ")
    elist = [e.strip(" ") for e in elist]

    non_element = ["compton", "elastic", "background"]
    total_list = elist + non_element

    result_map = dict()
    for v in total_list:
        if save_spectrum:
            result_map.update({v: np.zeros([datas[0], datas[1], datas[2]])})
        else:
            result_map.update({v: np.zeros([datas[0], datas[1]])})

    for i in range(datas[0]):
        for j in range(datas[1]):
            x, result, area_v = linear_spectrum_fitting(
                data[i, j, :], param, elemental_lines=elist, constant_weight=1.0
            )
            for v in total_list:
                if v in result:
                    if save_spectrum:
                        result_map[v][i, j, : len(result[v])] = result[v]
                    else:
                        result_map[v][i, j] = np.sum(result[v])

    return result_map


def fit_data_multi_files(dir_path, file_prefix, param, start_i, end_i, interpath="entry/instrument/detector/data"):
    """
    Fitting for multiple files with Multiprocessing.

    Parameters
    ----------
    dir_path : str
    file_prefix : str
    param : dict
    start_i : int
        start id of given file
    end_i: int
        end id of given file
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : list
        fitting result as list of dict
    """
    num_processors_to_use = multiprocessing.cpu_count()
    logger.info("cpu count: {}".format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    result_pool = [
        pool.apply_async(fit_pixel_per_file_no_multi, (dir_path, file_prefix, m, param, interpath))
        for m in range(start_i, end_i + 1)
    ]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()
    return results


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s : %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # pyxrf_batch(start_id=92276, end_id=92276,
    #             param_file_name="param_335",
    #             wd=".", save_tiff=False)

    pyxrf_batch(
        start_id=63339,
        param_file_name="test",
        wd=".",
        save_tiff=True,
        fln_quant_calib_data=["standard_41147a.json"],
        quant_distance_to_sample=2.0,
        scaler_name="i0",
        # fln_quant_calib_data=['standard_32654A.json',
        #                       'standard_32654A a.json',
        #                       'standard_41147.json'],
        fit_channel_sum=True,
        fit_channel_each=True,
        # param_channel_list=['det1', 'det2', 'det3']
    )
