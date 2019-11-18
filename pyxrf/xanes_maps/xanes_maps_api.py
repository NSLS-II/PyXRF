import os
import re
import numpy as np
from pystackreg import StackReg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import time as ttime
import tifffile

from skbeam.core.fitting.xrf_model import nnls_fit

from ..model.load_data_from_db import make_hdf
from ..model.command_tools import pyxrf_batch
from ..model.fileio import read_hdf_APS
from ..model.utils import (grid_interpolate, normalize_data_by_scaler, convert_time_to_nexus_string,
                           check_if_eline_is_activated, check_eline_name)

import logging
logger = logging.getLogger()


def build_xanes_map(*args, **kwargs):
    """
    A wrapper for the function ``build_xanes_map_api`` that catches exceptions
    and prints the error message. Use this wrapper to run processing manually from
    iPython and use ``build_xanes_map_api`` for custom scripts.

    For description of the function parameters see the docstring for
    ``build_xanes_map_api``
    """
    try:
        build_xanes_map_api(*args, **kwargs)
    except BaseException as ex:
        msg = f"Processing is incomplete! Exception was raised during execution:\n   {ex}"
        logger.error(msg)
    else:
        logger.info("Processing was completed successfully.")


def build_xanes_map_api(start_id=None, end_id=None, *, param_file_name,
                        scaler_name=None,
                        wd=None,
                        xrf_subdir="xrf_data",
                        sequence="build_xanes_map",
                        emission_line,
                        emission_line_alignment=None,
                        alignment_starts_from="top",
                        interpolation_enable=True,
                        alignment_enable=True,
                        ref_file_name=None,
                        incident_energy_low_bound=None,
                        use_incident_energy_from_param_file=True,
                        plot_results=True,
                        plot_use_position_coordinates=True,
                        plot_position_axes_units="$\mu $m",
                        output_file_formats = ["tiff"]):

    """
    Parameters
    ----------

    start_id : int
        scan ID of the first run of the sequence

    end_id : int
        scan ID of the last run of the sequence

        When processing data files from the local drive, both ``start_id`` and ``end_id``
        may be set to ``None`` (default). In this case all files in the ``xrf_subdir``
        subdirectory of the working directory ``wd`` are loaded and processed. The values
        of ``start_id`` and ``end_id`` will limit the range of processed scans.
        The values of ``start_id`` and ``end_id`` must be set to proper values in order
        to load data from the databroker.

    param_file_name : str
        the name of the JSON parameter file. The parameters are used for automated
        processing of data with ``pyxrf_batch``. The parameter file is typically produced
        by PyXRF.

    wd : str
        working directory: if ``wd`` is not specified then current directory will be
        used as the working directory for processing

    xrf_subdir : str
        subdirectory inside the working directory in which raw data files will be loaded.
        If the "process" or "build_xanes_map" sequence is executed, then the program
        looks for raw or processed files in this subfolder. In majority of cases, the
        default value is sufficient.

    sequence : str
        the sequence of operations performed by the function:

        -- ``load_and_process`` - loading data and full processing,

        -- ``process`` - full processing of data, including xrf mapping
        and building the energy map,

        -- ``build_xanes_map`` - build the energy map using xrf mapping data,
        all data must be processed.

    emission_line : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    emission_line_alignment : str
        the name of the emission line used for image alignment ("Ca_K", "Fe_K", etc.).
        If None, then the line specified as ``emission_line`` used for alignment

    alignment_starts_from : str
        The order of alignment of the image stack: "top" - start from the top of the stack
        (scan with highest energy) and proceed to the bottom of the stack (lowest energy),
        "bottom" - start from the bottom of the stack (lowest energy) and proceed to the top.
        Starting from the top typically produces better results, because scans at higher
        energy are more defined. If the same emission line is selected for XANES and
        for alignment reference, then alignment should always be performed starting from
        the top of the stack.

    interpolation_enable : True
        enable interpolation of XRF maps to uniform grid before alignment of maps.

    alignment_enable : True
        enable alignment of the stack of maps. In typical processing workflow the alignment
        should be enabled.

    ref_file_name : str
        file name with emission line references. If ``ref_file_name`` is not provided,
        then no XANES maps are generated. The rest of the processing is still performed
        as expected.

    incident_energy_low_bound : float
        files in the set are processed using the value of incident energy which equal to
        the greater of the values of ``incident_energy_low_bound`` or incident energy
        from file metadata. If None, then the lower energy bound is found automatically
        as the largest value of energy in the set which still activates the selected
        emission line (specified as the ``emission_line`` parameter). This parameter
        overrides the parameter ``use_incident_energy_from_param_file``.

    use_incident_energy_from_param_file : bool
        indicates if incident energy from parameter file will be used to process all
        files: True - use incident energy from parameter files, False - use incident
        energy from data files. If ``incident_energy_low_bound`` is specified, then
        this parameter is ignored.

    plot_results : bool
        indicates if results (image stack and XANES maps) are to be plotted. If set to
        False, the processing results are saved to file (if enabled) and the program
        exits without showing data plots.

    plot_use_position_coordinates : bool
        results (image stack and XANES maps) are plotted vs. position coordinates if
        the parameter is set to True, otherwise images are plotted vs. pixel number

    plot_position_axes_units : str
        units for position coordinates along X and Y axes. The units are used while
        plotting the results vs. position coordinates. The string specifying units
        may contain LaTeX expressions: for example ``"$\mu $m"`` will print units
        of ``micron`` as part of X and Y axes labels.

    file_output_formats : list(str)
        list of output file formats. Currently only "tiff" format is supported
        (XRF map stack and XANES maps are saved as stacked TIFF files).

    Returns
    -------

    Throws exception if processing can not be completed. The error message may be printed to
    indicate the reason of the failure to the user.
    """

    if wd is None:
        wd = '.'
    else:
        wd = os.path.expanduser(wd)
    wd = os.path.abspath(wd)

    param_file_name = os.path.expanduser(param_file_name)
    param_file_name = os.path.abspath(param_file_name)

    if ref_file_name:
        ref_file_name = os.path.expanduser(ref_file_name)
        ref_file_name = os.path.abspath(ref_file_name)

    if not xrf_subdir:
        raise ValueError("The parameter 'xrf_subdir' is None or contains an empty string "
                         "('build_xanes_map_api').")

    if not scaler_name:
        logger.warning("Scaler was not specified. The processing will still be performed,"
                       "but the DATA WILL NOT BE NORMALIZED!")

    # Convert all tags specifying output format to lower case
    for n, fmt in enumerate(output_file_formats):
        output_file_formats[n] = fmt.lower()
    # Now check every value agains the list of supported formats.
    supported_formats = ("tiff",)
    for fmt in output_file_formats:
        assert (fmt in supported_formats), f"Output format '{fmt}' is not supported. "\
                                           "Check values of the parameter 'output_file_formats'"

    alignment_starts_from_values = ["top", "bottom"]
    alignment_starts_from = alignment_starts_from.lower()
    if alignment_starts_from not in alignment_starts_from_values:
        raise ValueError("The parameter 'alignment_starts_from' has illegal value "
                         f"'{alignment_starts_from}' ('build_xanes_map_api').")

    # Set emission lines
    eline_selected = emission_line
    if emission_line_alignment:
        eline_alignment = emission_line_alignment
    else:
        eline_alignment = eline_selected

    # Check emission line names. They must be in the list of supported emission lines.
    #   The check is case-sensitive.
    if not check_eline_name(eline_selected):
        raise ValueError(f"The emission line '{eline_selected}' does not exist or is not supported. "
                         f"Check the value of the parameter 'eline_selected' ('build_xanes_map_api').")
    if not check_eline_name(eline_alignment):
        raise ValueError(f"The emission line '{eline_alignment}' does not exist or is not supported. "
                         f"Check the value of the parameter 'eline_alignment' ('build_xanes_map_api').")

    # Depending on the selected sequence, determine which steps must be executed
    seq_load_data = True
    seq_process_xrf_data = True
    seq_build_xrf_map_stack = True
    seq_generate_xanes_map = True
    if sequence == "load_and_process":
        pass
    elif sequence == "process":
        seq_load_data = False
    elif sequence == "build_xanes_map":
        seq_load_data = False
        seq_process_xrf_data = False
    else:
        raise ValueError(f"Unknown sequence name '{sequence}' is passed as a parameter "
                   "to the function 'build_xanes_map_api'.")

    # No XANES maps will be generated if references are not provided
    #                 (this is one of the built-in options, not an error)
    if not ref_file_name:
        seq_generate_xanes_map = False
        ref_energy = None
        ref_data = None
        ref_labels = None
    else:
        # Load reference file. (If there is a problem reading the reference file, it is better
        #   to learn it before the processing starts.)
        ref_energy, ref_data, ref_labels = read_ref_data(ref_file_name)

    # XRF data will be placed in the subdirectory 'xrf_data' of the directory 'wd'
    wd_xrf = os.path.join(wd, xrf_subdir)

    if seq_load_data:
        logger.info("Loading data from databroker ...")
        # Try to create the directory (does nothing if the directory exists)
        os.makedirs(wd_xrf, exist_ok=True)
        files_h5 = [fl.path for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
        if files_h5:
            logger.warning(f"The temporary directory '{wd_xrf}' is not empty. "
                           f"Deleting {len(files_h5)} files (.h5) ...")
            for fln in files_h5:
                logger.info(f"Removing raw xrf data file: '{fln}'.")
                os.remove(path=fln)
        make_hdf(start_id, end_id, wd=wd_xrf,
                 completed_scans_only=True, file_overwrite_existing=True,
                 create_each_det=False, save_scaler=True)
    else:
        logger.info("Loading of data from databroker is skipped ...")

    if seq_process_xrf_data:
        # Make sure that the directory with xrf data exists
        if not os.path.isdir(wd_xrf):
            # Unfortunately there is no way to continue if there is no directory with data
            raise IOError(f"XRF data directory '{wd_xrf}' does not exist.")

        # Load scan metadata (only ids, energies and file names are extracted)
        scan_ids, scan_energies, _, files_h5 = \
            _load_dataset_from_hdf5(start_id=start_id, end_id=end_id,
                                    wd_xrf=wd_xrf, load_fit_results=False)

        # Sort the lists based on incident beam energy. (The loaded data is sorted in the
        #   alphabetical order of file names, which may not match the acending order or
        #   incident energy values.
        scan_energies, sorted_indexes = list(zip(*sorted(zip(scan_energies, range(len(scan_energies))))))
        scan_energies = list(scan_energies)
        files_h5 = [files_h5[n] for n in sorted_indexes]
        scan_ids = [scan_ids[n] for n in sorted_indexes]

        scan_energies_adjusted = scan_energies.copy()

        print(f"scan_ids: {scan_ids}")
        print(f"scan_energies: {scan_energies}")
        print(f"files_h5: {files_h5}")

        ignore_metadata = False
        if incident_energy_low_bound is not None:
            for n, v in enumerate(scan_energies_adjusted):
                if v < incident_energy_low_bound:
                    scan_energies_adjusted[n] = incident_energy_low_bound
        elif use_incident_energy_from_param_file:
            # If 'pyxrf_batch' is called with 'incident_energy' set to None, and
            #   'ignore_datafile_metadata' is True, then
            #   the value of the incident energy from the parameter file is used
            scan_energies_adjusted = [None] * len(scan_energies)
            ignore_metadata = True
        else:
            scan_energies_adjusted = adjust_incident_beam_energies(scan_energies, emission_line)

        print(f"scan_energies_adjusted={scan_energies_adjusted}")

        # Process data files from the list. Use adjusted energy value.
        for fln, energy in zip(files_h5, scan_energies_adjusted):
            # Process .h5 files in the directory 'wd_xrf'. Processing results are saved
            #   as additional datasets in the original .h5 files.
            pyxrf_batch(data_files=fln,  # Process only one data file
                        param_file_name=param_file_name,
                        ignore_datafile_metadata=ignore_metadata,
                        incident_energy=energy,  # This value overrides incident energy from other sources
                        wd=wd_xrf, save_tiff=False)

        logger.info("Processing data files (computing XRF maps): success.")
    else:
        logger.info("Processing data files (computing XRF maps): skipped.")

    if seq_build_xrf_map_stack:
        logger.info("Building energy map ...")

        scan_ids, scan_energies, scan_img_dict, files_h5 = \
            _load_dataset_from_hdf5(start_id=start_id, end_id=end_id, wd_xrf=wd_xrf)

        # The following function checks dataset for consistency. If additional checks
        #   needs to be performed, they should be added to the implementation of this function.
        _check_dataset_consistency(scan_ids=scan_ids, scan_img_dict=scan_img_dict,
                                   files_h5=files_h5, scaler_name=scaler_name,
                                   eline_selected=eline_selected, eline_alignment=eline_alignment)

        logger.info("Checking dataset for consistency: success.")

        # Sort the lists based on energy. Prior to this point the data was arrange in the
        #   alphabetical order of files.
        scan_energies, sorted_indexes = list(zip(*sorted(zip(scan_energies, range(len(scan_energies))))))
        files_h5 = [files_h5[n] for n in sorted_indexes]
        scan_ids = [scan_ids[n] for n in sorted_indexes]
        scan_img_dict = [scan_img_dict[n] for n in sorted_indexes]

        logger.info("Sorting dataset: success.")

        # Create the lists of positional data for all scans
        positions_x_all = np.asarray([element['positions']['x_pos'] for element in scan_img_dict])
        positions_y_all = np.asarray([element['positions']['y_pos'] for element in scan_img_dict])

        # Find uniform grid that can be applied to the whole dataset (mostly for data plotting)
        positions_x_uniform, positions_y_uniform = _get_uniform_grid(positions_x_all,
                                                                     positions_y_all)
        logger.info("Generating common uniform grid: success.")

        # Create the arrays of XRF amplitudes for each emission line and normalize them
        eline_list, eline_data = _get_eline_data(scan_img_dict=scan_img_dict,
                                                 scaler_name=scaler_name)
        logger.info("Extracting XRF maps for emission lines: success.")

        if interpolation_enable:
            # Interpolate each image. I tried to use common uniform grid to do interpolation,
            #   but it didn't work very well. In the current implementation, the interpolation
            #   of each set is performed separately using the uniform grid specific for the set.
            for eline, data in eline_data.items():
                n_scans, _, _ = data.shape
                for n in range(n_scans):
                    data[n, :, :], _, _ = grid_interpolate(data[n, :, :],
                                                           xx=positions_x_all[n, :, :],
                                                           yy=positions_y_all[n, :, :])
            logger.info("Interpolating XRF maps to uniform grid: success.")
        else:
            logger.info("Interpolating XRF maps to uniform grid: skipped.")

        # Align the stack of images
        if alignment_enable:
            eline_data_aligned = _align_stacks(eline_data=eline_data,
                                               eline_alignment=eline_alignment,
                                               alignment_starts_from=alignment_starts_from)
            logger.info("Alignment of the image stack: success.")
        else:
            eline_data_aligned = eline_data
            logger.info("Alignment of the image stack: skipped.")

        if seq_generate_xanes_map:
            scan_absorption_refs = _interpolate_energy(scan_energies, ref_energy, ref_data)
            xanes_map_data, xanes_map_residual = _fit_xanes_map(eline_data_aligned[eline_selected], scan_absorption_refs)

            # Scale xanes maps so that the values represent counts
            n_refs, _, _ = xanes_map_data.shape
            xanes_map_data_counts = np.zeros(shape=xanes_map_data.shape)
            for n in range(n_refs):
                xanes_map_data_counts[n, :, :] = xanes_map_data[n, :, :] * np.sum(scan_absorption_refs[:, n])

            logger.info("XANES fitting: success.")
        else:
            scan_absorption_refs = None
            xanes_map_data = None
            xanes_map_data_counts = None
            xanes_map_residual = None
            logger.info("XANES fitting: skipped.")

        if "tiff" in output_file_formats:
            _save_xanes_maps_to_tiff(eline_data_aligned=eline_data_aligned,
                                     eline_selected=eline_selected,
                                     xanes_map_data=xanes_map_data_counts,
                                     xanes_map_labels=ref_labels,
                                     scan_energies=scan_energies,
                                     scan_ids=scan_ids)

        if plot_results:
            axes_units=plot_position_axes_units
            # If positions are none, then axes units are pixels
            pos_x, pos_y = (positions_x_uniform[0, :], positions_y_uniform[:, 0]) \
                if plot_use_position_coordinates else (None, None)

            if seq_generate_xanes_map:
                plot_absorption_references(ref_energy=ref_energy, ref_data=ref_data,
                                           scan_energies=scan_energies,
                                           scan_absorption_refs=scan_absorption_refs,
                                           ref_labels=ref_labels,
                                           block=False)

                figures = []
                for n, map_data in enumerate(xanes_map_data_counts):
                    fig = plot_xanes_map(map_data, label=ref_labels[n], block=False,
                                         positions_x=pos_x, positions_y=pos_y, axes_units=axes_units)
                    figures.append(fig)

                plot_xanes_map(xanes_map_residual, label="residual", block=False,
                               positions_x=pos_x, positions_y=pos_y, axes_units=axes_units)


            # Show image stacks for the selected elements
            show_image_stack(eline_data=eline_data_aligned, energies=scan_energies, eline_selected=eline_selected,
                             positions_x=pos_x, positions_y=pos_y, axes_units=axes_units,
                             xanes_map_data=xanes_map_data, absorption_refs=scan_absorption_refs,
                             ref_labels=ref_labels)
        else:
            logger.info("Plotting results: skipped.")

        logger.info("Processing is complete.")


def _load_dataset_from_hdf5(*, start_id, end_id, wd_xrf, load_fit_results=True):
    """
    Load dataset from processed HDF5 files

    Parameters
    ----------

    wd_xrf : str
        full (absolute) path name to the directory that contains processed HDF5 files
    load_fit_results : bool
        indicates if fit results should be loaded. If set to False, then only metadata
        is loaded and output dictionary ``scan_img_dict`` is empty.
    """

    # The list of file names
    files_h5 = [fl.name for fl in os.scandir(path=wd_xrf) if fl.name.lower().endswith(".h5")]
    # Sorting file names will make loading a little more orderly, but generally file names
    #   can be arbitrary (scan ID is not extracted from file names)
    files_h5.sort()

    scan_ids = []
    scan_energies = []
    scan_img_dict = []
    for fln in files_h5:
        img_dict, _, mdata = \
            read_hdf_APS(working_directory=wd_xrf, file_name=fln, load_summed_data=load_fit_results,
                         load_each_channel=False, load_processed_each_channel=False,
                         load_raw_data=False, load_fit_results=load_fit_results,
                         load_roi_results=False)

        if "scan_id" not in mdata:
            logger.error(f"Metadata value 'scan_id' is missing in data file '{fln}': "
                         " the file was not loaded.")
            continue

        # Make sure that the scan ID is in the specified range (if the range is
        #   not specified, then process all data files
        if (start_id is not None) and (mdata["scan_id"] < start_id) or \
                (end_id is not None) and (mdata["scan_id"] > end_id):
            continue

        if "instrument_mono_incident_energy" not in mdata:
            logger.error("Metadata value 'instrument_mono_incident_energy' is missing "
                         f"in data file '{fln}': the file was not loaded.")

        if load_fit_results:
            scan_img_dict.append(img_dict)
        scan_ids.append(mdata["scan_id"])
        scan_energies.append(mdata["instrument_mono_incident_energy"])

    return scan_ids, scan_energies, scan_img_dict, files_h5


def _check_dataset_consistency(*, scan_ids, scan_img_dict, files_h5, scaler_name,
                               eline_selected, eline_alignment):

    if not scan_img_dict:
        raise RuntimeError("Loaded dataset is empty. No data to process.")

    # First check if processing dataset exists. The following call will raise the exception
    #   if there is not dataset with processed data.
    _get_dataset_name(scan_img_dict[0])

    # Create a list of 'img_dict' keys
    img_data_keys = []
    img_data_keys_union = set()
    for img in scan_img_dict:
        ks = _get_img_keys(img)
        img_data_keys.append(ks)
        img_data_keys_union = img_data_keys_union.union(ks)
    img_data_keys_union = list(img_data_keys_union)

    def _raise_error_exception(slist, data_tuples, msg_phrase):
        """
        Report error by raising the RuntimeError exception. Print the list
        of scan_ids and file names for the files that have the problem

        Parameters
        ----------

        slist : list(int)
            the list of indices of the scans

        scan_ids : list(int)
            the list of scan IDs

        files_h5 : list(str)
            the list of file names

        msg_phrase : str
            the string, representing varying part of the message
        """
        msg = f"Some scans in the dataset {msg_phrase}:\n"
        for n in slist:
            for dt in data_tuples:
                msg += f"    {dt[1]}:  {dt[0][n]}"
            msg += "\n"
        raise RuntimeError(msg)

    def _check_for_specific_elines(img_data_keys, elines, files_h5, scan_ids, msg_phrase):
        """
        Check if all emission lines from the list are present in all scan data
        """
        slist = []
        for n, ks in enumerate(img_data_keys):
            all_lines_present = True
            for e in elines:
                if e not in ks:
                    all_lines_present = False
                    break
            if not all_lines_present:
                slist.append(n)

        if slist:
            _raise_error_exception(slist=slist,
                                   data_tuples=[(scan_ids, "scan ID"),
                                                (files_h5, "file")],
                                   msg_phrase=msg_phrase)

    def _check_for_identical_eline_key_set(img_data_keys, img_data_keys_union):
        """
        Check if all processed datasets contain data for the same emission lines.
        If not, then processing has to be run again on the datasets
        """
        success = True
        all_eline_keys = _get_eline_keys(img_data_keys_union)
        all_eline_keys.sort()
        for n, ks in enumerate(img_data_keys):
            eks = _get_eline_keys(ks)
            eks.sort()
            if eks != all_eline_keys:
                success = False
                break

        if not success:
            msg = ("Files in the dataset were processed for different sets of emission lines:\n"
                   "    may be fixed by rerunning the processing of the dataset "
                   "with the same parameter file.")
            raise RuntimeError(msg)

    def _check_for_positional_data(scan_img_dict):
        slist = []
        for n, img in enumerate(scan_img_dict):
            if ("positions" not in img) or ("x_pos" not in img["positions"]) or \
                    ("y_pos" not in img["positions"]):
                slist.append(n)

        if slist:
            _raise_error_exception(slist=slist,
                                   data_tuples=[(scan_ids, "scan ID"),
                                                (files_h5, "file")],
                                   msg_phrase="have no positional data ('x_pos' or 'y_pos')")

    def _check_for_identical_image_size(scan_img_dict, files_h5, scan_ids):
        # Create the list of image sizes for all files
        xy_list = []
        for img in scan_img_dict:
            xy_list.append(img["positions"]["x_pos"].shape)

        # Determine if all sizes are identical
        if [_ for _ in xy_list if _ != xy_list[0]]:
            _raise_error_exception(slist=list(range(len(xy_list))),
                                   data_tuples=[(scan_ids, "scan_ID"),
                                                (files_h5, "file"),
                                                (xy_list, "image size")],
                                   msg_phrase="contain XRF maps of different size"
                                   )

    # Check existence of the scaler only if scaler name is specified
    if scaler_name:
        _check_for_specific_elines(img_data_keys=img_data_keys,
                                   elines=[scaler_name],
                                   files_h5=files_h5,
                                   scan_ids=scan_ids,
                                   msg_phrase=f"have no scaler data ('{scaler_name}')")

    _check_for_identical_eline_key_set(img_data_keys=img_data_keys,
                                       img_data_keys_union=img_data_keys_union)

    _check_for_specific_elines(img_data_keys=img_data_keys,
                               elines=[eline_selected, eline_alignment],
                               files_h5=files_h5,
                               scan_ids=scan_ids,
                               msg_phrase=f"have no emission line data ('{eline_selected}' "
                                          f"or '{eline_alignment}')")

    _check_for_positional_data(scan_img_dict=scan_img_dict)

    _check_for_identical_image_size(scan_img_dict=scan_img_dict,
                                    files_h5=files_h5,
                                    scan_ids=scan_ids)


# ============================================================================================
#   Functions for parameter manipulation

def adjust_incident_beam_energies(scan_energies, emission_line):
    """
    Adjust the values of incident beam energy in the list ``scan_energies`` so that
    fitting for the ``emission_line`` could be done at each energy value. Adjustment
    is done by finding the lowest incident energy value in the list that still activates
    the ``emission_line`` and setting every smaller energy value to the lowest value.

    Parameters
    ----------

    scan_energies : list(float)
        the list of incident beam energy values, keV
    emission_line : str
        valid name of the emission line (supported by ``scikit-beam``) in the
        form of K_K or Fe_K

    Returns
    -------
        the list of adjusted values of the incident energy. The returned list has the
        same dimensions as ``scan_energies``.
    """

    e_activation = [_ for _ in scan_energies if check_if_eline_is_activated(emission_line, _)]

    if not e_activation:
        raise RuntimeError(
            f"The emission line '{emission_line}' is not activated\n"
            f"    in the range of energies {min(scan_energies)} - {max(scan_energies)} keV.\n"
            f"    Check if the emission line is specified correctly.")

    min_activation_energy = min(e_activation)
    return [max(_, min_activation_energy) for _ in scan_energies]


# ============================================================================================
#   Functions used for processing

def _get_uniform_grid(positions_x_all, positions_y_all):
    """Compute uniform grid common to the whole dataset"""
    # Median positions are probably the best for generating common uniform grid
    positions_x_median = np.median(positions_x_all, axis=0)
    positions_y_median = np.median(positions_y_all, axis=0)
    # Generate uniform grid
    _, positions_x_uniform, positions_y_uniform = grid_interpolate(
        None, positions_x_median, positions_y_median)
    return positions_x_uniform, positions_y_uniform


def _get_eline_data(scan_img_dict, scaler_name):
    """
    Create the list of emission lines and the array with emission data.

    Array ``eline_data`` contains XRF maps for the emission lines:

    -- dimension 0 - the number of the emission line,
    the name of the emission line may be extracted from the respective
    entry of the list ``eline_list``

    -- dimensions 1 and 2 - Y and X coordinates of the map
    """
    eline_list = _get_eline_keys(_get_img_keys(scan_img_dict[0]))
    eline_data = {}
    for eline in eline_list:
        data = []
        for img_dict in scan_img_dict:
            d = _get_img_data(img_dict=img_dict, key=eline)
            if scaler_name:  # Normalization
                d = normalize_data_by_scaler(d, _get_img_data(img_dict=img_dict,
                                                              key=scaler_name))
            data.append(d)
        eline_data[eline] = np.asarray(data)
    return eline_list, eline_data


def _align_stacks(eline_data, eline_alignment, alignment_starts_from="top"):
    """
    Align stack of images
    """
    alignment_starts_from = alignment_starts_from.lower()
    assert alignment_starts_from in ["top", "bottom"], \
        f"Parameter 'alignment_starts_from' has invalid value: '{alignment_starts_from}'"

    def _flip_stack(img_stack, alignment_starts_from):
        """Flip image stack if alignment should be started from the top image"""
        if alignment_starts_from == "top":
            return np.flip(img_stack, 0)
        else:
            return img_stack

    """Align stack of XRF maps for each element"""
    sr = StackReg(StackReg.TRANSLATION)
    sr.register_stack(_flip_stack(eline_data[eline_alignment], alignment_starts_from),
                      reference="previous")

    eline_data_aligned = {}
    for eline, data in eline_data.items():
        data_prepared = _flip_stack(data, alignment_starts_from)
        data_transformed = sr.transform_stack(data_prepared)
        data_transformed = _flip_stack(data_transformed, alignment_starts_from)
        eline_data_aligned[eline] = data_transformed.clip(min=0)
    return eline_data_aligned


def _interpolate_energy(energy, energy_refs, absorption_refs):
    _, n_states = absorption_refs.shape
    interpolated_refs = np.zeros(shape=[len(energy), n_states], dtype=float)
    for n in range(n_states):
        a_ref = absorption_refs[:, n]
        interpolated_refs[:, n] = np.interp(energy, energy_refs, a_ref)
    return interpolated_refs


def _fit_xanes_map(map_data, absorption_refs):
    _, nny, nnx = map_data.shape
    _, n_states = absorption_refs.shape
    map_data_fitted = np.zeros(shape=[n_states, nny, nnx])
    map_residual = np.zeros(shape=[nny, nnx])
    for ny in range(nny):
        for nx in range(nnx):
            result, residual = nnls_fit(map_data[:, ny, nx], absorption_refs, weights=None)

            # Residual computed by 'nnls' function is a scaled version of the square norm:
            #      sqrt(sqrt(N-1)) * sum((Ax-b)^2)
            # If there is a need for not scaled square norm, then use the following code:
            #      residual = residual / np.sqrt(np.sqrt(n_data_points - 1))

            map_data_fitted[:, ny, nx] = result
            map_residual[ny, nx] = residual
    return map_data_fitted, map_residual


# ==============================================================================================
#     Functions for plotting the results

def show_image_stack(*, eline_data, energies, eline_selected,
                     positions_x=None, positions_y=None, axes_units=None,
                     xanes_map_data=None, absorption_refs=None, ref_labels=None):
    """
    Display XRF Map stack


    Parameters
    ----------

    positions_x : ndarray
        values of coordinate X of the image. Must match the size of dimension 1 of the image

    positions_y : ndarray
        values of coordinate Y of the image. Must match the size of dimension 0 of the image
    """

    if not eline_data:
        logger.warning("Emission line data dictionary is empty. There is nothing to plot.")
        return

    class EnergyMapPlot:

        def __init__(self, *, energy, stack_all_data, label_default,
                     positions_x=None, positions_y=None, axes_units=None,
                     xanes_map_data=None, absorption_refs=None, ref_labels=None):
            """
            Parameters
            ----------

            energy : ndarray, 1-D
                values of energy, the number of elements must match the number of images in the stacks

            stack_all_data : dict of 3-D ndarrays
                dictionary that contains stacks of images: key - the label of the stack,
                value - 3D ndarray with dimension 0 representing the image number and dimensions
                1 and 2 representing positions along Y and X axes.

            label_default : str
                default label which is used to select the image stack which is selected at the start.
                If the value does not match any keys of ``stack_all_data``, then it is ignored

            positions_x : ndarray
                values of coordinate X of the image. Only the first and the last value of the list
                is used to set the axis range, so [xmin, xmax] is sufficient.

            positions_y : ndarray
                values of coordinate Y of the image. Only the first and the last value of the list
                is used to set the axis range, so [ymin, ymax] is sufficient.

            axes_units : str
                units for X and Y axes that are used if ``positions_x`` and ``positions_y`` are specified
                Units may include latex expressions, for example "$\mu $m" will print units of microns.
                If ``axes_units`` is None, then the name of arbitrary units (``a.u.``) is used.

            xanes_map_data : 3D ndarray

            absorption_refs : 2D ndarray
            """

            self.label_fontsize = 15

            self.stack_all_data = stack_all_data
            self.energy = energy
            self.label_default = label_default

            self.labels = list(eline_data.keys())

            self.textbox_nlabel = None
            self.busy = False

            self.xanes_map_data = xanes_map_data
            self.absorption_refs = absorption_refs
            self.ref_labels = ref_labels

            if self.label_default not in self.labels:
                logger.warning(f"XRF Energy Plot: the default label {self.label_default} "
                               "is not in the list and will be ignored")
                self.label_default = self.labels[0]  # Set the first emission line as default
            self.label_selected = self.label_default
            self.select_stack()

            n_images, ny, nx = self.stack_selected.shape
            self.n_energy_selected = int(n_images/2)  # Select image in the middle of the stack
            self.img_selected = self.stack_selected[self.n_energy_selected, :, :]

            # Select point in the middle of the plot
            self.pt_selected = [int(nx / 2), int(ny / 2)]  # The coordinates of the point are in pixels

            # Check existence or the size of 'positions_x' and 'positions_y' arrays
            ny, nx = self.img_selected.shape
            if (positions_x is None) or (positions_y is None):
                self.pos_x_min, self.pos_x_max = 0, nx - 1
                self.pos_y_min, self.pos_y_max = 0, ny - 1
                self.pos_dx, self.pos_dy = 1, 1
                self.axes_units = "pixels"
            else:
                self.pos_x_min, self.pos_x_max = positions_x[0], positions_x[-1]
                self.pos_y_min, self.pos_y_max = positions_y[0], positions_y[-1]
                self.pos_dx = (self.pos_x_max - self.pos_x_min)/max(nx - 1, 1)
                self.pos_dy = (self.pos_y_max - self.pos_y_min)/max(ny - 1, 1)
                self.axes_units = axes_units if axes_units else "a.u."

        def select_stack(self):
            self.stack_selected = self.stack_all_data[self.label_selected]
            if not isinstance(self.stack_selected, np.ndarray) or not self.stack_selected.ndim == 3:
                raise ValueError("Image stack must be 3-D numpy array.")

        def set_cbar_range(self):
            v_min = np.min(self.stack_selected)
            v_max = np.max(self.stack_selected)
            self.img_plot.set_clim(v_min, v_max)

        def show(self):

            self.textbox_nlabel = None

            self.fig = plt.figure(figsize=(11, 6), num="XRF MAPS")
            self.ax_img_stack = plt.axes([0.07, 0.25, 0.35, 0.65])
            self.ax_img_cbar = plt.axes([0.425, 0.26, 0.013, 0.63])
            self.ax_fluor_plot = plt.axes([0.55, 0.25, 0.4, 0.65])
            self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.25)
            self.ax_img_stack.set_xlabel(f"X, {self.axes_units}", fontsize=self.label_fontsize)
            self.ax_img_stack.set_ylabel(f"Y, {self.axes_units}", fontsize=self.label_fontsize)

            # display image
            extent = [self.pos_x_min, self.pos_x_max, self.pos_y_max, self.pos_y_min]
            self.img_plot = self.ax_img_stack.imshow(self.img_selected,
                                                     origin="upper",
                                                     extent=extent)
            self.img_label_text = self.ax_img_stack.text(
                0.5, 1.01, self.label_selected,
                ha='left', va='bottom',
                fontsize=self.label_fontsize,
                transform=self.ax_img_stack.axes.transAxes)
            self.cbar = self.fig.colorbar(self.img_plot, cax=self.ax_img_cbar, orientation="vertical")
            self.ax_img_cbar.ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')
            self.set_cbar_range()

            self.redraw_fluorescence_plot()

            # define slider
            axcolor = 'lightgoldenrodyellow'
            self.ax_slider_energy = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
            self.set_slider_energy_title()

            self.slider = Slider(self.ax_slider_energy, 'Energy',
                                 0, len(self.energy) - 1,
                                 valinit=self.n_energy_selected, valfmt='%i')

            if len(self.labels) <= 10:
                # Individual button per label (emission line). Only 10 buttons will fit windows
                self.create_buttons_each_label()
            else:
                # Alternative way to organize switching between labels
                self.create_buttons_prev_next()

            self.slider.on_changed(self.slider_update)

            self.fig.canvas.mpl_connect("button_press_event", self.canvas_onclick)

            plt.show()

        def show_fluor_point_coordinates(self):
            pt_x = self.pt_selected[0] *self.pos_dx + self.pos_x_min
            pt_y = self.pt_selected[1] *self.pos_dy + self.pos_y_min
            pt_x_str, pt_y_str = f"{pt_x:.5g}", f"{pt_y:.5g}"
            self.fluor_label_text = self.ax_fluor_plot.text(
                0.99, 0.99, f"({pt_x_str}, {pt_y_str})",
                ha='right', va='top', fontsize=self.label_fontsize,
                transform=self.ax_fluor_plot.axes.transAxes)

        def create_buttons_each_label(self):
            """
            Create separate button for each label. The maximum number of labels
            that can be assigned unique buttons is 10. More buttons may
            not fit the screen
            """
            n_labels = len(self.labels)

            max_labels = 10
            assert n_labels <= max_labels, \
                f"The stack contains too many labels ({len(self.labels)}). "\
                f"Maximum allowed number of labels is {max_labels}"

            bwidth = 0.07  # Width of a button
            bgap = 0.01  # Gap between buttons

            pos_left_start = 0.5 - n_labels / 2 * bwidth - (n_labels - 1) / 2 *bgap

            # Create buttons
            self.btn_eline, self.ax_btn_eline = [], []
            for n in range(n_labels):
                p_left = pos_left_start + (bwidth + bgap) * n
                self.ax_btn_eline.append(plt.axes([p_left, 0.03, bwidth, 0.04]))
                c = "#ffff00" if self.labels[n] == self.label_default else "#00ff00"
                self.btn_eline.append(Button(self.ax_btn_eline[-1], self.labels[n],
                                             color=c, hovercolor="#ff0000"))

            # Set events to each button
            for b in self.btn_eline:
                b.on_clicked(self.btn_stack_clicked)

        def create_buttons_prev_next(self):

            n_labels = len(self.labels)

            bwidth = 0.07  # Width of prev. and next button button
            lwidth = 0.08  # Width of the text label between prev and next buttons
            bgap = 0.01  # Gap between buttons

            self.ax_btn_prev = plt.axes([0.5 - lwidth/2 - bwidth - bgap, 0.03, bwidth, 0.04])
            self.btn_prev = Button(self.ax_btn_prev, "Previous", color="#00ff00", hovercolor="#ff0000")
            self.btn_prev.on_clicked(self.btn_stack_prev_clicked)

            self.ax_text_nlabel = plt.axes([0.5 - lwidth/2, 0.03, lwidth, 0.04])
            self.textbox_nlabel = Button(self.ax_text_nlabel, "",
                                         color="lightgray", hovercolor="lightgray")
            self.display_btn_stack_label()

            self.ax_btn_next = plt.axes([0.5 + lwidth/2 + bgap, 0.03, bwidth, 0.04])
            self.btn_next = Button(self.ax_btn_next, "Next", color="#00ff00", hovercolor="#ff0000")
            self.btn_next.on_clicked(self.btn_stack_next_clicked)

        def display_btn_stack_label(self):
            if self.textbox_nlabel:
                self.textbox_nlabel.label.set_text(
                    f"{self.labels.index(self.label_selected) + 1} ({len(self.labels)})")

        def set_slider_energy_title(self):
            n = self.n_energy_selected
            n_total = len(self.energy)
            e = self.energy[n]
            self.ax_slider_energy.set_title(f"{e:.5f} keV (frame #{n + 1} of {n_total})")

        def btn_stack_clicked(self, event):
            """Callback"""
            for n, ab in enumerate(self.ax_btn_eline):
                if event.inaxes is ab:
                    self.switch_to_different_stack(self.labels[n])
                    break

        def btn_stack_next_clicked(self, event):
            """Callback"""
            n_labels = len(self.labels)
            n_current = self.labels.index(self.label_selected)
            if n_current < n_labels - 1:
                self.switch_to_different_stack(self.labels[n_current + 1])

        def btn_stack_prev_clicked(self, event):
            """Callback"""
            n_current = self.labels.index(self.label_selected)
            if n_current > 0:
                self.switch_to_different_stack(self.labels[n_current - 1])

        def switch_to_different_stack(self, label):
            self.label_selected = label
            self.select_stack()
            self.redraw_image()
            self.set_cbar_range()
            self.display_btn_stack_label()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def redraw_fluorescence_plot(self):
            self.ax_fluor_plot.clear()

            self.ax_fluor_plot.plot(self.energy,
                                    self.stack_selected[:, self.pt_selected[0], self.pt_selected[1]],
                                    marker=".", linestyle="solid", label="XANES spectrum")

            # Plot the results of fitting (if the fitting was performed
            if (self.label_selected == self.label_default) and (self.xanes_map_data is not None) \
                    and (self.absorption_refs is not None):

                # Labels should always be supplied when calling the function.
                #   This is not user controlled option, therefore exception should be raised.
                assert self.labels, "No labels are provided. Fitting results can not be displayed properly"

                refs_scaled = self.absorption_refs.copy()
                xanes_fit_pt = self.xanes_map_data[:, self.pt_selected[0], self.pt_selected[1]]
                _, n_refs = refs_scaled.shape
                for n in range(n_refs):
                    refs_scaled[:, n] = refs_scaled[:, n] * xanes_fit_pt[n]
                refs_scaled_sum = np.sum(refs_scaled, axis=1)

                self.ax_fluor_plot.plot(self.energy, refs_scaled_sum, label="XANES fit")
                for n in range(n_refs):
                    self.ax_fluor_plot.plot(self.energy, refs_scaled[:, n], label=self.ref_labels[n])

            # Always display the legend
            self.ax_fluor_plot.legend(loc="upper left")

            self.ax_fluor_plot.grid()
            self.ax_fluor_plot.set_xlabel("Energy, keV", fontsize=self.label_fontsize)
            self.ax_fluor_plot.set_ylabel("Fluorescence", fontsize=self.label_fontsize)
            self.ax_fluor_plot.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
            self.show_fluor_point_coordinates()

        def redraw_image(self):
            self.img_selected = self.stack_selected[self.n_energy_selected, :, :]
            self.img_plot.set_data(self.img_selected)
            self.img_label_text.set_text(self.label_selected)
            self.redraw_fluorescence_plot()

        def slider_update(self, val):
            if not self.busy:
                self.busy = True
                n_slider = int(round(val))
                if n_slider != self.n_energy_selected:
                    self.n_energy_selected = n_slider
                    self.set_slider_energy_title()
                    self.redraw_image()
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                self.busy = False

        def canvas_onclick(self, event):
            """Callback"""
            if (event.inaxes == self.ax_img_stack) and (event.button == 1):
                xd, yd = event.xdata, event.ydata
                # Compute pixel coordinates
                xd_px = round((xd - self.pos_x_min)/self.pos_dx)
                yd_px = round((yd - self.pos_y_min)/self.pos_dy)
                self.pt_selected = [int(xd_px), int(yd_px)]
                self.redraw_fluorescence_plot()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    map_plot = EnergyMapPlot(energy=energies, stack_all_data=eline_data, label_default=eline_selected,
                             positions_x=positions_x, positions_y=positions_y, axes_units=axes_units,
                             xanes_map_data=xanes_map_data, absorption_refs=absorption_refs,
                             ref_labels=ref_labels)
    map_plot.show()


def plot_xanes_map(map_data, *, label=None, block=True,
                   positions_x=None, positions_y=None, axes_units=None):
    """
    Plot XANES map

    Parameters
    ----------

    map_data : ndarray, 2D
        map to be plotted, dimensions: (Ny, Nx)

    label : str
        reference label that is included in the image title and window title

    block : bool
        the parameter is passed to ``plt.show()``. Indicates if the execution of the
        program should be paused until the figure is closed.

    positions_x : ndarray
        values of coordinate X of the image. Only the first and the last value of the list
        is used to set the axis range, so [xmin, xmax] is sufficient.

    positions_y : ndarray
        values of coordinate Y of the image. Only the first and the last value of the list
        is used to set the axis range, so [ymin, ymax] is sufficient.

    axes_units : str
        units for X and Y axes that are used if ``positions_x`` and ``positions_y`` are specified
        Units may include latex expressions, for example "$\mu $m" will print units of microns.
        If ``axes_units`` is None, then the name of arbitrary units (``a.u.``) is used.

    Returns
    -------
        Reference to the figure containing the plot

    """

    # Check existence or the size of 'positions_x' and 'positions_y' arrays
    ny, nx = map_data.shape
    if (positions_x is None) or (positions_y is None):
        positions_x = range(nx)
        positions_y = range(ny)
        axes_units = "pixels"
    else:
        axes_units = axes_units if axes_units else "a.u."

    if label:
        fig_title = f"XANES MAP: {label}"
        img_title = f"XANES: {label}"

    fig = plt.figure(figsize=(6, 6), num=fig_title)

    # display image
    extent = [positions_x[0], positions_x[-1],
              positions_y[-1], positions_y[0]]
    img_plot = plt.imshow(map_data, origin="upper", extent=extent)
    plt.colorbar(img_plot, orientation="vertical")
    plt.axes().set_xlabel(f"X, {axes_units}", fontsize=15)
    plt.axes().set_ylabel(f"Y, {axes_units}", fontsize=15)
    fig.suptitle(img_title, fontsize=20)
    plt.show(block=block)
    return fig


def plot_absorption_references(*, ref_energy, ref_data, scan_energies,
                               scan_absorption_refs, ref_labels=None, block=True):
    """
    Plots absorption references

    Parameters
    ----------
    ref_energy : ndarray, 1D
        N elements, energy values for the original references (loaded from file)
    ref_data : ndarray, 2D
        NxK elements, K - the number of references
    scan_energies : ndarray, 1D
        M elements, energy values for resampled references, M - the number of scanned images
    scan_absorption_refs : ndarray, 2D
        MxK elements
    ref_labels : list(str)
        K elements, the labels for references. If None or empty, then no labels will be
        printed
    block : bool
        the parameter is passed to ``plt.show()``. Indicates if the execution of the
        program should be paused until the figure is closed.

    Returns
    -------
        Reference to the figure containing the plot
    """

    # Check if the number of original and sampled references match
    _, n_refs = ref_data.shape
    _, n2 = scan_absorption_refs.shape
    assert n_refs == n2, "The number of original and sample references does not match"
    if ref_labels:
        assert n_refs == len(ref_labels), "The number of references and labels does not match"

    if ref_labels:
        labels = ref_labels
    else:
        labels = [""] * n_refs

    fig = plt.figure(figsize=(6, 6), num="Absorption References")

    for n in range(n_refs):
        plt.plot(ref_energy, ref_data[:, n], label=labels[n])

    for n in range(n_refs):
        plt.plot(scan_energies, scan_absorption_refs[:, n], "o", label=labels[n])

    plt.axes().set_xlabel("Energy, keV", fontsize=15)
    plt.axes().set_ylabel("Absorption", fontsize=15)
    plt.grid(True)
    fig.suptitle("Absorption References", fontsize=20)
    if ref_labels:
        plt.legend(loc="upper right")
    plt.show(block=block)
    return fig


def _get_img_data(img_dict, key, detector=None):
    """
    Retrieves entry of ``img_dict``. The difficulty is that the dataset key (first key)
    contains file name so it is variable, so at the first step the appropriate key has to
    be found. Then the data for the element line or scaler (specified by ``key``)
    may be retrieved.

    Parameters
    ----------

    img_dict : dict
        The dictionary of data extracted from HDF5 file. Processed data for element lines
        may be accessed as ``img_dict[dataset_name][key]

    key : str
        The string that specifies element line or a scaler

    detector : int
        detector channel number: 1, 2, 3 etc. Set to None if the sum of all channels
        data should be used.
    """
    return img_dict[_get_dataset_name(img_dict, detector=detector)][key]


def _get_img_keys(img_dict, detector=None):
    """
    Returns list of keys in the dataset. See docstring for ``_get_img_data`` for more information.
    """
    return list(img_dict[_get_dataset_name(img_dict, detector=detector)].keys())


def _get_eline_keys(key_list):
    """
    Returns list of emission line keys that are present in the list of keys.
    """

    # The following pattern is based on assumption that elines names
    #        are of the form Si_K, La_L, P_K etc.
    re_pattern = "^[A-Z][a-z]?_[KLM]$"
    return [key for key in key_list if re.search(re_pattern, key)]


def _get_dataset_name(img_dict, detector=None):
    """
    Finds the name of dataset with fitted data in the ``img_dict``.
    Dataset name contains file name in it, so it is changing from file to file.
    The dataset ends with suffix '_fit'. Datasets for individual detector
    channels end with 'det1_fit', 'det2_fit' etc.

    Parameters
    ----------

    img_dict : dict
        dictionary of xrf image data loaded from .h5 file
    detector : int
        detector channel number: 1, 2, 3 etc. Set to None if the sum of all channels
        data should be used.

    Returns
    -------
    dataset name, raises RuntimeError exception if dataset is not found
    """
    if detector is None:
        # Dataset name for the sum should not have 'det'+number preceding '_fit'
        #   Assume that the number of digits does not exceed 3 (in practice it
        #   doesn't exceed 1)
        patterns = ["(?<!det\d)fit$", "(?<!det\d\d)fit$", "(?<!det\d\d\d)fit$"]  # noqa W605
    else:
        patterns = [f"det{detector}_fit"]
    for name in img_dict.keys():
        name_found = True
        for p in patterns:
            #  All patterns must show a match
            if not re.search(p, name):
                name_found = False
        if name_found:
            return name

    raise RuntimeError(f"No dataset name was found for the detector {detector} ('get_dataset_name').")


def read_ref_data(ref_file_name):
    """Read file with reference data (for XANES fitting)"""

    ref_labels=[]

    if not os.path.isfile(ref_file_name):
        raise ValueError(f"The parameter file '{ref_file_name}' does not exist. Check the value of"
                         f" the parameer 'ref_file_name' ('build_xanes_map_api').")

    with open(ref_file_name, "r") as f:
        line_first = f.readline()
        lb = line_first.split()
        # Strip spaces
        ref_labels = [_.strip() for _ in lb]
        # Remove the first label (which is probably 'Energy', but unused in processing)
        ref_labels.pop(0)

    data_ref_file = np.genfromtxt(ref_file_name, skip_header=1)
    ref_energy = data_ref_file[:, 0]
    # The references are columns 'ref_data'
    ref_data = data_ref_file[:, 1:]

    _, n_ref_data_columns = ref_data.shape

    if n_ref_data_columns != len(ref_labels):
        raise Exception(f"Reference data file '{ref_file_name}' has unequal number of labels and data columns")

    # Scale the energy in the reference file (given in eV, but the rest of the program is uing keV)
    ref_energy /= 1000.0
    # Data may contain some small negative values, so clip them to 0
    ref_data = ref_data.clip(min=0)

    return ref_energy, ref_data, ref_labels

def _save_xanes_maps_to_tiff(*, eline_data_aligned, eline_selected,
                             xanes_map_data, xanes_map_labels,
                             scan_energies, scan_ids):

    """
    Saves the results of processing in stacked .tiff files and creates .txt log file
    with the list of contents of .tiff files.
    It is assumed that the input data is consistent.

    If ``xanes_map_data`` or ``xanes_map_labels`` are None, then XANES maps are not
    saved. XANES maps are not generated if file with references is not available.
    This is one of the legitimate modes of operation.

    Parameters
    ----------
    eline_data_aligned : dict(ndarray)
        The dictionary that contains aligned datasets. Key: emission line,
        value: ndarray [K, Ny, Nx], K - the number of scans, Ny and Nx - the
        number of pixels along y- and x- axes.
    eline_selected : str
        The name of the selected emission line. If the emission line is not present
        in the dictionary ``eline_data_aligned``, then the image stack is not saved.
    xanes_map_data : ndarray [M, Ny, Nx]
        XANES maps. M - the number of maps
    xanes_map_labels : list(str)
        Labels for XANES maps. The number of labels must be equal to M.
    scan_energies : list(float)
        Beam energy values for the scans. The number of values must be K.
    scan_ids : list(int)
        Scan IDs of the scans. There must be K scan IDs.
\           """

    if eline_selected is None:
        eline_selected = ""

    # A .txt file is created along with saving the rest of the data.
    fln_log = f"maps_{eline_selected}.txt"
    with open(fln_log, "w") as f_log:

        print(f"Processing completed at {convert_time_to_nexus_string(ttime.localtime())}", file=f_log)

        if eline_data_aligned and eline_selected and (eline_selected in eline_data_aligned):
            # Save the stack of XRF maps for the selected emission line
            fln_stack = f"xrf_maps_{eline_selected}.tiff"
            tifffile.imsave(fln_stack, eline_data_aligned[eline_selected].astype(np.float32),
                       imagej=True)
            logger.info(f"The stack of XRF maps for the emission line {eline_selected} is saved "
                        f"to file '{fln_stack}'")

            # Save the contents of the .tiff file to .txt file
            print(f"\nThe stack of XRF maps is saved to file '{fln_stack}'.", file=f_log)
            print("Contents:", file=f_log)
            if scan_energies and scan_ids:
                for n, energy, scan_id in zip(range(len(scan_energies)), scan_energies, scan_ids):
                    print(f"   Frame {n + 1}:  scan ID = {scan_id}  beam energy = {energy}",
                          file=f_log)

        if (xanes_map_data is not None) and xanes_map_labels and eline_selected:
            # Save XANES maps for references
            fln_xanes = f"xanes_maps_{eline_selected}.tiff"
            tifffile.imsave(fln_xanes, xanes_map_data.astype(np.float32), imagej=True)
            logger.info(f"XANES maps for the emission line {eline_selected} are saved "
                        f"to file '{fln_xanes}'")

            # Save the contents of the .tiff file to .txt file
            print(f"\nXANES maps are saved to file '{fln_xanes}'.", file=f_log)
            print("Contents:", file=f_log)
            if xanes_map_labels:
                for n, label in enumerate(xanes_map_labels):
                    print(f"   Frame {n + 1}:  reference = '{xanes_map_labels[n]}'", file=f_log)


if __name__ == "__main__":

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    build_xanes_map_api(start_id=92276, end_id=92335,
                        param_file_name="param_335",
                        scaler_name="sclr1_ch4", wd=None,
                        # sequence="process",
                        sequence="build_xanes_map",
                        alignment_starts_from="top",
                        ref_file_name="refs_Fe_P23.txt",
                        emission_line="Fe_K", emission_line_alignment="P_K",
                        interpolation_enable=True,
                        alignment_enable=True,
                        plot_use_position_coordinates=True,
                        plot_results=True)
