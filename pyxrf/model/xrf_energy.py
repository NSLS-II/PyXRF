import os
import re
import numpy as np
from pystackreg import StackReg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .load_data_from_db import make_hdf
from .command_tools import pyxrf_batch
from .fileio import read_hdf_APS
from .utils import grid_interpolate, normalize_data_by_scaler

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
                         scaler_name=None,
                         wd=None,
                         xrf_subdir="xrf_data",
                         sequence="build_energy_map",
                         emission_line=None,
                         emission_line_references=None,
                         emission_line_alignment=None):

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
        If the "process" or "build_energy_map" sequence is executed, then the program
        looks for raw or processed files in this subfolder. In majority of cases, the
        default value is sufficient.

    sequence : str
        the sequence of operations performed by the function:

        -- ``load_and_process`` - loading data and full processing,

        -- ``process`` - full processing of data, including xrf mapping
        and building the energy map,

        -- ``build_energy_map`` - build the energy map using xrf mapping data,
        all data must be processed.

    emission_line : str
        the name of the selected emission line ("Ca_K", "Fe_K", etc.). The emission line
        of interest.

    emission_line_references : str
        file name with emission line references.

    emission_line : str
        the name of the emission line used for image alignment ("Ca_K", "Fe_K", etc.).
        If None, then the line specified as ``emission_line`` used for alignment

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

    if not xrf_subdir:
        raise ValueError("The parameter 'xrf_subdir' is None or contains an empty string "
                         "('build_energy_map_api'.")

    if not scaler_name:
        logger.warning("Scaler was not specified. The processing will still be performed,"
                       "but the DATA WILL NOT BE NORMALIZED!")

    # Set emission lines
    eline_selected = emission_line
    if emission_line_alignment:
        eline_alignment = emission_line_alignment
    else:
        eline_alignment = eline_selected

    # TODO: check if 'eline_selected' and 'eline_alignment' are valid emission lines
    #   (use the lists of K, L and M lines from scikit-beam)

    # Depending on the selected sequence, determine which steps must be executed
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
        # Process .h5 files in the directory 'wd_xrf'. Processing results are saved
        #   as additional datasets in the original .h5 files.
        pyxrf_batch(start_id=start_id, end_id=end_id,
                    param_file_name=param_file_name,
                    wd=wd_xrf, save_tiff=False)

    if seq_build_energy_map:
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
        def _get_uniform_grid():
            """Compute uniform grid common to the whole dataset"""
            # Median positions are probably the best for generating common uniform grid
            positions_x_median = np.median(positions_x_all, axis=0)
            positions_y_median = np.median(positions_y_all, axis=0)
            # Generate uniform grid
            _, positions_x_uniform, positions_y_uniform = grid_interpolate(
                None, positions_x_median, positions_y_median)
            return positions_x_uniform, positions_y_uniform

        positions_x_uniform, positions_y_uniform = _get_uniform_grid(positions_x_all,
                                                                     positions_y_all)

        logger.info("Generating common uniform grid: success.")

        # Create the arrays of XRF amplitudes for each emission line and normalize them
        def _get_eline_data(scan_img_dict, scaler_name):
            """
            Create the list of emission lines and the array with emission data.

            Array ``eline_data`` contains XRF maps for the emission lines:

            -- dimension 0 - the number of the emission line,
            the name of the emission line may be extracted from the respective
            entry of the list ``eline_list``

            -- dimensions 1 and 2 - Y and X coordinates of the map
            """
            eline_list = _get_eline_keys(_get_img_keys(scan_img_dict))
            eline_data = {}
            for eline in eline_list:
                data = []
                for img_dict in scan_img_dict:
                    d = _get_img_data(img_dict=scan_img_dict, key=eline)
                    if scaler_name:  # Normalization
                        d = normalize_data_by_scaler(d, _get_img_data(img_dict=scan_img_dict,
                                                                      key=scaler_name))
                    data.append(d)
                eline_data[eline] = np.asarray(data)
            return eline_list, eline_data

        eline_list, eline_data = _get_eline_data(scan_img_dict=scan_img_dict,
                                                 scaler_name=scaler_name)

        logger.info("Extracting XRF maps for emission lines: success.")

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

        # Align the stack of images
        def _align_stacks(eline_data, eline_alignment):
            """Align stack of XRF maps for each element"""
            sr = StackReg(StackReg.TRANSLATION)
            sr.register_stack(eline_data[eline_alignment], reference="previous")

            eline_data_aligned = {}
            for eline, data in eline_data.items():
                eline_data_aligned[eline] = sr.transform_stack(data)
            return eline_data_aligned

        eline_data_aligned = _align_stacks(eline_data=eline_data, eline_alignment=eline_alignment)

        logger.info("Aligning the image stack: success.")

        # Show stacks
        show_image_stack(eline_data_aligned, scan_energies)

        logger.info("Processing is complete.")


def _load_dataset_from_hdf5(*, start_id, end_id, wd_xrf):
    """
    Load dataset from processed HDF5 files

    Parameters
    ----------

    wd_xrf : str
        full (absolute) path name to the directory that contains processed HDF5 files
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
            read_hdf_APS(working_directory=wd_xrf, file_name=fln, load_summed_data=True,
                         load_each_channel=False, load_processed_each_channel=False,
                         load_raw_data=False, load_fit_results=True,
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
            xy_list.append(img["positions"]["xpos"].shape)

        # Determine if all sizes are identical
        if not [_ for _ in xy_list if _ != xy_list[0]]:
            _raise_error_exception(slist=list(range(xy_list)),
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


def show_image_stack(eline_data, energy, axis=0):
    """
    Display XRF Map stack
    """
    label_fontsize = 15

    if not eline_data:
        logger.warning("Emission line data dictionary is empty. There is nothing to plot.")
        return

    labels = list(eline_data.keys())
    label_current = labels[0]  # Set the first emission line as initial choice
    stack = eline_data[label_current]

    # Check dimensions
    if not stack.ndim == 3:
        raise ValueError("stack should be an ndarray with ndim == 3")

    fig = plt.figure(figsize=(11, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.25)
    ax1.set_xlabel("X", fontsize=label_fontsize)
    ax1.set_ylabel("Y", fontsize=label_fontsize)

    # select first image
    s = stack[0, :, :]
    im = s

    # display image
    im_plot = ax1.imshow(im)
    ax2.plot(energy, stack[:, 0, 0])
    ax2.grid()
    ax2.set_xlabel("Energy, keV", fontsize=label_fontsize)
    ax2.set_ylabel("Fluorescence", fontsize=label_fontsize)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider_energy = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
    ax_slider_energy.set_title(f"{energy[0]:.5f} keV")

    n_energy = 0
    slider = Slider(ax_slider_energy, 'Energy', 0, len(energy) - 1, valinit=n_energy, valfmt='%i')

    # Create buttons
    btn, ax_btn = [], []
    for n in range(len(labels)):
        p_left = 0.05 + 0.08 * n
        ax_btn.append(plt.axes([p_left, 0.01, 0.07, 0.04]))
        btn.append(Button(ax_btn[-1], labels[n]))

    def btn_clicked(event):
        nonlocal stack
        nonlocal label_current
        for n, ab in enumerate(ax_btn):
            if event.inaxes is ab:
                label_current = labels[n]  # Set the first emission line as initial choice
                stack = eline_data[label_current]
                redraw_image()
                print(f"Button is pressed: {labels[n]}")
                break

    # Set events to each button
    for b in btn:
        b.on_clicked(btn_clicked)

    def redraw_image():
        nonlocal n_energy
        im = stack[n_energy, :, :]
        im_plot.set_data(im)
        ax_slider_energy.set_title(f"{energy[n_energy]:.5f} keV")
        fig.canvas.draw()

    def update(val):
        nonlocal n_energy
        n_energy = int(slider.val)

        redraw_image()

    slider.on_changed(update)

    def onclick(event):
        if (event.inaxes == ax1) and (event.button == 1):
            xd, yd = event.xdata, event.ydata
            nx, ny = int(xd), int(yd)
            print(f"xd = {xd}   yd = {yd}    nx = {nx}  ny = {ny}")
            ax2.clear()
            ax2.plot(energy, stack[:, ny, nx])
            ax2.grid()
            ax2.set_xlabel("Energy, keV", fontsize=label_fontsize)
            ax2.set_ylabel("Fluorescence", fontsize=label_fontsize)
            fig.canvas.draw()
            fig.canvas.flush_events()

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()


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


if __name__ == "__main__":
    build_energy_map_api(start_id=92276, end_id=92281, param_file_name="param_335",
                         scaler_name="sclr1_ch4", wd=None, sequence="build_energy_map")
