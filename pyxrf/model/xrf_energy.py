import os
import re
import numpy as np
from pystackreg import StackReg

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

        #  TODO: check that the processed datasets are present for each file, also
        #     check that the selected scaler is present in each dataset

        print(f"Collected scan IDs: {scan_ids}")
        print(f"Collected scan energies: {scan_energy}")



        # Sort the lists based on energy
        #print(f"{sorted(zip(scan_energy, range(len(scan_energy))))}")
        scan_energy, sorted_indexes = list(zip(*sorted(zip(scan_energy, range(len(scan_energy))))))
        files_h5 = [files_h5[n] for n in sorted_indexes]
        scan_ids = [scan_ids[n] for n in sorted_indexes]
        scan_img_dict = [scan_img_dict[n] for n in sorted_indexes]
        scan_dataset = [scan_dataset[n] for n in sorted_indexes]



        print(f"scan_energy={scan_energy}")
        print(f"sorted_indexes={sorted_indexes}")

        n_scans = len(scan_ids)

        # Create the list of positional data
        positions_x_all = np.asarray([element['positions']['x_pos'] for element in scan_img_dict])
        positions_y_all = np.asarray([element['positions']['y_pos'] for element in scan_img_dict])
        # Median positions are probably the best for generating common uniform grid
        positions_x_median = np.median(positions_x_all, axis=0)
        positions_y_median = np.median(positions_y_all, axis=0)
        # Generate uniform grid
        _, positions_x_uniform, positions_y_uniform = grid_interpolate(
            None, positions_x_median, positions_y_median)


        # Create the arrays of XRF amplitudes for each emission line and normalize them
        dataset_names = [get_dataset_name(img_dict) for img_dict in scan_img_dict]
        eline_list = get_eline_list(files_h5[0], scan_img_dict[0])
        eline_data = {}
        for eline in eline_list:
            data = []
            for img_dict in scan_img_dict:
                dataset_name = get_dataset_name(img_dict)
                d = img_dict[dataset_name][eline]
                if scaler_name:
                    d = normalize_data_by_scaler(d, img_dict[dataset_name][scaler_name])
                data.append(d)
            eline_data[eline] = np.asarray(data)


        print(f"emission line keys: {eline_data.keys()}")
        print(f"{eline_data['Cl_K'].shape}")


        # Interpolate each image based on the common uniform positions
        for eline, data in eline_data.items():
            n_scans, _, _ = data.shape
            for n in range(n_scans):
                data[n, :, :], _, _ = grid_interpolate(data[n, :, :],
                                                       xx=positions_x_all[n, :, :],
                                                       yy=positions_y_all[n, :, :])
                                                       #xx_uniform=positions_x_uniform,
                                                       #yy_uniform=positions_y_uniform)


        # Align images

        # Select the reference element (strongest)
        ref_eline = eline_list[0]  # Select first for now TODO: rewrite this
        ref_eline = "Mn_K"
        sr = StackReg(StackReg.TRANSLATION)
        sr.register_stack(eline_data[ref_eline], reference="previous")

        eline_data_aligned = {}
        for eline, data in eline_data.items():
            eline_data_aligned[eline] = sr.transform_stack(data)

        print(f"after alignment: {eline_data_aligned[ref_eline].shape}")

        show_image_stack(eline_data_aligned[ref_eline], scan_energy)

        print(f"scan_ids={scan_ids}")
        print(f"files_h5={files_h5}")

        print(f"elines: {eline_list}")
        #print(f"scan_ids={scan_ids}")
        #print(f"files_h5={files_h5}")



        # Create common position grid

        # Interpolate all the scans to common grid


def show_image_stack(stack, energy, axis=0, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
	http://nbarbey.github.io/2011/07/08/matplotlib-slider.html

    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # check dim
    if not stack.ndim == 3:
        raise ValueError("stack should be an ndarray with ndim == 3")

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25)

    # select first image
    #s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    s = stack[0, :, :]
    #im = stack[s].squeeze()
    im = s

    # display image
    l = ax1.imshow(im)
    energy_plot, = ax2.plot(energy, stack[:, 0, 0])
    ax2.grid()
    plt.xlabel("Energy, keV")
    plt.ylabel("Fluorescence")

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider_energy = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_slider_energy.set_title(f"{energy[0]:.5f} keV")

    slider = Slider(ax_slider_energy, 'Energy', 0, len(energy) - 1, valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = stack[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()
        ax_slider_energy.set_title(f"{energy[ind]:.5f} keV")

    slider.on_changed(update)

    def onclick(event):
        if (event.inaxes == ax1) and (event.button == 1):
            xd, yd = event.xdata, event.ydata
            nx, ny = int(xd), int(yd)
            print(f"xd = {xd}   yd = {yd}    nx = {nx}  ny = {ny}")
            #energy_plot.set_data(energy, stack[:, ny, nx])
            ax2.clear()
            ax2.plot(energy, stack[:, ny, nx])
            ax2.grid()
            plt.xlabel("Energy, keV")
            plt.ylabel("Fluorescence")
            fig.canvas.draw()
            fig.canvas.flush_events()
            #ax2.cla()
            #ax2.plot(energy, stack[:, ny, nx])
            #plt. show()


    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

    #import time as ttime
    #ttime.sleep(10)


def get_dataset_name(img_dict, detector=None):
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
        patterns = ["(?<!det\d)fit$", "(?<!det\d\d)fit$", "(?<!det\d\d\d)fit$"]
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

def get_eline_list(file_name, img_dict):

    dataset_name = get_dataset_name(img_dict)
    all_keys = img_dict[dataset_name].keys()
    # The following pattern is based on assumption that elines names
    #        are of the form Si_K, La_L, P_K etc.
    re_pattern = "^[A-Z][a-z]?_[KLM]$"
    return [key for key in all_keys if re.search(re_pattern, key)]


if __name__ == "__main__":
    build_energy_map_api(start_id=92276, end_id=92281, param_file_name="param_335",
                         scaler_name="sclr1_ch4", wd=None, sequence="build_energy_map")
