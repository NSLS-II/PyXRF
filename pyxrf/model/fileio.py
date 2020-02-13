from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import sys
import h5py
import numpy as np
import os
import re
from collections import OrderedDict
import pandas as pd
import json
import skimage.io as sio
from PIL import Image
import copy
import glob
import ast
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Float, Enum, Bool
from .load_data_from_db import (db, fetch_data_from_db, flip_data,
                                helper_encode_list, helper_decode_list,
                                write_db_to_hdf)
from ..core.utils import normalize_data_by_scaler, grid_interpolate
from .scan_metadata import ScanMetadataXRF
import requests
from distutils.version import LooseVersion

import logging
import warnings

import pyxrf
pyxrf_version = pyxrf.__version__

logger = logging.getLogger()
warnings.filterwarnings('ignore')

sep_v = os.sep


class FileIOModel(Atom):
    """
    This class focuses on file input and output.

    Attributes
    ----------
    working_directory : str
        current working path
    file_name : str
        name of loaded file
    file_name_silent_change : bool
        If this flag is set to True, then ``file_name`` may be changed once without
        starting file read operation. The flag is automatically reset to False.
    load_status : str
        Description of file loading status
    data_sets : dict
        dict of experiment data, 3D array
    img_dict : dict
        Dict of 2D arrays, such as 2D roi pv or fitted data
    """
    window_title = Str()
    window_title_base = Str()

    working_directory = Str()
    file_name = Str()
    file_name_silent_change = Bool(False)
    file_path = Str()
    load_status = Str()
    data_sets = Typed(OrderedDict)
    img_dict = Dict()
    param_fit = Dict()
    file_channel_list = List()

    runid = Int(-1)
    h_num = Int(1)
    v_num = Int(1)
    fname_from_db = Str()

    file_opt = Int(-1)
    data = Typed(np.ndarray)
    data_all = Typed(np.ndarray)
    selected_file_name = Str()
    # file_name = Str()
    mask_data = Typed(object)
    mask_name = Str()
    mask_opt = Int(0)
    load_each_channel = Bool(False)

    # Used while loading data from database
    # True: overwrite existing data file if it exists
    # False: create new file with unique name (original name + version number)
    file_overwrite_existing = Bool(False)

    p1_row = Int(-1)
    p1_col = Int(-1)
    p2_row = Int(-1)
    p2_col = Int(-1)

    data_ready = Bool(False)

    # Scan metadata
    scan_metadata = Typed(ScanMetadataXRF)
    # Indicates if metadata is available for recently loaded scan
    scan_metadata_available = Bool(False)
    # Indicates if the incident energy is available in metadata for recently loaded scan
    incident_energy_available = Bool(False)

    # Changing this variable sets incident energy in the ``plot_model``
    #   Must be linked with the function ``plot_model.set_incident_energy``
    # This value is not updated if incident energy parameter is changed somewhere else, therefore
    #   its value should not be used for computations!!!
    incident_energy_set = Float(0.0)

    def __init__(self, **kwargs):
        self.working_directory = kwargs['working_directory']
        self.mask_data = None

        # Display PyXRF version in the window title
        ver_str, new_ver = self._get_pyxrf_version_str()
        if new_ver is not None:
            ver_str += f" - new version {new_ver} is available"
        self.window_title_base = f"PyXRF: X-ray Fluorescence Analysis Tool ({ver_str})"
        self.window_title = self.window_title_base

    def _get_pyxrf_version_str(self):
        """
        The function returns the tuple of strings:
        - current version number of PyXRF;
        - the latest version of PyXRF from nsls2forge conda channel.
        If current version is the latest, then the second string None.
        """

        # Determine the current version of PyXRF
        global pyxrf_version
        pyxrf_version_str = pyxrf_version
        if pyxrf_version_str[0].lower() != 'v':
            pyxrf_version_str = f"v{pyxrf_version_str}"

        logger.info("Checking for new version availability ...")

        # Now find the latest version available at nsls2forge
        pyxrf_latest_version_str = None
        try:
            # Create a list of available versions
            r = requests.get('https://conda.anaconda.org/nsls2forge/noarch/repodata.json')
            pkgs = r.json()
            pyxrf_ver = []
            for pkg in pkgs['packages'].keys():
                if pkg.startswith('pyxrf'):
                    pkg_version = LooseVersion(pkg.split('-')[1])
                    pyxrf_ver.append(pkg_version)

            if len(pyxrf_ver):
                max_version = pyxrf_ver[0]
                for pkg_version in pyxrf_ver:
                    if max_version < pkg_version:
                        max_version = pkg_version

                current_version = LooseVersion(pyxrf_version)

                if current_version < max_version:
                    pyxrf_latest_version_str = f"v{max_version}"

            if pyxrf_latest_version_str is not None:
                logger.info(f"New version of PyXRF ({pyxrf_latest_version_str}) was found "
                            "in the 'nsls2forge' conda channel")
            else:
                logger.info(f"You have the latest version of PyXRF")

        except Exception:
            # This exception is mostly likely to happen if there is no internet connection or
            #    nsls2forge is unreachable. Then ignore the procedure, assume that the current
            #    version is the latest.
            logger.warning("Failed to check availability of the latest version of PyXRF "
                           "in the 'nsls2forge' conda channel.")
            pass

        return pyxrf_version_str, pyxrf_latest_version_str

    def window_title_clear(self):
        self.window_title = self.window_title_base

    def window_title_set_file_name(self, file_name):
        self.window_title = f"{self.window_title_base} - File: {file_name}"

    def window_title_set_run_id(self, run_id):
        self.window_title = f"{self.window_title_base} - Scan ID: {run_id}"

    def _metadata_update_program_state(self):
        """
        Update program state based on metadata:

        -- enable controls (mostly ``View Metadata`` button in ``File IO`` tab

        -- set incident energy if it is available

        -- print logger warning if incident energy is not available in metadata
        (or if metadata does not exist)
        """

        self.scan_metadata_available = False
        self.incident_energy_available = False
        if self.scan_metadata is not None:
            self.scan_metadata_available = self.scan_metadata.is_metadata_available()
            self.incident_energy_available = self.scan_metadata.is_mono_incident_energy_available()

        if self.incident_energy_available:
            # Fetch incident energy from metadata if it exists
            self.incident_energy_set = self.scan_metadata.get_mono_incident_energy()
            logger.info(f"Incident energy {self.incident_energy_set} keV was extracted from the scan metadata")
        else:
            logger.warning(
                "Incident energy is not available in scan metadata and needs to be set manually: "
                "click 'Find Elements Automatically' button in 'Fit' "
                "tab to access the settings dialog box.")

    @observe(str('file_name'))
    def update_more_data(self, change):
        if change['value'] == 'temp':
            # 'temp' is used to reload the same file
            return

        if self.file_name_silent_change:
            self.file_name_silent_change = False
            logger.info(f"File name is silently changed. New file name is '{change['value']}'")
            return

        self.file_channel_list = []
        logger.info('File is loaded: %s' % (self.file_name))

        # focus on single file only
        self.img_dict, self.data_sets, self.scan_metadata = \
            file_handler(self.working_directory,
                         self.file_name,
                         load_each_channel=self.load_each_channel)

        # Process metadata
        self._metadata_update_program_state()

        self.data_ready = True

        self.file_channel_list = list(self.data_sets.keys())
        self.file_opt = 0  # use summed data as default

    @observe(str('runid'))
    def _update_fname(self, change):
        self.fname_from_db = 'scan2D_'+str(self.runid)

    def load_data_runid(self):
        """
        Load data according to runID number.

        requires databroker
        """
        if db is None:
            raise RuntimeError("databroker is not installed. This function "
                               "is disabled.  To install databroker, see "
                               "https://nsls-ii.github.io/install.html")
        # if self.h_num != 0 and self.v_num != 0:
        #     datashape = [self.v_num, self.h_num]

        #  one way to cache data is to save as h5 file, to be considered later
        # tmp_wd = '~/.tmp/'
        # if not os.path.exists(tmp_wd):
        #     os.makedirs(tmp_wd)
        # fpath = os.path.join(tmp_wd, self.fname_from_db)
        # if not os.path.exists(fpath):
        #     make_hdf(self.runid, fname=fpath)
        # self.img_dict, self.data_sets = file_handler(tmp_wd,
        #                                             self.fname_from_db,
        #                                             load_each_channel=self.load_each_channel)

        rv = render_data_to_gui(self.runid,
                                create_each_det=self.load_each_channel,
                                working_directory=self.working_directory,
                                file_overwrite_existing=self.file_overwrite_existing)

        if rv is None:
            logger.error(f"Data from scan #{self.runid} was not loaded")
            return

        img_dict, self.data_sets, fname, detector_name, self.scan_metadata = rv

        # Replace relative scan ID with true scan ID.
        if (self.runid) < 0 and ("scan_id" in self.scan_metadata):
            self.runid = int(self.scan_metadata["scan_id"])

        # Process metadata
        self._metadata_update_program_state()

        # Change file name without rereading the file
        self.file_name_silent_change = True
        self.file_name = os.path.basename(fname)
        logger.info(f"Data loading: complete dataset for the detector "
                    f"'{detector_name}' was loaded successfully.")

        self.file_channel_list = list(self.data_sets.keys())

        # Disable loading from 'analysis store' for now (because there is no 'analysis store')
        # ----------------------------------------------------------------
        # # Load results from the analysis store
        # from .data_to_analysis_store import get_analysis_result
        # hdr = get_analysis_result(self.runid)
        # if hdr is not None:
        #     d1 = hdr.table(stream_name='primary')
        #     # d2 = hdr.table(stream_name='spectrum')
        #     self.param_fit = hdr.start.processor_parameters
        #     # self.data = d2['summed_spectrum_experiment']
        #     fit_result = {k: v for k, v in zip(d1['element_name'], d1['map'])}
        #     # tmp = {k: v for k, v in self.img_dict.items()}
        #     img_dict['scan2D_{}_fit'.format(self.runid)] = fit_result
        # ----------------------------------------------------------------

        self.img_dict = img_dict

        try:
            self.selected_file_name = self.file_channel_list[self.file_opt]
        except IndexError:
            pass

        # passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[self.selected_file_name].raw_data
        # get summed data or based on mask
        self.data = self.data_sets[self.selected_file_name].get_sum()

        self.data_ready = True
        self.file_opt = 0  # use summed data as default

    @observe(str('file_opt'))
    def choose_file(self, change):

        if not self.data_ready:
            return

        if self.file_opt < 0 or self.file_opt >= len(self.file_channel_list):
            self.file_opt = 0

        # selected file name from all channels
        # controlled at top level gui.py startup
        try:
            self.selected_file_name = self.file_channel_list[self.file_opt]
        except IndexError:
            pass

        # passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[self.selected_file_name].raw_data
        # get summed data or based on mask
        self.data = self.data_sets[self.selected_file_name].get_sum()

    def get_selected_detector_channel(self):
        r"""
        Returns selected channel name. Expected values are ``sum``, ``det1``, ``det2``, etc.
        If no channel is selected or it is impossible to determine the channel name, then
        the return value is ``None`` (this is not a normal outcome).
        """
        det_channel = None
        if self.selected_file_name:
            try:
                # The channel is supposed to be the last component of the 'selected_file_name'
                det_channel = self.selected_file_name.split("_")[-1]
            except Exception:
                pass
        return det_channel

    def apply_mask(self):
        """Apply mask with different options.
        """
        if self.mask_opt == 2:
            # load mask data
            if len(self.mask_name) > 0:
                mask_file = os.path.join(self.working_directory,
                                         self.mask_name)
                try:
                    if 'npy' in mask_file:
                        self.mask_data = np.load(mask_file)
                    elif 'txt' in mask_file:
                        self.mask_data = np.loadtxt(mask_file)
                    else:
                        self.mask_data = np.array(Image.open(mask_file))
                except IOError:
                    logger.error('Mask file cannot be loaded.')

                for k in six.iterkeys(self.img_dict):
                    if 'fit' in k:
                        self.img_dict[k][self.mask_name] = self.mask_data
        else:
            self.mask_data = None
            data_s = self.data_all.shape
            if self.mask_opt == 1:
                valid_opt = False
                # define square mask region
                if self.p1_row >= 0 and self.p1_col >= 0 and self.p1_row < data_s[0] and self.p1_col < data_s[1]:
                    self.data_sets[self.selected_file_name].point1 = [self.p1_row, self.p1_col]
                    logger.info('Starting position is {}.'.format([self.p1_row, self.p1_col]))
                    valid_opt = True
                    if self.p2_row > self.p1_row and self.p2_col > self.p1_col and \
                            self.p2_row < data_s[0] and self.p2_col < data_s[1]:
                        self.data_sets[self.selected_file_name].point2 = [self.p2_row, self.p2_col]
                        logger.info('Ending position is {}.'.format([self.p2_row, self.p2_col]))
                if valid_opt is False:
                    logger.info('The positions are not valid. No mask is applied.')
            else:
                self.data_sets[self.selected_file_name].delete_points()
                logger.info('Do not apply mask.')

        # passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[self.selected_file_name].raw_data
        # get summed data or based on mask
        self.data = self.data_sets[self.selected_file_name].get_sum(self.mask_data)


plot_as = ['Sum', 'Point', 'Roi']


class DataSelection(Atom):
    """
    Attributes
    ----------
    filename : str
    plot_choice : enum
        methods ot plot
    point1 : str
        starting position
    point2 : str
        ending position
    roi : list
    raw_data : array
        experiment 3D data
    data : array
    plot_index : int
        plot data or not, sum or roi or point
    """
    filename = Str()
    plot_choice = Enum(*plot_as)
    # point1 = Str('0, 0')
    # point2 = Str('0, 0')
    point1 = List()
    point2 = List()
    raw_data = Typed(np.ndarray)
    data = Typed(np.ndarray)
    plot_index = Int(0)
    fit_name = Str()
    fit_data = Typed(np.ndarray)

    @observe(str('plot_index'))
    def _update_roi(self, change):
        if self.plot_index == 0:
            return
        elif self.plot_index == 1:
            self.data = self.get_sum()

    def delete_points(self):
        self.point1 = []
        self.point2 = []

    def get_sum(self, mask=None):
        if len(self.point1) == 0 and len(self.point2) == 0:
            SC = SpectrumCalculator(self.raw_data)
            spec = SC.get_spectrum(mask=mask)
        else:
            SC = SpectrumCalculator(self.raw_data,
                                    pos1=self.point1,
                                    pos2=self.point2)
            spec = SC.get_spectrum()
        # Return the 'sum' spectrum as regular 64-bit float (raw data is in 'np.float32')
        return spec.astype(np.float64, copy=False)


class SpectrumCalculator(object):
    """
    Calculate summed spectrum according to starting and ending positions.

    Attributes
    ----------
    data : array
        3D array of experiment data
    pos1 : str
        starting position
    pos2 : str
        ending position
    """

    def __init__(self, data,
                 pos1=None, pos2=None):
        self.data = data
        self.pos1 = pos1
        self.pos2 = pos2

    def get_spectrum(self, mask=None):
        """
        Get roi sum from point positions, or from mask file.
        """
        if mask is None:
            if not self.pos1 and not self.pos2:
                return np.sum(self.data, axis=(0, 1))
            elif self.pos1 and not self.pos2:
                return self.data[self.pos1[0], self.pos1[1], :]
            else:
                return np.sum(self.data[self.pos1[0]:self.pos2[0],
                                        self.pos1[1]:self.pos2[1], :],
                              axis=(0, 1))
        else:
            spectrum_sum = np.zeros(self.data.shape[2])
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    if mask[i, j] > 0:
                        spectrum_sum += self.data[i, j, :]
            return spectrum_sum


def file_handler(working_directory, file_name, load_each_channel=True, spectrum_cut=3000):
    # send information on GUI level later !
    get_data_nsls2 = True
    try:
        if get_data_nsls2 is True:
            return read_hdf_APS(working_directory, file_name,
                                spectrum_cut=spectrum_cut,
                                load_each_channel=load_each_channel)
        else:
            return read_MAPS(working_directory,
                             file_name, channel_num=1)
    except IOError as e:
        logger.error("I/O error({0}): {1}".format(e.errno, e.strerror))
        logger.error('Please select .h5 file')
    except Exception:
        logger.error("Unexpected error:", sys.exc_info()[0])
        raise


def read_xspress3_data(file_path):
    """
    Data IO for xspress3 format.

    Parameters
    ----------
    working_directory : str
        path folder
    file_name : str

    Returns
    -------
    data_output : dict
        with data from each channel
    """
    data_output = {}

    # file_path = os.path.join(working_directory, file_name)
    with h5py.File(file_path, 'r') as f:
        data = f['entry/instrument']

        # data from channel summed
        exp_data = np.asarray(data['detector/data'])
        xval = np.asarray(data['NDAttributes/NpointX'])
        yval = np.asarray(data['NDAttributes/NpointY'])

    # data size is (ysize, xsize, num of frame, num of channel, energy channel)
    exp_data = np.sum(exp_data, axis=2)
    num_channel = exp_data.shape[2]
    # data from each channel
    for i in range(num_channel):
        channel_name = 'channel_'+str(i+1)
        data_output.update({channel_name: exp_data[:, :, i, :]})

    # change x,y to 2D array
    xval = xval.reshape(exp_data.shape[0:2])
    yval = yval.reshape(exp_data.shape[0:2])

    data_output.update({'x_pos': xval})
    data_output.update({'y_pos': yval})

    return data_output


def output_data(dataset_dict=None, output_dir=None,
                file_format='tiff', scaler_name=None, use_average=False,
                dataset_name=None, quant_norm=False,
                param_quant_analysis=None,
                positions_dict=None,
                interpolate_to_uniform_grid=False,
                scaler_name_list=None):
    """
    Read data from h5 file and transfer them into txt.

    Parameters
    ----------
    dataset_dict : dict(ndarray)
        Dictionary of XRF maps contained in the selected dataset. Each XRF map is saved in
        the individual file. File name consists of the detector channel name (e.g. 'detsum', 'det1' etc.)
        and map name (dictionary key). Optional list of scaler names 'scaler_name_list' may be passed
        to the function. If map name is contained in the scaler name list, then the detector channel
        name is not attached to the file name.
    output_dir : str
        which folder to save those txt file
    file_format : str, optional
        tiff or txt
    scaler_name : str, optional
        if given, normalization will be performed.
    use_average : Bool, optional
        when normalization, multiply by the mean value of scaler,
        i.e., norm_data = data/scaler * np.mean(scaler)
    dataset_name : str
        the name of the selected datset (in Element Map tab)
        should end with the suffix '_fit' (for sum of all channels), '_det1_fit" etc.
    quant_norm : bool
        True - quantitative normalization is enabled, False - disabled
    param_quant_analysis : ParamQuantitativeAnalysis
        reference to class, which contains parameters for quantitative normalization
    interpolate_to_uniform_grid : bool
        interpolate the result to uniform grid before saving to tiff and txt files
        The grid dimensions match the dimensions of positional data for X and Y axes.
        The range of axes is chosen to fit the values of X and Y.
    scaler_name_list : list(str)
        The list of names of scalers that may exist in the dataset 'dataset_dict'
    """

    if not dataset_name:
        raise RuntimeError("Dataset is not selected. Data can not be saved.")

    if dataset_dict is None:
        dataset_dict = {}

    if positions_dict is None:
        positions_dict = {}

    # Extract the detector channel name from dataset name
    #   Dataset name ends with '_fit' for the sum of all channels
    #   and '_detX_fit' for detector X (X is 1, 2, 3 ..)
    # The extracted detector channel name should be 'detsum', 'det1', 'det2' etc.
    dset = None
    if re.search(r"_det\d+_fit$", dataset_name):
        dset = re.search(r"_det\d_", dataset_name)[0]
        dset = dset.strip('_')
    elif re.search(r"_fit$", dataset_name):
        dset = "detsum"
    if not dset:
        raise RuntimeError(f"Dataset '{dataset_name}' contains no useful data. "
                           "Select different dataset to save data.")

    file_format = file_format.lower()

    fit_output = {}
    for k, v in dataset_dict.items():
        fit_output[k] = v

    for k, v in positions_dict.items():
        fit_output[k] = v

    logger.info(f"Saving data as {file_format.upper()} files. Directory '{output_dir}'")
    if scaler_name:
        logger.info(f"Data is NORMALIZED before saving. Scaler: '{scaler_name}'")

    if(interpolate_to_uniform_grid):
        if ("x_pos" in fit_output) and ("y_pos" in fit_output):
            logger.info(f"Data is INTERPOLATED to uniform grid.")
            for k, v in fit_output.items():
                # Do not interpolation positions
                if 'pos' in k:
                    continue

                fit_output[k], xx, yy = grid_interpolate(v, fit_output["x_pos"], fit_output["y_pos"])

            fit_output["x_pos"] = xx
            fit_output["y_pos"] = yy
        else:
            logger.error(f"Positional data 'x_pos' and 'y_pos' is not found in the dataset.\n"
                         "Iterpolation to uniform grid can not be performed. "
                         "Data is saved without interpolation.")

    output_data_to_tiff(fit_output, output_dir=output_dir,
                        file_format=file_format, name_prefix_detector=dset, name_append="",
                        scaler_name=scaler_name, quant_norm=quant_norm,
                        param_quant_analysis=param_quant_analysis,
                        use_average=use_average,
                        scaler_name_list=scaler_name_list)


def output_data_to_tiff(fit_output,
                        output_dir=None,
                        file_format='tiff', name_prefix_detector=None, name_append=None,
                        scaler_name=None, scaler_name_list=None,
                        quant_norm=False, param_quant_analysis=None,
                        use_average=False):
    """
    Read data in memory and save them into tiff to txt.

    Parameters
    ----------
    fit_output:
        dict of fitting data and scaler data
    output_dir : str, optional
        which folder to save those txt file
    file_format : str, optional
        tiff or txt
    name_prefix_detector : str
        prefix appended to file name except for the files that contain positional data and scalers
    name_append: str, optional
        more information saved to output file name
    scaler_name : str, optional
        if given, normalization will be performed.
    scaler_name_list : list(str)
        The list of names of scalers that may exist in the dataset 'dataset_dict'
    quant_norm : bool
        True - apply quantitative normalization, False - use normalization by scaler
    param_quant_analysis : ParamQuantitativeAnalysis
        reference to class, which contains parameters for quantitative normalization,
        if None, then quantitative normalization will be skipped
    use_average : Bool, optional
        when normalization, multiply by the mean value of scaler,
        i.e., norm_data = data/scaler * np.mean(scaler)
    """

    if output_dir is None:
        raise ValueError("Output directory is not specified.")

    if name_append:
        name_append = f"_{name_append}"
    else:
        # If 'name_append' is None, set it to "" so that it could be safely appended to a string
        name_append = ""

    file_format = file_format.lower()

    allowed_formats = ('txt', 'tiff')
    if file_format not in allowed_formats:
        raise RuntimeError(f"The specified format '{file_format}' not in {allowed_formats}")

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    def _save_data(data, *, output_dir, file_name,
                   name_prefix_detector, name_append,
                   file_format, scaler_name_list):

        # The 'file_format' is specified as file extension
        file_extension = file_format.lower()

        # If data is scalar or position, then don't attach the prefix
        fname = f"{name_prefix_detector}_{file_name}" \
            if (file_name not in scaler_name_list) and ("pos" not in file_name) \
            else file_name

        fname = f"{fname}{name_append}.{file_extension}"
        fname = os.path.join(output_dir, fname)
        if file_format.lower() == 'tiff':
            sio.imsave(fname, data.astype(np.float32))
        elif file_format.lower() == 'txt':
            np.savetxt(fname, data.astype(np.float32))
        else:
            raise ValueError(f"Function is called with invalid file format '{file_format}'.")

    if quant_norm:
        if param_quant_analysis:
            for data_name, data in fit_output.items():
                # Quantitative normalization
                data_normalized, quant_norm_applied = param_quant_analysis.apply_quantitative_normalization(
                    data_in=data,
                    scaler_dict=fit_output,
                    scaler_name_default=None,  # We don't want data to be scaled
                    data_name=data_name,
                    name_not_scalable=None)  # For simplicity, all saved maps are normalized
                if quant_norm_applied:
                    # Save data only if quantitative normalization was performed.
                    _save_data(data_normalized, output_dir=output_dir,
                               file_name=data_name,
                               name_prefix_detector=name_prefix_detector,
                               name_append=f"{name_append}_quantitative",
                               file_format=file_format,
                               scaler_name_list=scaler_name_list)
        else:
            logger.error("Quantitative analysis parameters are not provided. "
                         f"Quantitative data is not saved in {file_format.upper()} format.")

    # Normalize data if scaler is provided
    if scaler_name is not None:
        if scaler_name in fit_output:
            scaler_data = fit_output[scaler_name]
            for data_name, data in fit_output.items():
                if 'pos' in data_name or 'r2' in data_name:
                    continue
                # Normalization of data
                data_normalized = normalize_data_by_scaler(data, scaler_data)
                if use_average is True:
                    data_normalized *= np.mean(scaler_data)

                _save_data(data_normalized, output_dir=output_dir,
                           file_name=data_name,
                           name_prefix_detector=name_prefix_detector,
                           name_append=f"{name_append}_norm",
                           file_format=file_format,
                           scaler_name_list=scaler_name_list)
        else:
            logger.warning(f"The scaler '{scaler_name}' was not found. Data normalization "
                           f"was not performed for {file_format.upper()} file.")

    # Always save not normalized data
    for data_name, data in fit_output.items():
        _save_data(data, output_dir=output_dir,
                   file_name=data_name,
                   name_prefix_detector=name_prefix_detector,
                   name_append=name_append,
                   file_format=file_format,
                   scaler_name_list=scaler_name_list)


def read_hdf_APS(working_directory,
                 file_name, spectrum_cut=3000,
                 # The following parameters allow fine grained control over what is loaded from the file
                 load_summed_data=True,  # Enable loading of RAW, FIT or ROI data from 'sum' channel
                 load_each_channel=False,  # .. RAW data from individual detector channels
                 load_processed_each_channel=True,  # .. FIT or ROI data from the detector channels
                 load_raw_data=True,  # For all channels: load RAW data
                 load_fit_results=True,  # .. load FIT data
                 load_roi_results=True):  # .. load ROI data
    """
    Data IO for files similar to APS Beamline 13 data format.
    This might be changed later.

    Parameters
    ----------

    working_directory : str
        path folder
    file_name : str
        selected h5 file
    spectrum_cut : int, optional
        only use spectrum from, say 0, 3000
    load_summed_data : bool, optional
        load summed spectrum or not
    load_each_channel : bool, optional
        indicates whether to load raw experimental data for each detector channel or not
    load_raw_data : bool
        load raw experimental data
    load_processed_each_channel : bool
        indicates whether or not to load processed results (fit, roi) for each detector channel
    load_fit_results :bool
        load fitting results
    load_roi_results : bool
        load results of roi computation

    Returns
    -------
    data_dict : dict
        with fitting data
    data_sets : dict
        data from each channel and channel summed, a dict of DataSelection objects
    """
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # Empty container for metadata
    mdata = ScanMetadataXRF()

    file_path = os.path.join(working_directory, file_name)

    # defined in other_list in config file
    try:
        dict_sc = retrieve_data_from_hdf_suitcase(file_path)
    except Exception:
        dict_sc = {}

    with h5py.File(file_path, 'r+') as f:

        # Retrieve metadata if it exists
        if "xrfmap/scan_metadata" in f:  # Metadata is always loaded
            metadata = f["xrfmap/scan_metadata"]
            for key, value in metadata.attrs.items():
                # Convert ndarrays to lists (they were lists before they were saved)
                if isinstance(value, np.ndarray):
                    value = list(value)
                mdata[key] = value

        data = f['xrfmap']
        fname = file_name.split('.')[0]
        if load_summed_data and load_raw_data:
            try:
                # data from channel summed
                exp_data = np.array(data['detsum/counts'][:, :, 0:spectrum_cut],
                                    dtype=np.float32)
                logger.warning(f"We use spectrum range from 0 to {spectrum_cut}")
                logger.info(f"Exp. data from h5 has shape of: {exp_data.shape}")

                fname_sum = f"{fname}_sum"
                DS = DataSelection(filename=fname_sum,
                                   raw_data=exp_data)

                data_sets[fname_sum] = DS
                logger.info('Data of detector sum is loaded.')
            except KeyError:
                print('No data is loaded for detector sum.')

        if 'scalers' in data:  # Scalers are always loaded if data is available
            det_name = data['scalers/name']
            temp = {}
            for i, n in enumerate(det_name):
                if not isinstance(n, six.string_types):
                    n = n.decode()
                temp[n] = data['scalers/val'].value[:, :, i]
            img_dict[f"{fname}_scaler"] = temp
            # also dump other data from suitcase if required
            if len(dict_sc) != 0:
                img_dict[f"{fname}_scaler"].update(dict_sc)

        if 'positions' in data:  # Positions are always loaded if data is available
            pos_name = data['positions/name']
            temp = {}
            for i, n in enumerate(pos_name):
                if not isinstance(n, six.string_types):
                    n = n.decode()
                temp[n] = data['positions/pos'].value[i, :]
            img_dict['positions'] = temp

        # TODO: rewrite the algorithm for finding the detector channels (not robust)
        # find total channel:
        channel_num = 0
        for v in list(data.keys()):
            if 'det' in v:
                channel_num = channel_num+1
        channel_num = channel_num-1  # do not consider det_sum

        # data from each channel
        if load_each_channel and load_raw_data:
            for i in range(1, channel_num+1):
                det_name = f"det{i}"
                file_channel = f"{fname}_det{i}"
                try:
                    exp_data_new = np.array(data[f"{det_name}/counts"][:, :, 0:spectrum_cut],
                                            dtype=np.float32)
                    DS = DataSelection(filename=file_channel,
                                       raw_data=exp_data_new)
                    data_sets[file_channel] = DS
                    logger.info(f"Data from detector channel {i} is loaded.")
                except KeyError:
                    print(f"No data is loaded for {det_name}.")

        if load_processed_each_channel:
            for i in range(1, channel_num + 1):
                det_name = f"det{i}"
                file_channel = f"{fname}_det{i}"
                if 'xrf_fit' in data[det_name] and load_fit_results:
                    try:
                        fit_result = get_fit_data(data[det_name]['xrf_fit_name'].value,
                                                  data[det_name]['xrf_fit'].value)
                        img_dict.update({f"{file_channel}_fit": fit_result})
                        # also include scaler data
                        if 'scalers' in data:
                            img_dict[f"{file_channel}_fit"].update(img_dict[f"{fname}_scaler"])
                    except IndexError:
                        logger.info(f"No fitting data is loaded for channel {i}.")

                if 'xrf_roi' in data[det_name] and load_roi_results:
                    try:
                        fit_result = get_fit_data(data[det_name]['xrf_roi_name'].value,
                                                  data[det_name]['xrf_roi'].value)
                        img_dict.update({f"{file_channel}_roi": fit_result})
                        # also include scaler data
                        if 'scalers' in data:
                            img_dict[f"{file_channel}_roi"].update(img_dict[f"{fname}_scaler"])
                    except IndexError:
                        logger.info(f"No ROI data is loaded for channel {i}.")

        # Prepare the following code for possible deletion in the future
        #               (we don't use ROI maps recorded during the experiment)
        # if 'roimap' in data:
        #    if 'sum_name' in data['roimap']:
        #        det_name = data['roimap/sum_name']
        #        temp = {}
        #        for i, n in enumerate(det_name):
        #            temp[n] = data['roimap/sum_raw'].value[:, :, i]
        #            # bad points on first one
        #            try:
        #                temp[n][0, 0] = temp[n][1, 0]
        #            except IndexError:
        #                temp[n][0, 0] = temp[n][0, 1]
        #        img_dict[f"{fname}_roi"] = temp
        #        # also include scaler data
        #        if 'scalers' in data:
        #            img_dict[f"{fname}_roi"].update(img_dict[f"{fname}_scaler"])
        #
        #    if 'det_name' in data['roimap']:
        #        det_name = data['roimap/det_name']
        #        temp = {}
        #        for i, n in enumerate(det_name):
        #            temp[n] = data['roimap/det_raw'].value[:, :, i]
        #            try:
        #                temp[n][0, 0] = temp[n][1, 0]
        #            except IndexError:
        #                temp[n][0, 0] = temp[n][0, 1]
        #        img_dict[f"{fname}_roi_each"] = temp

        # read fitting results from summed data
        if 'xrf_fit' in data['detsum'] and load_summed_data and load_fit_results:
            try:
                fit_result = get_fit_data(data['detsum']['xrf_fit_name'].value,
                                          data['detsum']['xrf_fit'].value)
                img_dict.update({f"{fname}_fit": fit_result})
                if 'scalers' in data:
                    img_dict[f"{fname}_fit"].update(img_dict[f"{fname}_scaler"])
            except (IndexError, KeyError):
                logger.info('No fitting data is loaded for channel summed data.')

        if 'xrf_roi' in data['detsum'] and load_summed_data and load_roi_results:
            try:
                fit_result = get_fit_data(data['detsum']['xrf_roi_name'].value,
                                          data['detsum']['xrf_roi'].value)
                img_dict.update({f"{fname}_roi": fit_result})
                if 'scalers' in data:
                    img_dict[f"{fname}_roi"].update(img_dict[f"{fname}_scaler"])
            except (IndexError, KeyError):
                logger.info('No ROI data is loaded for summed data.')

    return img_dict, data_sets, mdata


def render_data_to_gui(runid, *, create_each_det=False, working_directory=None, file_overwrite_existing=False):
    """
    Read data from databroker and save to Atom class which GUI can take.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        id number for given run
    create_each_det : bool
        True: load data from all detector channels
        False: load only the sum of all channels
    working_directory : str
        path to the directory where data files are saved
    file_overwrite_existing : bool
        True: overwrite data file if it exists
        False: create unique file name by adding version number
    """

    spectrum_cut = 3000  # Constant: the number of spectrum points to load 3000 ~ 3 keV

    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # Don't create unique file name if the existing file is to be overwritten
    fname_add_version = not file_overwrite_existing

    # Create file name here, so that working directory may be attached to the file name
    prefix = 'scan2D_'
    fname = f"{prefix}{runid}.h5"
    if working_directory:
        fname = os.path.join(working_directory, fname)

    data_from_db = fetch_data_from_db(runid,
                                      fpath=fname,
                                      fname_add_version=fname_add_version,
                                      file_overwrite_existing=file_overwrite_existing,
                                      create_each_det=create_each_det,
                                      # Always create data file (processing results
                                      #   are going to be saved in the file)
                                      output_to_file=True)

    if not len(data_from_db):
        logger.warning(f"No detector data was found in Scan #{runid}")
        return
    else:
        logger.info(f"Data from {len(data_from_db)} detectors were found in Scan #{runid}.")
        if len(data_from_db) > 1:
            logger.warning(f"Selecting only the first dataset from Scan #{runid}.")

    # If the experiment contains data from multiple detectors (for example two separate
    #   Xpress3 detectors) that need to be treated separately, only the data from the
    #   first detector is loaded. Data from the second detector is saved to file and
    #   can be loaded from the file. Currently this is a very rare case (only one set
    #   of such experiments from SRX beamline exists).
    data_out = data_from_db[0]['dataset']
    fname = data_from_db[0]['file_name']
    detector_name = data_from_db[0]['detector_name']
    scan_metadata = data_from_db[0]['metadata']

    # Create file name for the 'sum' dataset ('file names' are used as dictionary
    #   keys in data storage containers, as channel labels in plot legends,
    #   and as channel names in data selection widgets.
    #   Since there is currently no consistent metadata in the start documents
    #   and/or data files, let's leave original labeling conventions for now.
    fname_no_ext = os.path.splitext(os.path.basename(fname))[0]
    fname_sum = fname_no_ext + '_sum'

    # Determine the number of available detector channels and create the list
    #   of channel names. The channels are named as 'det1', 'det2', 'det3' etc.
    xrf_det_list = [nm for nm in data_out.keys() if 'det' in nm and 'sum' not in nm]

    det_sum = None
    if 'det_sum' in data_out:
        det_sum = np.copy(data_out['det_sum'][:, :, 0:spectrum_cut])
    else:
        for det_name in xrf_det_list:
            if det_sum is None:
                det_sum = np.array(data_out[det_name][:, :, 0:spectrum_cut])
            else:
                det_sum += data_out[det_name][:, :, 0:spectrum_cut]

    DS = DataSelection(filename=fname_sum,
                       raw_data=det_sum)
    data_sets[fname_sum] = DS

    logger.info("Data loading: channel sum is loaded successfully.")

    for det_name in xrf_det_list:
        exp_data = np.array(data_out[det_name][:, :, 0:spectrum_cut])
        fln = f"{fname_no_ext}_{det_name}"
        DS = DataSelection(filename=fln,
                           raw_data=exp_data)
        data_sets[fln] = DS

    logger.info("Data loading: channel data is loaded successfully.")

    if ('pos_data' in data_out) and ('pos_names' in data_out):
        if 'x_pos' in data_out['pos_names'] and 'y_pos' in data_out['pos_names']:
            p_dict = {}
            for v in ['x_pos', 'y_pos']:
                ind = data_out['pos_names'].index(v)
                p_dict[v] = data_out['pos_data'][ind, :, :]
            img_dict['positions'] = p_dict
            logger.info("Data loading: positions data are loaded successfully.")

    scaler_tmp = {}
    for i, v in enumerate(data_out['scaler_names']):
        scaler_tmp[v] = data_out['scaler_data'][:, :, i]
    img_dict[fname_no_ext+'_scaler'] = scaler_tmp
    logger.info("Data loading: scaler data are loaded successfully.")
    return img_dict, data_sets, fname, detector_name, scan_metadata


def retrieve_data_from_hdf_suitcase(fpath):
    """
    Retrieve data from suitcase part in hdf file.
    Data name is defined in config file.
    """
    data_dict = {}
    with h5py.File(fpath, 'r+') as f:
        other_data_list = [v for v in f.keys() if v != 'xrfmap']
        if len(other_data_list) > 0:
            f_hdr = f[other_data_list[0]].attrs['start']
            if not isinstance(f_hdr, six.string_types):
                f_hdr = f_hdr.decode('utf-8')
            start_doc = ast.literal_eval(f_hdr)
            other_data = f[other_data_list[0]+'/primary/data']

            if start_doc['beamline_id'] == 'HXN':
                current_dir = os.path.dirname(os.path.realpath(__file__))
                config_file = 'hxn_pv_config.json'
                config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
                with open(config_path, 'r') as json_data:
                    config_data = json.load(json_data)
                extra_list = config_data['other_list']
                fly_type = start_doc.get('fly_type', None)
                subscan_dims = start_doc.get('subscan_dims', None)

                if 'dimensions' in start_doc:
                    datashape = start_doc['dimensions']
                elif 'shape' in start_doc:
                    datashape = start_doc['shape']
                else:
                    logger.error('No dimension/shape is defined in hdr.start.')

                datashape = [datashape[1], datashape[0]]  # vertical first, then horizontal
                for k in extra_list:
                    # k = k.encode('utf-8')
                    if k not in other_data.keys():
                        continue
                    _v = np.array(other_data[k])
                    v = _v.reshape(datashape)
                    if fly_type in ('pyramid',):
                        # flip position the same as data flip on det counts
                        v = flip_data(v, subscan_dims=subscan_dims)
                    data_dict[k] = v
    return data_dict


def read_MAPS(working_directory,
              file_name, channel_num=1):
    # data_dict = OrderedDict()
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # Empty container for metadata
    mdata = ScanMetadataXRF()

    #  cut off bad point on the last position of the spectrum
    # bad_point_cut = 0

    fit_val = None
    fit_v_pyxrf = None

    file_path = os.path.join(working_directory, file_name)
    print('file path is {}'.format(file_path))

    with h5py.File(file_path, 'r+') as f:

        data = f['MAPS']
        fname = file_name.split('.')[0]

        #  for 2D MAP
        # data_dict[fname] = data

        # raw data
        exp_data = data['mca_arr'][:]

        # data from channel summed
        roi_channel = data['channel_names'].value
        roi_val = data['XRF_roi'][:]

        scaler_names = data['scaler_names'].value
        scaler_val = data['scalers'][:]

        try:
            # data from fit
            fit_val = data['XRF_fits'][:]
        except KeyError:
            logger.info('No fitting from MAPS can be loaded.')

        try:
            fit_data = f['xrfmap/detsum']
            fit_v_pyxrf = fit_data['xrf_fit'][:]
            fit_n_pyxrf = fit_data['xrf_fit_name'].value
            print(fit_n_pyxrf)
        except KeyError:
            logger.info('No fitting from pyxrf can be loaded.')

    # exp_shape = exp_data.shape
    exp_data = exp_data.T
    exp_data = np.rot90(exp_data, 1)
    logger.info('File : {} with total counts {}'.format(fname,
                                                        np.sum(exp_data)))
    DS = DataSelection(filename=fname,
                       raw_data=exp_data)
    data_sets.update({fname: DS})

    # save roi and fit into dict

    temp_roi = {}
    temp_fit = {}
    temp_scaler = {}
    temp_pos = {}

    for i, name in enumerate(roi_channel):
        temp_roi[name] = np.flipud(roi_val[i, :, :])
    img_dict[fname+'_roi'] = temp_roi

    if fit_val is not None:
        for i, name in enumerate(roi_channel):
            temp_fit[name] = fit_val[i, :, :]
        img_dict[fname+'_fit_MAPS'] = temp_fit

    cut_bad_col = 1
    if fit_v_pyxrf is not None:
        for i, name in enumerate(fit_n_pyxrf):
            temp_fit[name] = fit_v_pyxrf[i, :, cut_bad_col:]
        img_dict[fname+'_fit'] = temp_fit

    for i, name in enumerate(scaler_names):
        if name == 'x_coord':
            temp_pos['x_pos'] = np.flipud(scaler_val[i, :, :])
        elif name == 'y_coord':
            temp_pos['y_pos'] = np.flipud(scaler_val[i, :, :])
        else:
            temp_scaler[name] = np.flipud(scaler_val[i, :, :])
    img_dict[fname+'_scaler'] = temp_scaler
    img_dict['positions'] = temp_pos

    # read fitting results
    # if 'xrf_fit' in data[detID]:
    #     fit_result = get_fit_data(data[detID]['xrf_fit_name'].value,
    #                               data[detID]['xrf_fit'].value)
    #     img_dict.update({fname+'_fit': fit_result})

    return img_dict, data_sets, mdata


def get_roi_sum(namelist, data_range, data):
    data_temp = dict()
    for i in range(len(namelist)):
        lowv = data_range[i, 0]
        highv = data_range[i, 1]
        data_sum = np.sum(data[:, :, lowv: highv], axis=2)
        data_temp.update({namelist[i]: data_sum})
        # data_temp.update({namelist[i].replace(' ', '_'): data_sum})
    return data_temp


def get_fit_data(namelist, data):
    """
    Read fit data from h5 file. This is to be moved to filestore part.

    Parameters
    ---------
    namelist : list
        list of str for element lines
    data : array
        3D array of fitting results
    """
    data_temp = dict()
    for i, v in enumerate(namelist):
        if not isinstance(v, six.string_types):
            v = v.decode()
        data_temp.update({v: data[i, :, :]})
    return data_temp


def read_hdf_to_stitch(working_directory, filelist,
                       shape, ignore_file=None):
    """
    Read fitted results from each hdf file, and stitch them together.

    Parameters
    ----------
    working_directory : str
        folder with all the h5 files and also the place to save output
    filelist : list of str
        names for all the h5 files
    shape : list or tuple
        shape defines how to stitch all the h5 files. [veritcal, horizontal]
    ignore_file : list of str
        to be implemented

    Returns
    -------
    dict :
        combined results from each h5 file
    """
    out = {}
    # shape_v = {}
    horizontal_v = 0
    vertical_v = 0
    h_index = np.zeros(shape)
    v_index = np.zeros(shape)

    for i, file_name in enumerate(filelist):
        img, _ = read_hdf_APS(working_directory, file_name,
                              load_summed_data=False, load_each_channel=False)
        tmp_shape = img['positions']['x_pos'].shape
        m = i // shape[1]
        n = i % shape[1]

        if n == 0:
            h_step = 0

        h_index[m][n] = h_step
        v_index[m][n] = m * tmp_shape[0]
        h_step += tmp_shape[1]

        if i < shape[1]:
            horizontal_v += tmp_shape[1]
        if i % shape[1] == 0:
            vertical_v += tmp_shape[0]
        if i == 0:
            out = copy.deepcopy(img)

    data_tmp = np.zeros([vertical_v, horizontal_v])

    for k, v in six.iteritems(out):
        for m, n in six.iteritems(v):
            v[m] = np.array(data_tmp)

    for i, file_name in enumerate(filelist):
        img, _ = read_hdf_APS(working_directory, file_name,
                              load_summed_data=False, load_each_channel=False)

        tmp_shape = img['positions']['x_pos'].shape
        m = i // shape[1]
        n = i % shape[1]
        h_i = h_index[m][n]
        v_i = v_index[m][n]

        keylist = ['fit', 'scaler', 'position']

        for key_name in keylist:
            fit_key0, = [v for v in list(out.keys()) if key_name in v]
            fit_key, = [v for v in list(img.keys()) if key_name in v]
            for k, v in six.iteritems(img[fit_key]):
                out[fit_key0][k][v_i:v_i+tmp_shape[0], h_i:h_i+tmp_shape[1]] = img[fit_key][k]

    return out


def get_data_from_folder_helper(working_directory, foldername,
                                filename, flip_h=False):
    """
    Read fitted data from given folder.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    foldername : string
        folder name of given fitting result
    filename : string
        given element
    flip_h : bool
        x position is saved in a wrong way, so we may want to flip left right on the data,
        to be removed.

    Returns
    -------
    2D array
    """
    fpath = os.path.join(working_directory, foldername, filename)
    if 'txt' in filename:
        data = np.loadtxt(fpath)
    elif 'tif' in filename:
        data = np.array(Image.open(fpath))

    # x position is saved in a wrong way
    if flip_h is True:
        data = np.fliplr(data)
    return data


def get_data_from_multiple_folders_helper(working_directory, folderlist,
                                          filename, flip_h=False):
    """
    Read given element from fitted results in multiple folders.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    folderlist : list
        list of folder names saving fitting result
    filename : string
        given element
    flip_h : bool
        x position is saved in a wrong way, so we may want to flip left right on the data,
        to be removed.

    Returns
    -------
    2D array
    """
    output = np.array([])
    for foldername in folderlist:
        result = get_data_from_folder_helper(working_directory, foldername,
                                             filename, flip_h=flip_h)
        output = np.concatenate([output, result.ravel()])
    return output


def stitch_fitted_results(working_directory, folderlist, output=None):
    """
    Stitch fitted data from multiple folders. Output stiched results as 1D array.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    folderlist : list
        list of folder names saving fitting result
    output : string, optional
        output folder name to save all the stiched results.
    """

    # get all filenames
    fpath = os.path.join(working_directory, folderlist[0], '*')
    pathlist = [name for name in glob.glob(fpath)]
    filelist = [name.split(sep_v)[-1] for name in pathlist]
    out = {}
    for filename in filelist:
        if 'x_pos' in filename:
            flip_h = True
        else:
            flip_h = False
        data = get_data_from_multiple_folders_helper(working_directory, folderlist,
                                                     filename, flip_h=flip_h)
        out[filename.split('.')[0]] = data

    if output is not None:
        outfolder = os.path.join(working_directory, output)
        if os.path.exists(outfolder) is False:
            os.mkdir(outfolder)
        for k, v in out.items():
            outpath = os.path.join(outfolder, k+'_stitched.txt')
            np.savetxt(outpath, v)
    return out


def save_fitdata_to_hdf(fpath, data_dict,
                        datapath='xrfmap/detsum',
                        data_saveas='xrf_fit',
                        dataname_saveas='xrf_fit_name'):
    """
    Add fitting results to existing h5 file. This is to be moved to filestore.

    Parameters
    ----------
    fpath : str
        path of the hdf5 file
    data_dict : dict
        dict of array
    datapath : str
        path inside h5py file
    data_saveas : str, optional
        name in hdf for data array
    dataname_saveas : str, optional
        name list in hdf to explain what the saved data mean
    """
    f = h5py.File(fpath, 'a')
    try:
        dataGrp = f.create_group(datapath)
    except ValueError:
        dataGrp = f[datapath]

    data = []
    namelist = []
    for k, v in six.iteritems(data_dict):
        if not isinstance(k, six.string_types):
            k = k.decode()
        namelist.append(k)
        data.append(v)

    if data_saveas in dataGrp:
        del dataGrp[data_saveas]

    data = np.asarray(data)
    ds_data = dataGrp.create_dataset(data_saveas, data=data)
    ds_data.attrs['comments'] = ' '

    if dataname_saveas in dataGrp:
        del dataGrp[dataname_saveas]

    if not isinstance(dataname_saveas, six.string_types):
        dataname_saveas = dataname_saveas.decode()
    namelist = np.array(namelist).astype('|S9')
    name_data = dataGrp.create_dataset(dataname_saveas, data=namelist)
    name_data.attrs['comments'] = ' '

    f.close()


def export_to_view(fpath, output_name=None, output_folder='', namelist=None):
    """
    Output fitted data to tablet data for visulization.

    Parameters
    ----------
    fpath : str
        input file path, file is pyxrf h5 file
    output_name : str
        output file name
    otuput_folder : str, optional
        default as current working folder
    namelist : list, optional
        list of elemental names
    """
    with h5py.File(fpath, 'r') as f:
        d = f['xrfmap/detsum/xrf_fit'][:]
        d = d.reshape([d.shape[0], -1])
        elementlist = f['xrfmap/detsum/xrf_fit_name'][:]
        elementlist = helper_decode_list(elementlist)

        xy = f['xrfmap/positions/pos'][:]
        xy = xy.reshape([xy.shape[0], -1])
        xy_name = ['X', 'Y']

        names = xy_name + elementlist
        data = np.concatenate((xy, d), axis=0)

    data_dict = OrderedDict()
    if namelist is None:
        for i, k in enumerate(names):
            if 'Userpeak' in k or 'r2_adjust' in k:
                continue
            data_dict.update({k: data[i, :]})
    else:
        for i, k in enumerate(names):
            if k in namelist or k in xy_name:
                data_dict.update({k: data[i, :]})

    df = pd.DataFrame(data_dict)
    if output_name is None:
        fname = fpath.split(sep_v)[-1]
        output_name = fname.split('.')[0] + '_fit_view.csv'

    outpath = os.path.join(output_folder, output_name)
    print('{} is created.'.format(outpath))
    df.to_csv(outpath, index=False)


def get_header(fname):
    """
    helper function to extract header in spec file.
    .. warning :: This function works fine for spec file format
    from Canadian light source. Others may need to be tested.

    Parameters
    ----------
    fname : spec file name
    """
    mydata = []
    with open(fname, 'r') as f:
        for v in f:   # iterate the file
            mydata.append(v)
            _sign = '#'
            _sign = _sign.encode('utf-8')
            if _sign not in v:
                break
    header_line = mydata[-2]  # last line is space
    n = [v.strip() for v in header_line[1:].split('\t') if v.strip() != '']
    return n


def combine_data_to_recon(element_list, datalist, working_dir, norm=True,
                          file_prefix='scan2D_', ic_name='sclr1_ch4',
                          expand_r=2, internal_path='xrfmap/detsum'):
    """
    Combine 2D data to 3D array for reconstruction.

    Parameters
    ----------
    element_list : list
        list of elements
    datalist : list
        list of run number
    working_dir : str
    norm : bool, optional
        normalization or not
    file_prefix : str, optional
        prefix name for h5 file
    ic_name : str
        ion chamber name for normalization
    expand_r: int
        expand initial array to a larger size to include each 2D image easily,
        as each 2D image may have different size. Crop the 3D array back to a proper size in the end.
    internal_path : str, optional
        inside path to get fitting data in h5 file

    Returns
    -------
    dict of 3d array with each array's shape like [num_sequences, num_row, num_col]
    """
    element3d = {}
    for element_name in element_list:
        element3d[element_name] = None

    max_h = 0
    max_v = 0
    for i, v in enumerate(datalist):
        filename = file_prefix+str(v)+'.h5'
        filepath = os.path.join(working_dir, filename)
        with h5py.File(filepath, 'r+') as f:
            dataset = f[internal_path]
            try:
                data_all = dataset['xrf_fit'].value
                data_name = dataset['xrf_fit_name'].value
                data_name = helper_decode_list(data_name)
            except KeyError:
                print('Need to do fitting first.')
            scaler_dataset = f['xrfmap/scalers']
            scaler_v = scaler_dataset['val'].value
            scaler_n = scaler_dataset['name'].value
            scaler_n = helper_decode_list(scaler_n)

        data_dict = {}
        for name_i, name_v in enumerate(data_name):
            data_dict[name_v] = data_all[name_i, :, :]
        if norm is True:
            scaler_dict = {}
            for s_i, s_v in enumerate(scaler_n):
                scaler_dict[s_v] = scaler_v[:, :, s_i]

        for element_name in element_list:
            data = data_dict[element_name]
            if norm is True:
                normv = scaler_dict[ic_name]
                data = data/normv
            if element3d[element_name] is None:
                element3d[element_name] = np.zeros(
                    [len(datalist),
                     data.shape[0]*expand_r,
                     data.shape[1]*expand_r])
            element3d[element_name][i, :data.shape[0], :data.shape[1]] = data

        max_h = max(max_h, data.shape[0])
        max_v = max(max_v, data.shape[1])

    for k, v in element3d.items():
        element3d[k] = v[:, :max_h, :max_v]
    return element3d


def h5file_for_recon(element_dict, angle, runid=None, filename=None):
    """
    Save fitted 3d elemental data into h5 file for reconstruction use.

    Parameters
    ----------
    element_dict : dict
        elements 3d data after normalization
    angle : list
        angle information
    runid : list or optional
        run ID
    filename : str
    """

    if filename is None:
        filename = 'xrf3d.h5'
    with h5py.File(filename) as f:
        d_group = f.create_group('element_data')
        for k, v in element_dict.items():
            sub_g = d_group.create_group(k)
            sub_g.create_dataset('data', data=np.asarray(v),
                                 compression='gzip')
            sub_g.attrs['comments'] = 'normalized fluorescence data for {}'.format(k)
        angle_g = f.create_group('angle')
        angle_g.create_dataset('data', data=np.asarray(angle))
        angle_g.attrs['comments'] = 'angle information'
        if runid is not None:
            runid_g = f.create_group('runid')
            runid_g.create_dataset('data', data=np.asarray(runid))
            runid_g.attrs['comments'] = 'run id information'


def create_movie(data, fname='demo.mp4', dpi=100, cmap='jet',
                 clim=None, fig_size=(6, 8), fps=20, data_power=1, angle=None, runid=None):
    """
    Transfer 3d array into a movie.

    Parameters
    ----------
    data : 3d array
        data shape is [num_sequences, num_row, num_col]
    fname : string, optional
        name to save movie
    dpi : int, optional
        resolution of the movie
    cmap : string, optional
        color format
    clim : list, tuple, optional
        [low, high] value to define plotting range
    fig_size : list, tuple, optional
        size (horizontal size, vertical size) of each plot
    fps : int, optional
        frame per second
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(np.zeros([data.shape[1], data.shape[2]]),
                   cmap=cmap, interpolation='nearest')

    fig.set_size_inches(fig_size)
    fig.tight_layout()

    def update_img(n):
        tmp = data[n, :, :]
        im.set_data(tmp**data_power)
        if clim is not None:
            im.set_clim(clim)
        else:
            im.set_clim([0, np.max(data[n, :, :])])
        figname = ''
        if runid is not None:
            figname = 'runid: {} '.format(runid[n])
        if angle is not None:
            figname += 'angle: {}'.format(angle[n])
        # if len(figname) != 0:
        #     im.ax.set_title(figname)
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, data.shape[0], interval=30)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(fname, writer=writer, dpi=dpi)


def spec_to_hdf(wd, spec_file, spectrum_file, output_file, img_shape,
                ic_name=None, x_name=None, y_name=None):
    """
    Transform spec data to hdf file pyxrf can take. Using this function, users need to
    have two input files ready, sepc_file and spectrum_file, with explanation as below.

    .. warning :: This function should be better defined to take care spec file in general.
    The work in suitcase should also be considered. This function works fine for spec file format
    from Canadian light source. Others may need to be tested.

    Parameters
    ----------
    wd : str
        working directory for spec file, and created hdf
    spec_file : str
        spec txt data file
    spectrum_file : str
        fluorescence spectrum data file
    output_file : str
        the output h5 file for pyxrf
    img_shape : list or array
        the shape of two D scan, [num of row, num of column]
    ic_name : str
        the name of ion chamber for normalization, listed in spec file
    x_name : str
        x position name, listed in spec file
    y_name : str
        y position name, listed in spec file
    """
    # read scaler data from spec file
    spec_path = os.path.join(wd, spec_file)
    h = get_header(spec_path)
    spec_data = pd.read_csv(spec_path, names=h, sep='\t', comment='#', index_col=False)

    if ic_name is not None:
        scaler_name = [str(ic_name)]
        scaler_val = spec_data[scaler_name].values
        scaler_val = scaler_val.reshape(img_shape)
        scaler_data = np.zeros([img_shape[0], img_shape[1], 1])
        scaler_data[:, :, 0] = scaler_val

    if x_name is not None and y_name is not None:
        xy_data = np.zeros([2, img_shape[0], img_shape[1]])
        xy_data[0, :, :] = spec_data[x_name].values.reshape(img_shape)
        xy_data[1, :, :] = spec_data[y_name].values.reshape(img_shape)
        xy_name = ['x_pos', 'y_pos']

    spectrum_path = os.path.join(wd, spectrum_file)
    sum_data0 = np.loadtxt(spectrum_path)
    sum_data = np.reshape(sum_data0, [sum_data0.shape[0], img_shape[0], img_shape[1]])
    sum_data = np.transpose(sum_data, axes=(1, 2, 0))

    interpath = 'xrfmap'

    fpath = os.path.join(wd, output_file)
    with h5py.File(fpath) as f:
        dataGrp = f.create_group(interpath+'/detsum')
        ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
        ds_data.attrs['comments'] = 'Experimental data from channel sum'

        if ic_name is not None:
            dataGrp = f.create_group(interpath+'/scalers')
            dataGrp.create_dataset('name', data=helper_encode_list(scaler_name))
            dataGrp.create_dataset('val', data=scaler_data)

        if x_name is not None and y_name is not None:
            dataGrp = f.create_group(interpath+'/positions')
            dataGrp.create_dataset('name', data=helper_encode_list(xy_name))
            dataGrp.create_dataset('pos', data=xy_data)


def make_hdf_stitched(working_directory, filelist, fname,
                      shape):
    """
    Read fitted results from each hdf file, stitch them together and save to
    a new h5 file.

    Parameters
    ----------
    working_directory : str
        folder with all the h5 files and also the place to save output
    filelist : list of str
        names for all the h5 files
    fname : str
        name of output h5 file
    shape : list or tuple
        shape defines how to stitch all the h5 files. [veritcal, horizontal]
    """
    print('Reading data from each hdf file.')
    fpath = os.path.join(working_directory, fname)
    out = read_hdf_to_stitch(working_directory, filelist, shape)

    result = {}
    img_shape = None
    for k, v in six.iteritems(out):
        for m, n in six.iteritems(v):
            if img_shape is None:
                img_shape = n.shape
            result[m] = n.ravel()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = 'srx_pv_config.json'
    config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)

    print('Saving all the data into one hdf file.')
    write_db_to_hdf(fpath, result,
                    img_shape,
                    det_list=config_data['xrf_detector'],
                    pos_list=('x_pos', 'y_pos'),
                    scaler_list=config_data['scaler_list'],
                    base_val=config_data['base_value'])  # base value shift for ic

    fitkey, = [v for v in list(out.keys()) if 'fit' in v]
    save_fitdata_to_hdf(fpath, out[fitkey])

    print('Done!')
