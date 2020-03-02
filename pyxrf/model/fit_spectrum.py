from __future__ import absolute_import

import numpy as np
import time
import copy
import six
import os
import re
import math
from collections import OrderedDict, deque
import multiprocessing
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import lmfit

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool
from skbeam.core.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                           # sum_area,
                                           set_parameter_bound,
                                           # ParamController,
                                           K_LINE, L_LINE, M_LINE,
                                           nnls_fit, trim, construct_linear_model,
                                           # linear_spectrum_fitting,
                                           register_strategy, TRANSITIONS_LOOKUP)
from skbeam.core.fitting.background import snip_method
from skbeam.fluorescence import XrfElement as Element
from .guessparam import (calculate_profile, fit_strategy_list,
                         trim_escape_peak, define_range, get_energy,
                         get_Z, PreFitStatus, ElementController,
                         update_param_from_element)
from .fileio import save_fitdata_to_hdf, output_data

from ..core.utils import (gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma,
                          gaussian_max_to_area, gaussian_area_to_max)

from ..core.quant_analysis import ParamQuantEstimation

import logging
logger = logging.getLogger()


class Fit1D(Atom):
    """
    Fit 1D fluorescence spectrum. Users can choose multiple strategies
    for this fitting.

    Parameters
    ----------
    x_data : 2D array
        x position data from hdf file
    y_data : 2D array
        y position data from hdf file
    result_map : Dict
        dict of 2D array map for each fitted element
    map_interpolation : bool
        option to interpolate the 2D map according to x,y position or not
        Interpolation is performed only before exporting data
        (currently .tiff or .txt)
    hdf_path : str
        path to hdf file
    hdf_name : str
        name of hdf file
    """
    file_status = Str()
    default_parameters = Dict()
    param_dict = Dict()

    img_dict = Dict()
    data_sets = Typed(OrderedDict)

    element_list = List()
    data_sets = Dict()
    data_all = Typed(np.ndarray)
    data = Typed(np.ndarray)
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    residual = Typed(np.ndarray)
    comps = Dict()
    fit_strategy1 = Int(0)
    fit_strategy2 = Int(0)
    fit_strategy3 = Int(0)
    fit_strategy4 = Int(0)
    fit_strategy5 = Int(0)
    fit_result = Typed(object)
    data_title = Str()
    runid = Int(0)

    working_directory = Str()
    result_folder = Str()

    all_strategy = Typed(object)

    x0 = Typed(np.ndarray)
    y0 = Typed(np.ndarray)
    bg = Typed(np.ndarray)
    es_peak = Typed(np.ndarray)
    cal_x = Typed(np.ndarray)
    cal_y = Typed(np.ndarray)
    cal_spectrum = Dict()

    #  attributes used by the ElementEdit window
    selected_element = Str()
    selected_index = Int()
    elementinfo_list = List()

    # The variable contains the image title displayed in "Element Map' tab.
    #   The variable is synchronized with the identical variable in 'DrawImageAdvanced' class.
    img_title = Str()
    # Reference to the dictionary that contains dataset currently displayed in 'Element Map' tab
    #   The variable is synchronized with the identical variable in 'DrawImageAdvanced' class
    dict_to_plot = Dict()

    # The variable is replicating the identical variable in 'DrawImageAdvanced' class
    #   True - quantitative normalization is ON, False - OFF
    quantitative_normalization = Bool()
    # The reference to the object holding parameters for quantitative normalization
    #   The variable is synchronized to the identical variable in 'DrawImageAdvanced' class
    param_quant_analysis = Typed(object)

    function_num = Int(0)
    nvar = Int(0)
    chi2 = Float(0.0)
    red_chi2 = Float(0.0)
    r2 = Float(0.0)
    global_param_list = List()

    fit_num = Int(100)
    ftol = Float(1e-5)
    c_weight = Float(1e2)

    fit_img = Dict()

    save_point = Bool(False)
    point1v = Int(0)
    point1h = Int(0)
    point2v = Int(0)
    point2h = Int(0)

    EC = Typed(object)
    result_dict_names = List()
    e_name = Str()
    add_element_intensity = Float(100.0)
    pileup_data = Dict()

    raise_bg = Float(0.0)
    pixel_bin = Int(0)
    linear_bg = Bool(False)
    use_snip = Bool(True)
    bin_energy = Int(0)
    fit_info = Str()
    pixel_fit_info = Str()

    pixel_fit_method = Int(0)
    param_q = Typed(object)

    result_map = Dict()
    map_interpolation = Bool(False)
    hdf_path = Str()
    hdf_name = Str()

    roi_sum_opt = Dict()
    scaler_keys = List()
    scaler_index = Int(0)

    # Reference to GuessParamModel object
    param_model = Typed(object)

    # Reference to FileIOMOdel
    io_model = Typed(object)

    # Fields for updating user defined peak parameters
    add_userpeak_energy = Float(0.0)
    add_userpeak_fwhm = Float(0.0)
    # Copies of the variables that hold old value during update
    add_userpeak_fwhm_old = Float(0.0)
    # The names for the respective parameters
    #   (used to access parameters in
    #   self.param_model.param_dict)
    name_userpeak_dcenter = Str()
    name_userpeak_dsigma = Str()
    name_userpeak_area = Str()

    # Quantitative analysis: used during estimation step
    param_quant_estimation = ParamQuantEstimation()
    # *** The following two references are used exclusively to update the list of standards ***
    # ***   in the dialog box. ***
    qe_param_built_in_ref = Typed(object)
    qe_param_custom_ref = Typed(object)
    # The following reference used to track the selected standard while the selection
    #   dialog box is open. Once the dialog box is opened again, the reference becomes
    #   invalid, since the descriptions of the standards are reloaded from files.
    qe_standard_selected_ref = Typed(object)
    # Keep the actual copy of the selected standard. The copy is used to keep information
    #   on the selected standard while descriptions are reloaded (references become invalid).
    qe_standard_selected_copy = Typed(object)
    # *** The following fields are used exclusively to store input values ***
    # ***   for the 'SaveQuantCalibration' dialog box. ***
    # *** The fields are not guaranteed to have valid values at any other time. ***
    qe_standard_path_name = Str()
    qe_standard_file_name = Str()
    qe_standard_distance = Str("0.0")
    qe_standard_overwrite_existing = Bool(False)
    qe_standard_data_preview = Str()

    def __init__(self, param_model, io_model, *args, **kwargs):
        self.working_directory = kwargs['working_directory']
        self.result_folder = kwargs['working_directory']
        self.default_parameters = kwargs['default_parameters']
        self.param_dict = copy.deepcopy(self.default_parameters)
        self.param_q = deque()
        self.all_strategy = OrderedDict()

        # Reference to GuessParamModel object
        self.param_model = param_model

        # Reference to FileIOMOdel
        self.io_model = io_model

        self.EC = ElementController()
        self.pileup_data = {'element1': 'Si_K',
                            'element2': 'Si_K',
                            'intensity': 100.0}

        # don't argue, plotting purposes
        self.fit_strategy1 = 0
        self.fit_strategy2 = 0
        self.fit_strategy1 = 1
        self.fit_strategy2 = 0

        # perform roi sum in given range
        self.roi_sum_opt['status'] = False
        self.roi_sum_opt['low'] = 0.0
        self.roi_sum_opt['high'] = 10.0

        self.qe_standard_selected_ref = None
        self.qe_standard_selected_copy = None

    def result_folder_changed(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.result_folder = change['value']

    def data_title_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_title = change['value']

    def runid_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.runid = change['value']

    def img_dict_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        change : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.img_dict = change['value']
        _key = [k for k in self.img_dict.keys() if 'scaler' in k]
        if len(_key) != 0:
            self.scaler_keys = sorted(self.img_dict[_key[0]].keys())

    def data_sets_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup
        """
        self.data_sets = change['value']

    def scaler_index_update(self, change):
        """
        Observer function to be connected to the fit_model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.scaler_index = change['value']

    def img_title_update(self, change):
        r"""Observer function. Sets ``img_title`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.img_title = change['value']

    def dict_to_plot_update(self, change):
        r"""Observer function. Sets ``dict_to_plot`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.dict_to_plot = change['value']

    def quantitative_normalization_update(self, change):
        r"""Observer function. Sets ``quantitative_normalization`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.quantitative_normalization = change['value']

    def param_quant_analysis_update(self, change):
        r"""Observer function. Sets ``param_quant_analysis`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.param_quant_analysis = change['value']

    def energy_bound_high_update(self, change):
        """
        Observer function that connects 'param_model' (GuessParamModel)
        attribute 'energy_bound_high_buf' with the respective
        value in 'self.param_dict'
        """
        self.param_dict['non_fitting_values']['energy_bound_high']['value'] = change['value']

    def energy_bound_low_update(self, change):
        """
        Observer function that connects 'param_model' (GuessParamModel)
        attribute 'energy_bound_low_buf' with the respective
        value in 'self.param_dict'
        """
        self.param_dict['non_fitting_values']['energy_bound_low']['value'] = change['value']

    def update_selected_index(self, selected_element=None,
                              element_list_new=None):

        if selected_element is None:
            # Currently selected element
            element = self.selected_element
        else:
            # Selected element (probably saved before update of the element list)
            element = selected_element

        if element_list_new is None:
            # Current element list
            element_list = self.element_list
        else:
            # Future element list (before it is updated)
            element_list = element_list_new

        if not element_list:
            # Empty element list
            ind = 0
        else:
            try:
                # Combo-box has additional element 'Select element'
                ind = element_list.index(element) + 1
            except ValueError:
                # Element is not found (was deleted), so deselect the element
                ind = 0
        if ind == self.selected_index:
            # We want the content to update (deselect, then select again)
            self.selected_index = 0
        self.selected_index = ind

    @observe('selected_index')
    def _selected_element_changed(self, change):
        if change['value'] > 0:
            ind_sel = change['value'] - 1
            if ind_sel >= len(self.element_list):
                ind_sel = len(self.element_list) - 1
                self.selected_index = ind_sel + 1  # Change the selection as well
            self.selected_element = self.element_list[ind_sel]
            if len(self.selected_element) <= 4:
                element = self.selected_element.split('_')[0]
                self.elementinfo_list = sorted([e for e in list(self.param_dict.keys())
                                                if (element+'_' in e) and  # error between S_k or Si_k
                                                ('pileup' not in e)])  # Si_ka1 not Si_K
                logger.info(f"Element line info: {self.elementinfo_list}")
            else:
                element = self.selected_element  # for pileup peaks
                self.elementinfo_list = sorted([e for e in list(self.param_dict.keys())
                                                if element.replace('-', '_') in e])
                logger.info(f"User defined or pileup peak info: {self.elementinfo_list}")
        else:
            self.elementinfo_list = []

    @observe('qe_standard_distance')
    def _qe_standard_distance_changed(self, change):
        try:
            d = float(change["value"])
        except Exception:
            d = None
        if d <= 0.0:
            d = None
        self.param_quant_estimation.set_distance_to_sample_in_data_dict(distance_to_sample=d)
        # Change preview if distance value changed
        self.qe_standard_data_preview = \
            self.param_quant_estimation.get_fluorescence_data_dict_text_preview()

    def get_qe_standard_distance_as_float(self):
        r"""Return distance from sample as positive float or None"""
        try:
            d = float(self.qe_standard_distance)
        except Exception:
            d = None
        if (d is not None) and (d <= 0.0):
            d = None
        return d

    def _compute_fwhm_base(self):
        # Computes 'sigma' value based on default parameters and peak energy (for Userpeaks)
        #   does not include corrections for fwhm
        # If both peak center (energy) and fwhm is updated, energy needs to be set first,
        #   since it is used in computation of ``fwhm_base``

        sigma = gaussian_fwhm_to_sigma(self.param_model.default_parameters["fwhm_offset"]["value"])

        sigma_sqr = self.param_dict[self.name_userpeak_dcenter]["value"] + 5.0  # center
        sigma_sqr *= self.param_model.default_parameters["non_fitting_values"]["epsilon"]  # epsilon
        sigma_sqr *= self.param_model.default_parameters["fwhm_fanoprime"]["value"]  # fanoprime
        sigma_sqr += sigma * sigma  # We have computed the expression under sqrt

        sigma_total = np.sqrt(sigma_sqr)

        return gaussian_sigma_to_fwhm(sigma_total)

    def select_index_by_eline_name(self, eline_name):
        # Select the element by name. If element is selected, then the ``elementinfo_list`` with
        #   names of parameters is created. Originally the element could only be selected
        #   by its index in the list (from dialog box ``ElementEdit``. This function is
        #   used by interface components for editing parameters of user defined peaks.
        if eline_name in self.element_list:
            # This will fill the list ``self.elementinfo_list``
            self.selected_index = self.element_list.index(eline_name) + 1
            if "Userpeak" in eline_name:
                names = [name for name in self.elementinfo_list if "_delta_center" in name]
                if names:
                    self.name_userpeak_dcenter = names[0]
                else:
                    self.name_userpeak_dcenter = ""

                names = [name for name in self.elementinfo_list if "_delta_sigma" in name]
                if names:
                    self.name_userpeak_dsigma = names[0]
                else:
                    self.name_userpeak_dsigma = ""

                names = [name for name in self.elementinfo_list if "_area" in name]
                if names:
                    self.name_userpeak_area = names[0]
                else:
                    self.name_userpeak_area = ""

                if self.name_userpeak_dcenter and self.name_userpeak_dsigma:
                    # Userpeak always has energy of 5.0 keV, the user can set only the offset
                    #   This is the internal representation, but we must display and let the user
                    #   enter the true value of energy
                    self.add_userpeak_energy = \
                        self.param_dict[self.name_userpeak_dcenter]["value"] + 5.0
                    # Same with FWHM for the user defined peak.
                    #   Also, sigma must be converted to FWHM: FWHM = 2.355 * sigma
                    self.add_userpeak_fwhm = \
                        gaussian_sigma_to_fwhm(self.param_dict[self.name_userpeak_dsigma]["value"]) + \
                        self._compute_fwhm_base()

                    # Create copies (before rounding)
                    self.add_userpeak_fwhm_old = self.add_userpeak_fwhm

                    # Adjust formatting (5 digits after dot is sufficient)
                    self.add_userpeak_energy = float(f"{self.add_userpeak_energy:.5f}")
                    self.add_userpeak_fwhm = float(f"{self.add_userpeak_fwhm:.5f}")

        else:
            raise Exception(f"Line '{eline_name}' is not in the list of selected element lines.")

    def _update_userpeak_energy(self):

        # According to the accepted peak model, as energy of the peak center grows,
        #   the peak becomes wider. The most user friendly solution is to automatically
        #   increase FWHM as the peak moves along the energy axis to the right and
        #   decrease otherwise. So generally, the user should first place the peak
        #   center at the desired energy, and then adjust FWHM.

        # We change energy, so we will have to change FWHM as well
        #  so before updating energy we will save the difference between
        #  the default (base) FWHM and the displayed FWHM
        fwhm_difference = self.add_userpeak_fwhm - self._compute_fwhm_base()

        # Now we change energy.
        energy = self.add_userpeak_energy - 5.0

        v_center = self.param_dict[self.name_userpeak_dcenter]["value"]
        v_max = self.param_dict[self.name_userpeak_dcenter]["max"]
        v_min = self.param_dict[self.name_userpeak_dcenter]["min"]
        # Keep the possible range for value change the same
        self.param_dict[self.name_userpeak_dcenter]["value"] = energy
        self.param_dict[self.name_userpeak_dcenter]["max"] = energy + v_max - v_center
        self.param_dict[self.name_userpeak_dcenter]["min"] = energy - (v_center - v_min)

        # The base value is updated now (since the energy has changed)
        fwhm_base = self._compute_fwhm_base()
        fwhm = fwhm_difference + fwhm_base
        # Also adjust precision, so that the new value fits the input field
        fwhm = float(f"{fwhm:.5f}")

        # Finally update the displayed 'fwhm'. It will be saved to ``param_dict`` later.
        self.add_userpeak_fwhm = fwhm

    def _update_userpeak_fwhm(self):

        fwhm_base = self._compute_fwhm_base()
        fwhm = self.add_userpeak_fwhm - fwhm_base

        sigma = gaussian_fwhm_to_sigma(fwhm)

        v_center = self.param_dict[self.name_userpeak_dsigma]["value"]
        v_max = self.param_dict[self.name_userpeak_dsigma]["max"]
        v_min = self.param_dict[self.name_userpeak_dsigma]["min"]
        # Keep the possible range for value change the same
        self.param_dict[self.name_userpeak_dsigma]["value"] = sigma
        self.param_dict[self.name_userpeak_dsigma]["max"] = sigma + v_max - v_center
        self.param_dict[self.name_userpeak_dsigma]["min"] = sigma - (v_center - v_min)

    def update_userpeak(self):
        # Update current user peak. Called when 'Update peak' button is pressed.

        # Some checks of the input values
        if self.add_userpeak_energy <= 0.0:
            logger.warning("User peak energy must be a positive number greater than 0.001.")
            return
        if self.add_userpeak_fwhm <= 0:
            logger.warning("User peak FWHM must be a positive number.")
            return

        # Make sure that the energy of the user peak is within the selected fitting range
        energy_bound_high = \
            self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        energy_bound_low = \
            self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        self.add_userpeak_energy = np.clip(self.add_userpeak_energy, energy_bound_low, energy_bound_high)

        # Ensure, that the values are greater than some small value to ensure that
        #   there is no computational problems.
        # Energy resolution for the existing beamlines is 0.01 keV, so 0.001 is small
        #   enough both for center energy and FWHM.
        energy_small_value = 0.001
        self.add_userpeak_energy = max(self.add_userpeak_energy, energy_small_value)
        self.add_userpeak_fwhm = max(self.add_userpeak_fwhm, energy_small_value)

        self._update_userpeak_energy()
        self._update_userpeak_fwhm()

        # Find and save the value of peak maximum. Restore the maximum after FWHM is changed.
        # Note, that ``peak_sigma`` and ``peak_area`` may change if energy changes,
        #   but ``peak_max`` must remain the same (for better visual presentation)
        peak_sigma = gaussian_fwhm_to_sigma(self.add_userpeak_fwhm_old)
        peak_area = self.param_dict[self.name_userpeak_area]["value"]
        peak_max = gaussian_area_to_max(peak_area, peak_sigma)  # Keep this value

        # Restore peak height by adjusting the area (for new fwhm)
        peak_sigma = gaussian_fwhm_to_sigma(self.add_userpeak_fwhm)
        peak_area = gaussian_max_to_area(peak_max, peak_sigma)
        self.param_dict[self.name_userpeak_area]["value"] = peak_area

        # Create copies
        self.add_userpeak_fwhm_old = self.add_userpeak_fwhm

        logger.debug(f"The parameters of the user defined peak. The new values:\n"
                     f"          Energy: {self.add_userpeak_energy} keV\n"
                     f"          FWHM: {self.add_userpeak_fwhm} keV")

    def update_userpeak_controls(self):
        """
        The function should be called right after adding a userpeak to update the fields
        for userpeak energy and fwhm. Uses data for the currently selected Userpeak
        (the peak should be selected at the time of creation!!!)
        """
        if (self.name_userpeak_dcenter not in self.param_dict) or \
                (self.name_userpeak_dsigma not in self.param_dict):
            return
        # Set energy
        v_center = self.param_dict[self.name_userpeak_dcenter]["value"]
        v_energy = v_center + 5.0
        v_energy = round(v_energy, 3)
        self.add_userpeak_energy = v_energy
        # Set fwhm
        fwhm_base = self._compute_fwhm_base()
        v_dsigma = self.param_dict[self.name_userpeak_dsigma]["value"]
        v_dfwhm = gaussian_sigma_to_fwhm(v_dsigma)
        v_fwhm = v_dfwhm + fwhm_base
        v_fwhm = round(v_fwhm, 5)
        self.add_userpeak_fwhm = v_fwhm

    def keep_size(self):
        """Keep the size of deque as 2.
        """
        while len(self.param_q) > 2:
            self.param_q.popleft()

    def read_param_from_file(self, param_path):
        """
        Update parameters if new param_path is given.

        Parameters
        ----------
        param_path : str
            path to save the file
        """
        with open(param_path, 'r') as json_data:
            self.default_parameters = json.load(json_data)

        #  use queue to save the status of parameters
        self.param_q.append(copy.deepcopy(self.default_parameters))
        self.keep_size()

    def update_default_param(self, param):
        """assigan new values to default param.

        Parameters
        ----------
        param : dict
        """
        self.default_parameters = copy.deepcopy(param)
        #  use queue to save the status of parameters
        self.param_q.append(copy.deepcopy(self.default_parameters))
        self.keep_size()

    def apply_default_param(self):
        """
        Update param_dict with default parameters, also update element list.
        """
        # Save currently selected element name
        selected_element = self.selected_element
        self.selected_index = 0

        element_list = self.default_parameters['non_fitting_values']['element_list']
        element_list = [e.strip(' ') for e in element_list.split(',')]
        element_list = [_ for _ in element_list if _]  # Get rid of empty strings in the list
        self.element_list = element_list

        self.param_dict = copy.deepcopy(self.default_parameters)

        # show the list of elements on add/remove window
        self.EC.delete_all()
        self.create_EC_list(self.element_list)
        self.update_name_list()

        # Update the index in case the selected emission line disappeared from the list
        self.update_selected_index(selected_element=selected_element,
                                   element_list_new=element_list)

        # global parameters
        # for GUI purpose only
        # if we do not clear the list first, there is not update on the GUI
        self.global_param_list = []
        self.global_param_list = sorted([k for k in six.iterkeys(self.param_dict)
                                         if k == k.lower() and k != 'non_fitting_values'])

        self.define_range()

        #  register the strategy and extend the parameter list
        #  to cover all given elements
        # for strat_name in fit_strategy_list:
        #     strategy = extract_strategy(self.param_dict, strat_name)
        #     #  register the strategy and extend the parameter list
        #     #  to cover all given elements
        #     register_strategy(strat_name, strategy)
        #     set_parameter_bound(self.param_dict, strat_name)

        #  define element_adjust as fixed
        # self.param_dict = define_param_bound_type(self.param_dict)

    def exp_data_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data = np.asarray(change['value'])

    def exp_data_all_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_all = np.asarray(change['value'])

    def filename_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.hdf_name = change['value']
        # output to .h5 file
        self.hdf_path = os.path.join(self.result_folder, self.hdf_name)

    @observe('fit_strategy1')
    def update_strategy1(self, change):
        self.all_strategy.update({'strategy1': change['value']})
        if change['value']:
            logger.info('Strategy at step 1 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy2')
    def update_strategy2(self, change):
        self.all_strategy.update({'strategy2': change['value']})
        if change['value']:
            logger.info('Strategy at step 2 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy3')
    def update_strategy3(self, change):
        self.all_strategy.update({'strategy3': change['value']})
        if change['value']:
            logger.info('Strategy at step 3 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy4')
    def update_strategy4(self, change):
        self.all_strategy.update({'strategy4': change['value']})
        if change['value']:
            logger.info('Strategy at step 4 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    @observe('fit_strategy5')
    def update_strategy5(self, change):
        self.all_strategy.update({'strategy5': change['value']})
        if change['value']:
            logger.info('Strategy at step 5 is: {}'.
                        format(fit_strategy_list[change['value']-1]))

    def update_param_with_result(self):
        update_parameter_dict(self.param_dict, self.fit_result)

    def define_range(self):
        """
        Cut x range according to values define in param_dict.
        """
        lowv = self.param_dict['non_fitting_values']['energy_bound_low']['value']
        highv = self.param_dict['non_fitting_values']['energy_bound_high']['value']
        self.x0, self.y0 = define_range(self.data, lowv, highv,
                                        self.param_dict['e_offset']['value'],
                                        self.param_dict['e_linear']['value'])

    def get_background(self):
        self.bg = snip_method(self.y0,
                              self.param_dict['e_offset']['value'],
                              self.param_dict['e_linear']['value'],
                              self.param_dict['e_quadratic']['value'],
                              width=self.param_dict['non_fitting_values']['background_width'])

    def get_profile(self):
        """
        Calculate profile based on current parameters.
        """
        # self.define_range()

        #  Do nothing if no data is loaded
        if self.x0 is None or self.y0 is None:
            return

        self.cal_x, self.cal_spectrum, area_dict = calculate_profile(self.x0,
                                                                     self.y0,
                                                                     self.param_dict,
                                                                     self.element_list)
        #  add escape peak
        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.cal_spectrum['escape'] = trim_escape_peak(self.data,
                                                           self.param_dict,
                                                           len(self.y0))

        self.cal_y = np.zeros(len(self.cal_x))
        for k, v in six.iteritems(self.cal_spectrum):
            self.cal_y += v

        self.residual = self.cal_y - self.y0

    def fit_data(self, x0, y0):
        fit_num = self.fit_num
        ftol = self.ftol
        c_weight = 1  # avoid zero point
        MS = ModelSpectrum(self.param_dict, self.element_list)
        MS.assemble_models()

        # weights = 1/(c_weight + np.abs(y0))
        weights = 1/np.sqrt(c_weight + np.abs(y0))
        # weights /= np.sum(weights)
        result = MS.model_fit(x0, y0,
                              weights=weights,
                              maxfev=fit_num,
                              xtol=ftol, ftol=ftol, gtol=ftol)
        self.fit_x = (result.values['e_offset'] +
                      result.values['e_linear'] * x0 +
                      result.values['e_quadratic'] * x0**2)
        self.fit_y = result.best_fit
        self.fit_result = result
        self.residual = self.fit_y - y0

    def fit_multiple(self):
        """
        Fit data in sequence according to given strategies.
        The param_dict is extended to cover elemental parameters.
        Use app.precessEvents() for multi-threading.
        """
        # app = QApplication.instance()
        self.define_range()
        self.get_background()

        # PC = ParamController(self.param_dict, self.element_list)
        # self.param_dict = PC.params

        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.es_peak = trim_escape_peak(self.data,
                                            self.param_dict,
                                            self.y0.size)
            y0 = self.y0 - self.bg - self.es_peak
        else:
            y0 = self.y0 - self.bg

        t0 = time.time()
        self.fit_info = "Spectrum fitting of the sum spectrum (incident energy "\
                        f"{self.param_dict['coherent_sct_energy']['value']})."
        # app.processEvents()
        # logger.info('-------- '+self.fit_info+' --------')

        for k, v in six.iteritems(self.all_strategy):
            if v:
                strat_name = fit_strategy_list[v-1]
                # self.fit_info = 'Fit with {}: {}'.format(k, strat_name)

                logger.info(self.fit_info)
                strategy = extract_strategy(self.param_dict, strat_name)
                #  register the strategy and extend the parameter list
                #  to cover all given elements
                register_strategy(strat_name, strategy)
                set_parameter_bound(self.param_dict, strat_name)

                self.fit_data(self.x0, y0)
                self.update_param_with_result()

                # The following is a patch for rare cases when fitting results in negative
                #   areas for some emission lines. These are typically non-existent lines, but
                #   they should not be automatically eliminated from the list. To prevent
                #   elimination, set the area to some small positive value.
                for key, val in self.param_dict.items():
                    if key.endswith("_area") and val["value"] <= 0.0:
                        _small_value_for_area = 0.1
                        logger.warning(
                            f"Fitting resulted in negative value for '{key}' ({val['value']}). \n"
                            f"    In order to continue using the emission line in future computations, "
                            f"the fitted area is set to a small value ({_small_value_for_area}).\n    Delete "
                            f"the emission line from the list if you know it is not present in "
                            f"the sample.")
                        val["value"] = _small_value_for_area  # Some small number

                #  calculate r2
                self.r2 = cal_r2(y0, self.fit_y)
                self.assign_fitting_result()
                # app.processEvents()

        t1 = time.time()
        logger.warning('Time used for summed spectrum fitting is : {}'.format(t1-t0))

        # for GUI purpose only
        # if we do not clear the dict first, there is not update on the GUI
        param_temp = copy.deepcopy(self.param_dict)
        del self.param_dict['non_fitting_values']
        self.param_dict = param_temp

        self.comps.clear()
        comps = self.fit_result.eval_components(x=self.x0)
        self.comps = combine_lines(comps, self.element_list, self.bg)

        if self.param_dict['non_fitting_values']['escape_ratio'] > 0:
            self.fit_y += self.bg + self.es_peak
            self.comps['escape'] = self.es_peak
        else:
            self.fit_y += self.bg

        self.save_result()
        self.assign_fitting_result()
        self.fit_info = 'Summed spectrum fitting is done!'
        logger.info('-------- ' + self.fit_info + ' --------')

        self.param_q.append(copy.deepcopy(self.param_dict))
        self.keep_size()

    def output_summed_data_fit(self):
        """Save energy, summed data and fitting curve to a file.
        """
        data = np.array([self.x0, self.y0, self.fit_y])
        output_fit_name = self.data_title + '_summed_spectrum_fit.txt'
        fpath = os.path.join(self.result_folder, output_fit_name)
        np.savetxt(fpath, data.T)

    def assign_fitting_result(self):
        self.function_num = self.fit_result.nfev
        self.nvar = self.fit_result.nvarys
        # self.chi2 = np.around(self.fit_result.chisqr, 4)
        self.red_chi2 = np.around(self.fit_result.redchi, 4)

    def fit_single_pixel(self):
        """
        This function performs single pixel fitting.
        Multiprocess is considered.
        """
        # app = QApplication.instance()
        raise_bg = self.raise_bg
        pixel_bin = self.pixel_bin
        comp_elastic_combine = False
        linear_bg = self.linear_bg
        use_snip = self.use_snip
        bin_energy = self.bin_energy

        if self.pixel_fit_method == 0:
            pixel_fit = 'nnls'
        elif self.pixel_fit_method == 1:
            pixel_fit = 'nonlinear'

        logger.info("-------- Fitting of single pixels starts (incident_energy "
                    f"{self.param_dict['coherent_sct_energy']['value']} keV) --------")
        t0 = time.time()
        self.pixel_fit_info = 'Pixel fitting is in process.'

        # app.processEvents()
        self.result_map, calculation_info = single_pixel_fitting_controller(
            self.data_all,
            self.param_dict,
            method=pixel_fit,
            pixel_bin=pixel_bin,
            raise_bg=raise_bg,
            comp_elastic_combine=comp_elastic_combine,
            linear_bg=linear_bg,
            use_snip=use_snip,
            bin_energy=bin_energy)

        t1 = time.time()
        logger.info('Time used for pixel fitting is : {}'.format(t1-t0))

        #  get fitted spectrum and save them to figs
        if self.save_point is True:
            self.pixel_fit_info = 'Saving output ...'
            # app.processEvents()
            elist = calculation_info['fit_name']
            matv = calculation_info['regression_mat']
            results = calculation_info['results']
            # fit_range = calculation_info['fit_range']
            x = calculation_info['energy_axis']
            x = (self.param_dict['e_offset']['value'] +
                 self.param_dict['e_linear']['value']*x +
                 self.param_dict['e_quadratic']['value'] * x**2)
            data_fit = calculation_info['exp_data']

            p1 = [self.point1v, self.point1h]
            p2 = [self.point2v, self.point2h]

            if self.point2v > 0 or self.point2h > 0:
                prefix_fname = self.hdf_name.split('.')[0]
                output_folder = os.path.join(self.result_folder, prefix_fname+'_pixel_fit')
                if os.path.exists(output_folder) is False:
                    os.mkdir(output_folder)
                save_fitted_fig(x, matv, results[:, :, 0:len(elist)],
                                p1, p2,
                                data_fit, self.param_dict,
                                output_folder, use_snip=use_snip)

            # the output movie are saved as the same name
            # need to define specific name for prefix, to be updated
            # save_fitted_as_movie(x, matv, results[:, :, 0:len(elist)],
            #                      p1, p2,
            #                      data_fit, self.param_dict,
            #                      self.result_folder, prefix=prefix_fname, use_snip=use_snip)
            logger.info('Done with saving fitting plots.')
        try:
            self.save2Dmap_to_hdf(calculation_info=calculation_info, pixel_fit=pixel_fit)
            self.pixel_fit_info = 'Pixel fitting is done!'
            # app.processEvents()
        except ValueError:
            logger.warning('Fitting result can not be saved to h5 file.')
        except IOError as ex:
            logger.warning(f"{ex}")
        logger.info('-------- Fitting of single pixels is done! --------')

    def save_pixel_fitting_to_db(self):
        """Save fitting results to analysis store
        """
        from .data_to_analysis_store import save_data_to_db
        doc = {}
        doc['param'] = self.param_dict
        doc['exp'] = self.data
        doc['fitted'] = self.fit_y
        save_data_to_db(self.runid, self.result_map, doc)

    def save2Dmap_to_hdf(self, *, calculation_info=None, pixel_fit='nnls'):
        """
        Save fitted 2D map of elements into hdf file after fitting is done. User
        can choose to interpolate the image based on x,y position or not.

        Parameters
        ----------
        pixel_fit : str
            If nonlinear is chosen, more information needs to be saved.
        """

        prefix_fname = self.hdf_name.split('.')[0]
        if len(prefix_fname) == 0:
            prefix_fname = 'tmp'

        # Generate the path to computed ROIs in the HDF5 file
        det_name = "detsum"  # Assume that ROIs are computed using the sum of channels
        fit_name = f"{prefix_fname}_fit"

        # Search for channel name in the data title. Channels are named
        #   det1, det2, ... , i.e. 'det' followed by integer number.
        # The channel name is always located at the end of the ``data_title``.
        # If the channel name is found, then build the path using this name.
        srch = re.search("det\d+$", self.data_title)  # noqa: W605
        if srch:
            det_name = srch.group(0)
            fit_name = f"{prefix_fname}_{det_name}_fit"
        inner_path = f"xrfmap/{det_name}"

        # Update GUI so that results can be seen immediately
        self.fit_img[fit_name] = self.result_map

        if not os.path.isfile(self.hdf_path):
            raise IOError(f"File '{self.hdf_path}' does not exist. Data is not saved to HDF5 file.")

        save_fitdata_to_hdf(self.hdf_path, self.result_map, datapath=inner_path)

        # output error
        if pixel_fit == 'nonlinear':
            error_map = calculation_info['error_map']
            save_fitdata_to_hdf(self.hdf_path, error_map, datapath=inner_path,
                                data_saveas='xrf_fit_error',
                                dataname_saveas='xrf_fit_error_name')

    def calculate_roi_sum(self):
        if self.roi_sum_opt['status'] is True:
            low = int(self.roi_sum_opt['low']*100)
            high = int(self.roi_sum_opt['high']*100)
            logger.info('ROI sum at range ({}, {})'.format(self.roi_sum_opt['low'],
                                                           self.roi_sum_opt['high']))
            sumv = np.sum(self.data_all[:, :, low:high], axis=2)
            self.result_map['ROI'] = sumv
            # save to hdf again, this is not optimal
            self.save2Dmap_to_hdf()

    def get_latest_single_pixel_fitting_data(self):
        r"""
        Returns the latest results of single pixel fitting. The returned results include
        the name of the scaler (None if no scaler is selected) and the dictionary of
        the computed XRF maps for selected emission lines.
        """

        # Find the selected scaler name. Scaler is None if not scaler is selected.
        scaler_name = None
        if self.scaler_index > 0:
            scaler_name = self.scaler_keys[self.scaler_index-1]

        # Result map. If 'None', then no fitting results exists.
        #   Single pixel fitting must be run
        result_map = self.result_map.copy()

        return result_map, scaler_name

    def output_2Dimage(self, to_tiff=True):
        """Read data from h5 file and save them into either tiff or txt.

        Parameters
        ----------
        to_tiff : str, optional
            save to tiff or not
        """
        scaler_v = None
        _post_name_folder = "_".join(self.data_title.split('_')[:-1])
        if self.scaler_index > 0:
            scaler_v = self.scaler_keys[self.scaler_index-1]
            logger.info(f"*** NORMALIZED data is saved. Scaler: '{scaler_v}' ***")

        if to_tiff:
            dir_prefix = "output_tiff_"
            file_format = "tiff"
        else:
            dir_prefix = "output_txt_"
            file_format = "txt"

        output_n = dir_prefix + _post_name_folder
        output_dir = os.path.join(self.result_folder, output_n)

        # self.img_dict contains ALL loaded datasets, including a separate "positions" dataset
        if "positions" in self.img_dict:
            positions_dict = self.img_dict["positions"]
        else:
            positions_dict = {}

        # Scalers are located in a separate dataset in 'img_dict'. They are also referenced
        #   in each '_fit' dataset (and in the selected dataset 'self.dict_to_plot')
        #   The list of scaler names is used to avoid attaching the detector channel name
        #   to file names that contain scaler data (scalers typically do not depend on
        #   the selection of detector channels.
        scaler_dsets = [_ for _ in self.img_dict.keys() if re.search(r"_scaler$", _)]
        if scaler_dsets:
            scaler_name_list = list(self.img_dict[scaler_dsets[0]].keys())
        else:
            scaler_name_list = None

        output_data(output_dir=output_dir,
                    interpolate_to_uniform_grid=self.map_interpolation,
                    dataset_name=self.img_title, quant_norm=self.quantitative_normalization,
                    param_quant_analysis=self.param_quant_analysis,
                    dataset_dict=self.dict_to_plot, positions_dict=positions_dict,
                    file_format=file_format, scaler_name=scaler_v,
                    scaler_name_list=scaler_name_list)

    def save_result(self, fname=None):
        """
        Save fitting results.

        Parameters
        ----------
        fname : str, optional
            name of output file
        """
        if not fname:
            fname = self.data_title+'_out.txt'
        filepath = os.path.join(self.result_folder, fname)

        area_list = []
        for v in list(self.fit_result.params.keys()):
            if 'ka1_area' in v or 'la1_area' in v or 'ma1_area' in v or 'amplitude' in v:
                area_list.append(v)
        try:
            with open(filepath, 'w') as myfile:
                myfile.write('\n {:<10} \t {} \t {}'.format('name', 'summed area', 'error in %'))
                for k, v in six.iteritems(self.comps):
                    if k == 'background':
                        continue
                    for name in area_list:
                        if k.lower() in name.lower():
                            std_error = self.fit_result.params[name].stderr
                            if std_error is None:
                                # Do not print 'std_error' if it is not computed by lmfit
                                errorv_s = ''
                            else:
                                errorv = std_error / (self.fit_result.params[name].value + 1e-8)
                                errorv *= 100
                                errorv = np.round(errorv, 3)
                                errorv_s = f"{errorv}%"
                            myfile.write('\n {:<10} \t {} \t {}'.format(k, np.round(np.sum(v), 3), errorv_s))
                myfile.write('\n\n')

                # Print the report from lmfit
                # Remove strings (about 50%) on the variables that stayed at initial value
                report = lmfit.fit_report(self.fit_result, sort_pars=True)
                report = report.split('\n')
                report = [s for s in report if 'at initial value' not in s and '##' not in s]
                report = '\n'.join(report)
                myfile.write(report)

                logger.warning('Results are saved to {}'.format(filepath))
        except FileNotFoundError:
            print("Summed spectrum fitting results are not saved.")

    def update_name_list(self):
        """
        When result_dict_names change, the looper in enaml will update.
        """
        # need to clean list first, in order to refresh the list in GUI
        self.selected_index = 0
        self.elementinfo_list = []

        self.result_dict_names = []
        self.result_dict_names = list(self.EC.element_dict.keys())
        self.param_dict = update_param_from_element(self.param_dict,
                                                    list(self.EC.element_dict.keys()))

        self.element_list = []
        self.element_list = list(self.EC.element_dict.keys())
        logger.info('The full list for fitting is {}'.format(self.element_list))

    def create_EC_list(self, element_list):
        temp_dict = OrderedDict()
        for e in element_list:
            if e == "":
                pass
            elif '-' in e:  # pileup peaks
                e1, e2 = e.split('-')
                energy = float(get_energy(e1))+float(get_energy(e2))

                ps = PreFitStatus(z=get_Z(e),
                                  energy=str(energy), norm=1)
                temp_dict[e] = ps

            else:
                ename = e.split('_')[0]
                ps = PreFitStatus(z=get_Z(ename),
                                  energy=get_energy(e),
                                  norm=1)

                temp_dict[e] = ps
        self.EC.add_to_dict(temp_dict)

    # def manual_input(self):
    #     #default_area = 1e2
    #     ps = PreFitStatus(z=get_Z(self.e_name),
    #                       energy=get_energy(self.e_name),
    #                       #area=area_dict[self.e_name]*ratio_v,
    #                       #spectrum=data_out[self.e_name]*ratio_v,
    #                       #maxv=self.add_element_intensity,
    #                       norm=1)
    #                       #lbd_stat=False)
    #
    #     self.EC.add_to_dict({self.e_name: ps})
    #     logger.info('')
    #     self.update_name_list()

    # def add_pileup(self):
    #     if self.pileup_data['intensity'] > 0:
    #         e_name = (self.pileup_data['element1'] + '-'
    #                   + self.pileup_data['element2'])
    #
    #         energy = str(float(get_energy(self.pileup_data['element1']))
    #                      + float(get_energy(self.pileup_data['element2'])))
    #
    #         ps = PreFitStatus(z=get_Z(e_name),
    #                           energy=energy,
    #                           #area=area_dict[e_name]*ratio_v,
    #                           #spectrum=data_out[e_name]*ratio_v,
    #                           #maxv=self.pileup_data['intensity'],
    #                           norm=1)
    #                           #lbd_stat=False)
    #         logger.info('{} peak is added'.format(e_name))
    #     self.EC.add_to_dict({e_name: ps})
    #     self.update_name_list()


def combine_lines(components, element_list, background):
    """
    Combine results for different lines of the same element.
    And also add pileup, userpeak, background, compton and elastic.

    Parameters
    ----------
    components : dict
        output results from lmfit
    element_list : list
        list of elemental lines
    background : array
        background calculated in given range

    Returns
    -------
    dict :
        combined results for elements and other related peaks.
    """
    new_components = {}
    for e in element_list:
        if len(e) <= 4:
            e_temp = e.split('_')[0]
            intensity = 0
            for k, v in six.iteritems(components):
                if (e_temp in k) and (e not in k):
                    intensity += v
            new_components[e] = intensity
        elif 'user' in e.lower():
            for k, v in six.iteritems(components):
                if e in k:
                    new_components[e] = v
        else:
            comp_name = 'pileup_' + e.replace('-', '_') + '_'  # change Si_K-Si_K to Si_K_Si_K
            new_components[e] = components[comp_name]

    # add background and elastic
    new_components['background'] = background
    new_components['compton'] = components['compton']
    new_components['elastic'] = components['elastic_']
    return new_components


def extract_strategy(param, name):
    """
    Extract given strategy from param dict.

    Parameters
    ----------
    param : dict
        saving all parameters
    name : str
        strategy name

    Returns
    -------
    dict :
        with given strategy as value
    """
    param_new = copy.deepcopy(param)
    return {k: v[name] for k, v in six.iteritems(param_new)
            if k != 'non_fitting_values'}


def define_param_bound_type(param,
                            strategy_list=['adjust_element2, adjust_element3'],
                            b_type='fixed'):
    param_new = copy.deepcopy(param)
    for k, v in six.iteritems(param_new):
        for data in strategy_list:
            if data in list(v.keys()):
                param_new[k][data] = b_type
    return param_new


def extract_result(data, element):
    """
    Extract fitting result returned from fitting of multi files.

    Parameters
    ----------
    data : list
        list of dict
    element : str
        elemental line
    """
    data_map = []
    for v in data:
        data_map.append(v[element])
    return np.array(data_map)


def bin_data_pixel(data, nearest_n=4):
    """
    Bin 3D data according to number of pixels defined in dim 1 and 2.

    Parameters
    ----------
    data : 3D array
        exp data with energy channel in 3rd dim.
    nearest_n : int, optional
        define how many pixels to be considered.
    """
    new_data = np.array(data)
    d_shape = data.shape

    # if nearest_n == 4:
    #     for i in [-1, 1]:
    #         new_data[1:-1, 1:-1, :] += data[1+i:d_shape[0]-1+i, 1:d_shape[1]-1, :]
    #     for j in [-1, 1]:
    #         new_data[1:-1, 1:-1, :] += data[1:d_shape[0]-1, 1+j:d_shape[1]-1+j, :]

    if nearest_n == 4:
        for i in np.arange(d_shape[0]-1):
            for j in np.arange(d_shape[1]-1):
                new_data[i, j, :] += (new_data[i+1, j, :] +
                                      new_data[i, j+1, :] +
                                      new_data[i+1, j+1, :])
        new_data[:-1, :-1, :] /= nearest_n

    if nearest_n == 9:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                new_data[1:-1, 1:-1, :] += data[1+i:d_shape[0]-1+i, 1+j:d_shape[1]-1+j, :]

        new_data[1:-1, 1:-1, :] /= nearest_n

    return new_data


def bin_data_spacial(data, bin_size=4):
    """
    Bin 2D/3D data based on first and second dim, i.e., 2 by 2 window, or 4 by 4.

    Parameters
    ----------
    data : array
        2D or 3D dataset
    bin_size : int
        window size. 2 means 2 by 2 window
    """
    if bin_size <= 1:
        return data

    data = np.asarray(data)
    if data.ndim == 2:
        d_shape = np.array([data.shape[0], data.shape[1]])/bin_size

        data_new = np.zeros([d_shape[0], d_shape[1]])
        for i in np.arange(d_shape[0]):
            for j in np.arange(d_shape[1]):
                data_new[i, j] = np.sum(data[i*bin_size:i*bin_size+bin_size,
                                             j*bin_size:j*bin_size+bin_size])
    elif data.ndim == 3:
        d_shape = np.array([data.shape[0], data.shape[1]])/bin_size

        data_new = np.zeros([d_shape[0], d_shape[1], data.shape[2]])
        for i in np.arange(d_shape[0]):
            for j in np.arange(d_shape[1]):
                data_new[i, j, :] = np.sum(data[i*bin_size:i*bin_size+bin_size,
                                                j*bin_size:j*bin_size+bin_size, :], axis=(0, 1))
    return data_new


def conv_expdata_energy(data, width=2):
    """
    Do convolution on the 3rd axis, energy axis.
    Paremeters
    ----------
    data : 3D array
        exp spectrum
    width : int, optional
        width of the convolution function.
    Returns
    -------
    array :
        after convolution
    """
    data_new = np.array(data)
    if width == 2:
        conv_f = [1.0/2, 1.0/2]
    if width == 3:
        conv_f = [1.0/3, 1.0/3, 1.0/3]
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[1]):
            data_new[i, j, :] = np.convolve(data_new[i, j, :], conv_f, mode='same')

    return data_new


def bin_data_energy2D(data, bin_step=2, axis_v=0, sum_data=False):
    """
    Bin data based on given dim, i.e., a dim for energy spectrum.
    Return a copy of the data. Currently only binning along first
    dim is implemented.

    Parameters
    ----------
    data : 2D array
    bin_step : int, optional
        size to bin the data
    axis_v : int, optional
        along which dir to bin data.
    sum_data : bool, optional
        sum data from each bin or not

    Returns
    -------
    binned data with reduced dim of the previous size.
    """
    if bin_step == 1:
        return data

    data = np.array(data)
    if axis_v == 0:
        if bin_step == 2:
            new_len = data.shape[0]/bin_step
            m1 = data[::2, :]
            m2 = data[1::2, :]
            if sum_data is True:
                return (m1[:new_len, :] +
                        m2[:new_len, :])/bin_step
            else:
                return m1[:new_len, :]
        elif bin_step == 3:
            new_len = data.shape[0]/bin_step
            m1 = data[::3, :]
            m2 = data[1::3, :]
            m3 = data[2::3, :]
            if sum_data is True:
                return (m1[:new_len, :] +
                        m2[:new_len, :] +
                        m3[:new_len, :])/bin_step
            else:
                return m1[:new_len, :]
        elif bin_step == 4:
            new_len = data.shape[0]/bin_step
            m1 = data[::4, :]
            m2 = data[1::4, :]
            m3 = data[2::4, :]
            m4 = data[3::4, :]
            if sum_data is True:
                return (m1[:new_len, :] +
                        m2[:new_len, :] +
                        m3[:new_len, :] +
                        m4[:new_len, :])/bin_step
            else:
                return m1[:new_len, :]


def bin_data_energy3D(data, bin_step=2, sum_data=False):
    """
    Bin 3D data along 3rd axis, i.e., a dim for energy spectrum.
    Return a copy of the data.

    Parameters
    ----------
    data : 3D array
    sum_data : bool, optional
        sum data from each bin or not

    Returns
    -------
    binned data with reduced dim of the previous size.
    """
    if bin_step == 1:
        return data
    data = np.array(data)
    if bin_step == 2:
        new_len = data.shape[2]/2
        new_data1 = data[:, :, ::2]
        new_data2 = data[:, :, 1::2]
        if sum_data is True:
            return (new_data1[:, :, :new_len] +
                    new_data2[:, :, :new_len])/bin_step
        else:
            return new_data1[:, :, :new_len]
    elif bin_step == 3:
        new_len = data.shape[2]/3
        new_data1 = data[:, :, ::3]
        new_data2 = data[:, :, 1::3]
        new_data3 = data[:, :, 2::3]
        if sum_data is True:
            return (new_data1[:, :, :new_len] +
                    new_data2[:, :, :new_len] +
                    new_data3[:, :, :new_len])/bin_step
        else:
            return new_data1[:, :, :new_len]
    elif bin_step == 4:
        new_len = data.shape[2]/4
        new_data1 = data[:, :, ::4]
        new_data2 = data[:, :, 1::4]
        new_data3 = data[:, :, 2::4]
        new_data4 = data[:, :, 3::4]
        if sum_data is True:
            return (new_data1[:, :, :new_len] +
                    new_data2[:, :, :new_len] +
                    new_data3[:, :, :new_len] +
                    new_data4[:, :, :new_len])/bin_step
        else:
            return new_data1[:, :, :new_len]


def cal_r2(y, y_cal):
    """
    Calculate r2 statistics.
    Parameters
    ----------
    y : array
        exp data
    y_cal : array
        fitted data
    Returns
    -------
    float
    """
    sse = np.sum((y-y_cal)**2)
    sst = np.sum((y - np.mean(y))**2)
    return 1-sse/sst


def calculate_area(e_select, matv, results,
                   param, first_peak_area=False):
    """
    Parameters
    ----------
    e_select : list
        elements
    matv : 2D array
        matrix constains elemental profile as columns
    results : 3D array
        x, y positions, and each element's weight on third dim
    param : dict
        parameters of fitting
    first_peak_area : Bool, optional
        get overal peak area or only the first peak area, such as Ar_Ka1

    Returns
    -------
    dict :
        dict of each 2D elemental distribution
    """
    total_list = e_select + ['snip_bkg'] + ['r2_adjust']
    mat_sum = np.sum(matv, axis=0)

    result_map = dict()
    for i in range(len(e_select)):
        if first_peak_area is not True:
            result_map.update({total_list[i]: results[:, :, i]*mat_sum[i]})
        else:
            if total_list[i] not in K_LINE+L_LINE+M_LINE:
                ratio_v = 1
            else:
                ratio_v = get_branching_ratio(total_list[i],
                                              param['coherent_sct_energy']['value'])
            result_map.update({total_list[i]: results[:, :, i]*mat_sum[i]*ratio_v})

    # add background and res
    result_map.update({total_list[-2]: results[:, :, -2]})
    result_map.update({total_list[-1]: results[:, :, -1]})

    return result_map


def save_fitted_fig(x_v, matv, results,
                    p1, p2, data_all, param_dict,
                    result_folder, use_snip=False):
    """
    Save single pixel fitting resutls to figs.
    """
    low_limit_v = 0.5

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    max_v = np.max(data_all[p1[0]:p2[0], p1[1]:p2[1], :])

    fitted_sum = None
    for m in range(p1[0], p2[0]):
        for n in range(p1[1], p2[1]):
            data_y = data_all[m, n, :]

            fitted_y = np.sum(matv*results[m, n, :], axis=1)
            if use_snip is True:
                bg = snip_method(data_y,
                                 param_dict['e_offset']['value'],
                                 param_dict['e_linear']['value'],
                                 param_dict['e_quadratic']['value'],
                                 width=param_dict['non_fitting_values']['background_width'])
                fitted_y += bg

            if fitted_sum is None:
                fitted_sum = fitted_y
            else:
                fitted_sum += fitted_y
            ax.cla()
            ax.set_title('Single pixel fitting for point ({}, {})'.format(m, n))
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Counts')
            ax.set_ylim(low_limit_v, max_v*2)

            ax.semilogy(x_v, data_y, label='exp', linestyle='', marker='.')
            ax.semilogy(x_v, fitted_y, label='fit')

            ax.legend()
            output_path = os.path.join(result_folder,
                                       'data_out_'+str(m)+'_'+str(n)+'.png')
            plt.savefig(output_path)

    ax.cla()
    sum_y = np.sum(data_all[p1[0]:p2[0], p1[1]:p2[1], :], axis=(0, 1))
    ax.set_title('Summed spectrum from point ({},{}) '
                 'to ({},{})'.format(p1[0], p1[1], p2[0], p2[1]))
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    ax.set_ylim(low_limit_v, np.max(sum_y)*2)
    ax.semilogy(x_v, sum_y, label='exp', linestyle='', marker='.')
    ax.semilogy(x_v, fitted_sum, label='fit', color='red')

    ax.legend()
    fit_sum_name = 'pixel_sum_'+str(p1[0])+'-'+str(p1[1])+'_'+str(p2[0])+'-'+str(p2[1])+'.png'
    output_path = os.path.join(result_folder, fit_sum_name)
    plt.savefig(output_path)


def save_fitted_as_movie(x_v, matv, results,
                         p1, p2, data_all, param_dict,
                         result_folder, prefix=None, use_snip=False, dpi=150):
    """
    Create movie to save single pixel fitting resutls.
    """
    total_n = data_all.shape[1]*p2[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_aspect('equal')
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    max_v = np.max(data_all[p1[0]:p2[0], p1[1]:p2[1], :])
    ax.set_ylim([0, 1.1*max_v])

    l1,  = ax.plot(x_v,  x_v, label='exp', linestyle='-', marker='.')
    l2,  = ax.plot(x_v,  x_v, label='fit', color='red', linewidth=2)

    # fitted_sum = None
    plist = []
    for v in range(total_n):
        m = v / data_all.shape[1]
        n = v % data_all.shape[1]
        if m >= p1[0] and m <= p2[0] and n >= p1[1] and n <= p2[1]:
            plist.append((m, n))

    def update_img(p_val):
        m = p_val[0]
        n = p_val[1]
        data_y = data_all[m, n, :]

        fitted_y = np.sum(matv*results[m, n, :], axis=1)
        if use_snip is True:
            bg = snip_method(data_y,
                             param_dict['e_offset']['value'],
                             param_dict['e_linear']['value'],
                             param_dict['e_quadratic']['value'],
                             width=param_dict['non_fitting_values']['background_width'])
            fitted_y += bg

        ax.set_title('Single pixel fitting for point ({}, {})'.format(m, n))
        # ax.set_ylim(low_limit_v, max_v*2)
        l1.set_ydata(data_y)
        l2.set_ydata(fitted_y)
        return l1, l2

    writer = animation.writers['ffmpeg'](fps=30)
    ani = animation.FuncAnimation(fig, update_img, plist)
    if prefix:
        output_file = prefix+'_pixel.mp4'
    else:
        output_file = 'fit_pixel.mp4'
    output_p = os.path.join(result_folder, output_file)
    ani.save(output_p, writer=writer, dpi=dpi)


def fit_per_line_nnls(row_num, data,
                      matv, param, use_snip,
                      num_data, num_feature):
    """
    Fit experiment data for a given row using nnls algorithm.

    Parameters
    ----------
    row_num : int
        which row to fit
    data : array
        selected one row of experiment spectrum
    matv : array
        matrix for regression analysis
    param : dict
        fitting parameters
    use_snip : bool
        use snip algorithm to remove background or not
    num_data : int
        number of total data points
    num_feature : int
        number of data features

    Returns
    -------
    array :
        fitting values for all the elements at a given row. Background is
        calculated as a summed value. Also residual is included.
    """
    logger.debug(f"Row number at {row_num}")
    out = []
    bg_sum = 0
    for i in range(data.shape[0]):
        if use_snip is True:
            bg = snip_method(data[i, :],
                             param['e_offset']['value'],
                             param['e_linear']['value'],
                             param['e_quadratic']['value'],
                             width=param['non_fitting_values']['background_width'])
            y = data[i, :] - bg
            bg_sum = np.sum(bg)

        else:
            y = data[i, :]

        result, res = nnls_fit(y, matv, weights=None)
        sst = np.sum((y-np.mean(y))**2)
        if not math.isclose(sst, 0, abs_tol=1e-20):
            r2_adjusted = 1 - res / (num_data - num_feature - 1) / (sst / (num_data - 1))
        else:
            # This happens if all elements of 'y' are equal (most likely == 0)
            r2_adjusted = 0
        result = list(result) + [bg_sum, r2_adjusted]
        out.append(result)
    return np.array(out)


def fit_pixel_multiprocess_nnls(exp_data, matv, param,
                                use_snip=False, lambda_reg=0.0):
    """
    Multiprocess fit of experiment data.

    Parameters
    ----------
    exp_data : array
        3D data of experiment spectrum
    matv : array
        matrix for regression analysis
    param : dict
        fitting parameters
    use_snip : bool, optional
        use snip algorithm to remove background or not
    lambda_reg : float, optional
        applied L2 norm regularizaiton if set above zero

    Returns
    -------
    dict :
        fitting values for all the elements
    """
    num_processors_to_use = multiprocessing.cpu_count()

    logger.info('cpu count: {}'.format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)
    n_data, n_feature = matv.shape

    if lambda_reg > 0:
        logger.info('nnls fit with regularization term, lambda={}'.format(lambda_reg))
        diag_m = np.diag(np.ones(n_feature))*np.sqrt(lambda_reg)
        matv = np.concatenate((matv, diag_m), axis=0)
        exp_tmp = np.zeros([exp_data.shape[0], exp_data.shape[1], n_feature])
        exp_data = np.concatenate((exp_data, exp_tmp), axis=2)

    result_pool = [pool.apply_async(fit_per_line_nnls,
                                    (n, exp_data[n, :, :], matv,
                                     param, use_snip, n_data, n_feature))
                   for n in range(exp_data.shape[0])]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()

    # chop data back
    if lambda_reg > 0:
        matv = matv[:-n_feature, :]
        exp_data = exp_data[:, :, :-n_feature]

    return np.asarray(results)


def spectrum_nonlinear_fit(pars, x, reg_mat):
    vals = pars.valuesdict()
    return np.sum(vals['a{}'.format(i)] * reg_mat[:, i] for i in range(len(vals)))


def residual_nonlinear_fit(pars, x, data=None, reg_mat=None):
    return spectrum_nonlinear_fit(pars, x, reg_mat) - data


def fit_pixel_nonlinear_per_line(row_num, data, x0,
                                 param, reg_mat,
                                 use_snip):  # c_weight, fit_num, ftol):

    # c_weight = 1
    # fit_num = 100
    # ftol = 1e-3

    elist = param['non_fitting_values']['element_list'].split(', ')
    elist = [e.strip(' ') for e in elist]

    # LinearModel = lmfit.Model(simple_spectrum_fun_for_nonlinear)
    # for i in np.arange(reg_mat.shape[0]):
    #     LinearModel.set_param_hint('a'+str(i), value=0.1, min=0, vary=True)

    logger.info('Row number at {}'.format(row_num))
    out = []
    snip_bg = 0
    for i in range(data.shape[0]):
        if use_snip is True:
            bg = snip_method(data[i, :],
                             param['e_offset']['value'],
                             param['e_linear']['value'],
                             param['e_quadratic']['value'],
                             width=param['non_fitting_values']['background_width'])
            y0 = data[i, :] - bg
            snip_bg = np.sum(bg)
        else:
            y0 = data[i, :]

        fit_params = lmfit.Parameters()
        for i in range(reg_mat.shape[1]):
            fit_params.add('a'+str(i), value=1.0, min=0, vary=True)

        result = lmfit.minimize(residual_nonlinear_fit,
                                fit_params, args=(x0,),
                                kws={'data': y0, 'reg_mat': reg_mat})

        # result = MS.model_fit(x0, y0,
        #                       weights=1/np.sqrt(c_weight+y0),
        #                       maxfev=fit_num,
        #                       xtol=ftol, ftol=ftol, gtol=ftol)
        # namelist = list(result.keys())
        temp = {}
        temp['value'] = [result.params[v].value for v in list(result.params.keys())]
        temp['err'] = [result.params[v].stderr for v in list(result.params.keys())]
        temp['snip_bg'] = snip_bg
        out.append(temp)
    return out


def fit_pixel_multiprocess_nonlinear(data, x, param, reg_mat, use_snip=False):
    """
    Multiprocess fit of experiment data.

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

    num_processors_to_use = multiprocessing.cpu_count()
    logger.info('cpu count: {}'.format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    # fit_params = lmfit.Parameters()
    # for i in range(reg_mat.shape[1]):
    #     fit_params.add('a'+str(i), value=1.0, min=0, vary=True)

    result_pool = [pool.apply_async(fit_pixel_nonlinear_per_line,
                                    (n, data[n, :, :], x,
                                     param, reg_mat, use_snip))
                   for n in range(data.shape[0])]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()

    return results


def get_area_and_error_nonlinear_fit(elist, fit_results, reg_mat):

    mat_sum = np.sum(reg_mat, axis=0)
    area_dict = OrderedDict()
    error_dict = OrderedDict()
    for name in elist:
        area_dict[name] = np.zeros([len(fit_results), len(fit_results[0])])
        error_dict[name] = np.zeros([len(fit_results), len(fit_results[0])])
    # area_dict = OrderedDict({name:np.zeros([len(fit_results), len(fit_results[0])]) for name in elist})
    # error_dict = OrderedDict({name:np.zeros([len(fit_results), len(fit_results[0])]) for name in elist})
    area_dict['snip_bg'] = np.zeros([len(fit_results), len(fit_results[0])])
    weights_mat = np.zeros([len(fit_results), len(fit_results[0]), len(error_dict)])

    for i in range(len(fit_results)):
        for j in range(len(fit_results[0])):
            for m, v in enumerate(six.iterkeys(area_dict)):
                if v == 'snip_bg':
                    area_dict[v][i, j] = fit_results[i][j]['snip_bg']
                else:
                    area_dict[v][i, j] = fit_results[i][j]['value'][m]
                    error_dict[v][i, j] = fit_results[i][j]['err'][m]
                    weights_mat[i, j, m] = fit_results[i][j]['value'][m]

    for i, v in enumerate(six.iterkeys(area_dict)):
        if v == 'snip_bg':
            continue
        area_dict[v] *= mat_sum[i]
        error_dict[v] *= mat_sum[i]

    return area_dict, error_dict, weights_mat


def single_pixel_fitting_controller(input_data, parameter,
                                    incident_energy=None, method='nnls',
                                    pixel_bin=0, raise_bg=0,
                                    comp_elastic_combine=False,
                                    linear_bg=False,
                                    use_snip=True,
                                    bin_energy=1):
    """
    Parameters
    ----------
    input_data : array
        3D array of spectrum
    parameter : dict
        parameter for fitting
    incident_energy : float, optional
        incident beam energy in KeV
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

    Returns
    -------
    result_map : dict
        of elemental map for given elements
    calculation_info : dict
        dict of fitting information
    """
    param = copy.deepcopy(parameter)
    if incident_energy is not None:
        param['coherent_sct_energy']['value'] = incident_energy
    # cut data into proper range
    x, exp_data, fit_range = get_cutted_spectrum_in3D(input_data,
                                                      param['non_fitting_values']['energy_bound_low']['value'],
                                                      param['non_fitting_values']['energy_bound_high']['value'],
                                                      param['e_offset']['value'],
                                                      param['e_linear']['value'])

    # calculate matrix for regression analysis
    elist = param['non_fitting_values']['element_list'].split(', ')
    elist = [e.strip(' ') for e in elist]
    e_select, matv, e_area = construct_linear_model(x, param, elist)

    # The initial list of elines may contain lines that are not activated for the incident beam
    #   energy. This always happens for at least one line when batches of XRF scans obtained for
    #   the range of beam energies are fitted for the same selection of emission lines to generate
    #   XANES amps. In such experiments, the line of interest is typically not activated at the
    #   lower energies of the band. It is impossible to include non-activated lines in the linear
    #   model. In order to make processing results consistent throughout the batch (contain the
    #   same set of emission lines), the non-activated lines are represented by maps filled with zeros.
    elist_non_activated = list(set(elist) - set(e_select))
    if elist_non_activated:
        logger.warning("Some of the emission lines in the list are not activated: "
                       f"{elist_non_activated} at {param['coherent_sct_energy']['value']} keV.")

    if comp_elastic_combine is True:
        e_select = e_select[:-1]
        e_select[-1] = 'comp_elastic'

        matv_old = np.array(matv)
        matv = matv_old[:, :-1]
        matv[:, -1] += matv_old[:, -1]

    if linear_bg is True:
        e_select.append('const_bkg')

        matv_old = np.array(matv)
        matv = np.ones([matv_old.shape[0], matv_old.shape[1]+1])
        matv[:, :-1] = matv_old

    logger.info('Matrix used for linear fitting has components: {}'.format(e_select))

    # add const background, so nnls works better for values above zero
    if raise_bg > 0:
        exp_data += raise_bg

    # bin data based on nearest pixels, only two options
    if pixel_bin in [4, 9]:
        logger.info('Bin pixel data with parameter: {}'.format(pixel_bin))
        exp_data = bin_data_spacial(exp_data, bin_size=int(np.sqrt(pixel_bin)))
        # exp_data = bin_data_pixel(exp_data, nearest_n=pixel_bin)  # return a copy of data

    # bin data based on energy spectrum
    if bin_energy in [2, 3]:
        exp_data = conv_expdata_energy(exp_data, width=bin_energy)

    # make matrix smaller for single pixel fitting
    matv /= exp_data.shape[0]*exp_data.shape[1]
    # save matrix to analyze collinearity
    # np.save('mat.npy', matv)
    error_map = None

    if method == 'nnls':
        logger.info('Fitting method: non-negative least squares')
        lambda_reg = 0.0
        results = fit_pixel_multiprocess_nnls(exp_data, matv, param,
                                              use_snip=use_snip, lambda_reg=lambda_reg)
        # output area of dict
        result_map = calculate_area(e_select, matv, results,
                                    param, first_peak_area=False)
    else:
        logger.info('Fitting method: nonlinear least squares')
        matrix_norm = exp_data.shape[0]*exp_data.shape[1]
        fit_results = fit_pixel_multiprocess_nonlinear(exp_data, x, param, matv/matrix_norm,
                                                       use_snip=use_snip)

        result_map, error_map, results = get_area_and_error_nonlinear_fit(e_select,
                                                                          fit_results,
                                                                          matv/matrix_norm)

    # Generate 'zero' maps for the emission lines that were not activated
    for eline in elist_non_activated:
        result_map[eline] = np.zeros(shape=exp_data.shape[0:2])

    calculation_info = dict()
    if error_map is not None:
        calculation_info['error_map'] = error_map

    calculation_info['fit_name'] = e_select
    calculation_info['regression_mat'] = matv
    calculation_info['results'] = results
    calculation_info['fit_range'] = fit_range
    calculation_info['energy_axis'] = x
    calculation_info['exp_data'] = exp_data

    return result_map, calculation_info


def get_cutted_spectrum_in3D(exp_data, low_e, high_e,
                             e_offset, e_linear):
    """
    Cut exp data on the 3rd axis, energy axis.
    Parameters
    ----------
    exp_data : 3D array
    low_e : float
        low energy bound in KeV
    high_e : float
        high energy bound in KeV
    e_offset : float
        offset term in energy calibration
    e_linear : float
        linear term in energy calibration
    Returns
    -------
    x : array
        channel data
    data : 3D array
        after cutting into the correct range
    list :
        fitting range
    """

    # cut range
    data = np.array(exp_data)
    y0 = data[0, 0, :]
    x0 = np.arange(len(y0))

    # transfer energy value back to channel value
    lowv = (low_e - e_offset) / e_linear
    highv = (high_e - e_offset) / e_linear
    lowv = int(lowv)
    highv = int(highv)
    x, y = trim(x0, y0, lowv, highv)

    data = data[:, :, lowv: highv+1]
    return x, data, [lowv, highv]


def get_branching_ratio(elemental_line, energy):
    """
    Calculate the ratio of branching ratio, such as ratio of
    branching ratio of Ka1 to sum of br of all K lines.

    It doesn't matter which unit is used for cs calculation,
    as we only want the ratio. So we still use e.cs function.

    Parameters
    ----------
    elemental_line : str
        e.g., 'Mg_K', refers to the K lines of Magnesium
    energy : float
        incident energy in keV

    Returns
    -------
    float :
        calculated ratio
    """

    name, line = elemental_line.split('_')
    e = Element(name)
    transition_lines = TRANSITIONS_LOOKUP[line.upper()]

    sum_v = 0
    for v in transition_lines:
        sum_v += e.cs(energy)[v]
    ratio_v = e.cs(energy)[transition_lines[0]]/sum_v
    return ratio_v


def roi_sum_calculation(dir_path, file_prefix, fileID,
                        element_dict, interpath):
    """
    Parameters
    -----------
    dir_path : str
    file_prefix : str
    fileID : int
    element_dict : dict
        element name with low/high bound
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : dict
        roi sum for all given elements
    """
    num_str = '{:03d}'.format(fileID)
    # logger.info('File number is {}'.format(fileID))
    filename = file_prefix + num_str
    file_path = os.path.join(dir_path, filename)
    with h5py.File(file_path, 'r') as f:
        data = f[interpath][:]

    result_map = dict()
    # for v in six.iterkeys(element_dict):
    #     result_map[v] = np.zeros([datas[0], datas[1]])

    for k, v in six.iteritems(element_dict):
        result_map[k] = np.sum(data[:, :, v[0]: v[1]], axis=2)

    return result_map


def roi_sum_multi_files(dir_path, file_prefix,
                        start_i, end_i, element_dict,
                        interpath='entry/instrument/detector/data'):
    """
    Fitting for multiple files with Multiprocessing.

    Parameters
    -----------
    dir_path : str
    file_prefix : str
    start_i : int
        start id of given file
    end_i: int
        end id of given file
    element_dict : dict
        dict of element with [low, high] bounds as values
    interpath : str
        path inside hdf5 file to fetch the data

    Returns
    -------
    result : list
        fitting result as list of dict
    """
    num_processors_to_use = multiprocessing.cpu_count()
    logger.info('cpu count: {}'.format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    result_pool = [pool.apply_async(roi_sum_calculation,
                                    (dir_path, file_prefix,
                                     m, element_dict, interpath))
                   for m in range(start_i, end_i+1)]

    results = []
    for r in result_pool:
        results.append(r.get())

    pool.terminate()
    pool.join()
    return results


def get_cs(elemental_line, eng=12, norm=False, round_n=2):
    """
    Calculate cross section in barns/atom, use e.csb function.

    Parameters
    ----------
    elemental_line: str
        like Pt_L, Si_K
    eng : float
        incident energy in KeV
    norm : bool, optional
        normalized to the primary cs value or not.
    round_n : int
        number of decimal point.
    """
    if 'pileup' in elemental_line:
        return '-'
    elif '_K' in elemental_line:
        name_label = 'ka1'
        ename = elemental_line.split('_')[0]
    elif '_L' in elemental_line:
        name_label = 'la1'
        ename = elemental_line.split('_')[0]
    elif '_M' in elemental_line:
        name_label = 'ma1'
        ename = elemental_line.split('_')[0]
    else:
        return '-'

    e = Element(ename)
    sumv = 0
    for line_name in list(e.csb(eng).keys()):
        if name_label[0] in line_name:
            sumv += e.csb(eng)[line_name]
    if norm is True:
        return np.around(sumv/e.csb(eng)[name_label], round_n)
    else:
        return np.around(sumv, round_n)
