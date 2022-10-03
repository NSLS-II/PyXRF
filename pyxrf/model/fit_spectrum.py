from __future__ import absolute_import

import numpy as np
import time
import copy
import os
import re
import math
from collections import OrderedDict
import multiprocessing
import multiprocessing.pool
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import lmfit
import platform
from distutils.version import LooseVersion
import dask.array as da

from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Float, Bool
from skbeam.core.fitting.xrf_model import (
    ModelSpectrum,
    update_parameter_dict,
    # sum_area,
    set_parameter_bound,
    # ParamController,
    K_LINE,
    L_LINE,
    M_LINE,
    nnls_fit,
    construct_linear_model,
    # linear_spectrum_fitting,
    register_strategy,
    TRANSITIONS_LOOKUP,
)
from skbeam.fluorescence import XrfElement as Element
from .parameters import calculate_profile, fit_strategy_list, trim_escape_peak, define_range
from .fileio import save_fitdata_to_hdf, output_data

from ..core.fitting import rfactor
from ..core.quant_analysis import ParamQuantEstimation
from ..core.map_processing import fit_xrf_map, TerminalProgressBar, prepare_xrf_map, snip_method_numba

import logging

logger = logging.getLogger(__name__)


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
    """

    file_status = Str()

    img_dict = Dict()

    element_list = List()
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

    result_map = Dict()
    map_interpolation = Bool(False)
    hdf_path = Str()

    roi_sum_opt = Dict()
    scaler_keys = List()
    scaler_index = Int(0)

    # Reference to ParamModel object
    param_model = Typed(object)
    # Reference to FileIOModel object
    io_model = Typed(object)

    # Reference to FileIOMOdel
    io_model = Typed(object)

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
    qe_standard_distance_to_sample = Float(0.0)

    def __init__(self, *, param_model, io_model):
        self.all_strategy = OrderedDict()

        # Reference to ParamModel object
        self.param_model = param_model

        # Reference to FileIOMOdel
        self.io_model = io_model

        self.pileup_data = {"element1": "Si_K", "element2": "Si_K", "intensity": 100.0}

        # don't argue, plotting purposes
        self.fit_strategy1 = 0
        self.fit_strategy2 = 0
        self.fit_strategy1 = 1
        self.fit_strategy2 = 0

        # perform roi sum in given range
        self.roi_sum_opt["status"] = False
        self.roi_sum_opt["low"] = 0.0
        self.roi_sum_opt["high"] = 10.0

        self.qe_standard_selected_ref = None
        self.qe_standard_selected_copy = None

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
        self.data_title = change["value"]

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
        self.runid = change["value"]

    def img_dict_updated(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : bool
            True - 'io_model.img_dict` was updated, False - ignore
        """
        if change["value"]:
            _key = [k for k in self.io_model.img_dict.keys() if "scaler" in k]
            if len(_key) != 0:
                self.scaler_keys = sorted(self.io_model.img_dict[_key[0]].keys())

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
        self.scaler_index = change["value"]

    def img_title_update(self, change):
        r"""Observer function. Sets ``img_title`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.img_title = change["value"]

    def dict_to_plot_update(self, change):
        r"""Observer function. Sets ``dict_to_plot`` field to the same value as
        the identical variable in ``DrawImageAdvanced`` class"""
        self.dict_to_plot = change["value"]

    def update_selected_index(self, selected_element=None, element_list_new=None):

        if selected_element is None:
            # Currently selected element
            element = self.selected_element
        else:
            # Selected element (probably saved before update of the element list)
            element = selected_element

        if element_list_new is None:
            # Current element list
            element_list = self.param_model.element_list
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

    @observe("selected_index")
    def _selected_element_changed(self, change):
        if change["value"] > 0:
            ind_sel = change["value"] - 1
            if ind_sel >= len(self.param_model.element_list):
                ind_sel = len(self.param_model.element_list) - 1
                self.selected_index = ind_sel + 1  # Change the selection as well
            self.selected_element = self.param_model.element_list[ind_sel]
            if len(self.selected_element) <= 4:
                element = self.selected_element.split("_")[0]
                self.elementinfo_list = sorted(
                    [
                        e
                        for e in list(self.param_model.param_new.keys())
                        if (element + "_" in e) and ("pileup" not in e)  # error between S_k or Si_k
                    ]
                )  # Si_ka1 not Si_K
                logger.info(f"Element line info: {self.elementinfo_list}")
            else:
                element = self.selected_element  # for pileup peaks
                self.elementinfo_list = sorted(
                    [e for e in list(self.param_model.param_new.keys()) if element.replace("-", "_") in e]
                )
                logger.info(f"User defined or pileup peak info: {self.elementinfo_list}")
        else:
            self.elementinfo_list = []

    def select_index_by_eline_name(self, eline_name):
        # Select the element by name. If element is selected, then the ``elementinfo_list`` with
        #   names of parameters is created. Originally the element could only be selected
        #   by its index in the list (from dialog box ``ElementEdit``. This function is
        #   used by interface components for editing parameters of user defined peaks.
        if eline_name in self.param_model.element_list:
            # This will fill the list ``self.elementinfo_list``
            self.selected_index = self.param_model.element_list.index(eline_name) + 1
        else:
            raise Exception(f"Line '{eline_name}' is not in the list of selected element lines.")

    def apply_default_param(self):
        """
        Update parameters with default parameters, also update element list.
        """
        # Save currently selected element name
        selected_element = self.selected_element
        self.selected_index = 0

        element_list = self.param_model.param_new["non_fitting_values"]["element_list"]
        element_list = [e.strip(" ") for e in element_list.split(",")]
        element_list = [_ for _ in element_list if _]  # Get rid of empty strings in the list
        self.param_model.element_list = element_list

        # Update 'self.param_model.EC'
        # self.param_model.create_spectrum_from_param_dict()

        self.update_element_info()

        # Update the index in case the selected emission line disappeared from the list
        self.update_selected_index(selected_element=selected_element, element_list_new=element_list)

        # global parameters
        # for GUI purpose only
        # if we do not clear the list first, there is not update on the GUI
        self.global_param_list = []
        self.global_param_list = sorted(
            [k for k in self.param_model.param_new.keys() if k == k.lower() and k != "non_fitting_values"]
        )

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

    def filepath_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.hdf_path = change["value"]

    @observe("fit_strategy1")
    def update_strategy1(self, change):
        self.all_strategy.update({"strategy1": change["value"]})
        if change["value"]:
            logger.info("Setting strategy (preset) for step 1: {}".format(fit_strategy_list[change["value"] - 1]))

    @observe("fit_strategy2")
    def update_strategy2(self, change):
        self.all_strategy.update({"strategy2": change["value"]})
        if change["value"]:
            logger.info("Setting strategy (preset) for step 2: {}".format(fit_strategy_list[change["value"] - 1]))

    @observe("fit_strategy3")
    def update_strategy3(self, change):
        self.all_strategy.update({"strategy3": change["value"]})
        if change["value"]:
            logger.info("Setting strategy (preset) for step 3: {}".format(fit_strategy_list[change["value"] - 1]))

    @observe("fit_strategy4")
    def update_strategy4(self, change):
        self.all_strategy.update({"strategy4": change["value"]})
        if change["value"]:
            logger.info("Strategy at step 4 is: {}".format(fit_strategy_list[change["value"] - 1]))

    @observe("fit_strategy5")
    def update_strategy5(self, change):
        self.all_strategy.update({"strategy5": change["value"]})
        if change["value"]:
            logger.info("Strategy at step 5 is: {}".format(fit_strategy_list[change["value"] - 1]))

    def update_param_with_result(self):
        update_parameter_dict(self.param_model.param_new, self.fit_result)

    def define_range(self):
        """
        Cut x range according to values define in param_dict.
        """
        lowv = self.param_model.param_new["non_fitting_values"]["energy_bound_low"]["value"]
        highv = self.param_model.param_new["non_fitting_values"]["energy_bound_high"]["value"]
        self.x0, self.y0 = define_range(
            self.io_model.data,
            lowv,
            highv,
            self.param_model.param_new["e_offset"]["value"],
            self.param_model.param_new["e_linear"]["value"],
        )

    def get_background(self):
        self.bg = snip_method_numba(
            self.y0,
            self.param_model.param_new["e_offset"]["value"],
            self.param_model.param_new["e_linear"]["value"],
            self.param_model.param_new["e_quadratic"]["value"],
            width=self.param_model.param_new["non_fitting_values"]["background_width"],
        )

    def get_profile(self):
        """
        Calculate profile based on current parameters.
        """
        # self.define_range()

        #  Do nothing if no data is loaded
        if self.x0 is None or self.y0 is None:
            return

        self.cal_x, self.cal_spectrum, area_dict = calculate_profile(
            self.x0, self.y0, self.param_model.param_new, self.param_model.element_list
        )
        #  add escape peak
        if self.param_model.param_new["non_fitting_values"]["escape_ratio"] > 0:
            self.cal_spectrum["escape"] = trim_escape_peak(
                self.io_model.data, self.param_model.param_new, len(self.y0)
            )

        self.cal_y = np.zeros(len(self.cal_x))
        for k, v in self.cal_spectrum.items():
            self.cal_y += v

        self.residual = self.cal_y - self.y0

    def fit_data(self, x0, y0, *, init_params=True):
        fit_num = self.fit_num
        ftol = self.ftol
        c_weight = 1  # avoid zero point

        params = copy.deepcopy(self.param_model.param_new)
        # Initialize parameters to some large default value. This increases the rate of convergence
        #   and prevents optimization algorithm from getting stuck. There could be better solution,
        #   but this seems to be functional workaround.
        if init_params:
            for k, p in params.items():
                if "area" in k:
                    p["value"] = 1000

        MS = ModelSpectrum(params, self.param_model.element_list)
        MS.assemble_models()

        # weights = 1/(c_weight + np.abs(y0))
        weights = 1 / np.sqrt(c_weight + np.abs(y0))
        # weights /= np.sum(weights)
        result = MS.model_fit(x0, y0, weights=weights, maxfev=fit_num, xtol=ftol, ftol=ftol, gtol=ftol)
        self.fit_x = (
            result.values["e_offset"] + result.values["e_linear"] * x0 + result.values["e_quadratic"] * x0**2
        )
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

        # PC = ParamController(self.param_dict, self.param_model.element_list)
        # self.param_dict = PC.params

        if self.param_model.param_new["non_fitting_values"]["escape_ratio"] > 0:
            self.es_peak = trim_escape_peak(self.io_model.data, self.param_model.param_new, self.y0.size)
            y0 = self.y0 - self.bg - self.es_peak
        else:
            y0 = self.y0 - self.bg

        t0 = time.time()
        self.fit_info = (
            "Spectrum fitting of the sum spectrum (incident energy "
            f"{self.param_model.param_new['coherent_sct_energy']['value']})."
        )
        # app.processEvents()
        # logger.info('-------- '+self.fit_info+' --------')

        # Parameters should be initialized only once
        init_params = True
        for k, v in self.all_strategy.items():
            if v:
                strat_name = fit_strategy_list[v - 1]
                # self.fit_info = 'Fit with {}: {}'.format(k, strat_name)

                logger.info(self.fit_info)
                strategy = extract_strategy(self.param_model.param_new, strat_name)
                #  register the strategy and extend the parameter list
                #  to cover all given elements
                register_strategy(strat_name, strategy)
                set_parameter_bound(self.param_model.param_new, strat_name)

                self.fit_data(self.x0, y0, init_params=init_params)
                init_params = False

                self.update_param_with_result()

                # The following is a patch for rare cases when fitting results in negative
                #   areas for some emission lines. These are typically non-existent lines, but
                #   they should not be automatically eliminated from the list. To prevent
                #   elimination, set the area to some small positive value.
                for key, val in self.param_model.param_new.items():
                    if key.endswith("_area") and val["value"] <= 0.0:
                        _small_value_for_area = 0.1
                        logger.warning(
                            f"Fitting resulted in negative value for '{key}' ({val['value']}). \n"
                            f"    In order to continue using the emission line in future computations, "
                            f"the fitted area is set to a small value ({_small_value_for_area}).\n    Delete "
                            f"the emission line from the list if you know it is not present in "
                            f"the sample."
                        )
                        val["value"] = _small_value_for_area  # Some small number

                #  calculate r2
                self.r2 = cal_r2(y0, self.fit_y)
                self.assign_fitting_result()
                # app.processEvents()

        t1 = time.time()
        logger.warning("Time used for summed spectrum fitting is : {}".format(t1 - t0))

        self.comps.clear()
        comps = self.fit_result.eval_components(x=self.x0)
        self.comps = combine_lines(comps, self.param_model.element_list, self.bg)

        if self.param_model.param_new["non_fitting_values"]["escape_ratio"] > 0:
            self.fit_y += self.bg + self.es_peak
            self.comps["escape"] = self.es_peak
        else:
            self.fit_y += self.bg

        self.save_result()
        self.assign_fitting_result()
        self.fit_info = "Summed spectrum fitting is done!"
        logger.info("-------- " + self.fit_info + " --------")

    def compute_current_rfactor(self, save_fit=True):
        """
        Compute current R-factor value. The fitted array is selected
        based on `save_fit`. The same arrays are selected as in `output_summed_data_fit`

        Parameters
        ----------
        save_fit: bool
            True - use total spectrum fitting data (available after fitting was done),
            False - use weighted spectral components with weights equal to current parameters.

        Returns
        -------
        float or None
            Value of R-factor or None if R-factor can not be computed.
        """
        rf = None
        # R-factor is for visualization purposes only, so display 0 if it can not be computed.
        if save_fit:
            if (self.y0 is not None) and (self.fit_y is not None) and (self.y0.shape == self.fit_y.shape):
                rf = rfactor(self.y0, self.fit_y)
        else:
            if (
                (self.param_model.y0 is not None)
                and (self.param_model.total_y is not None)
                and (self.param_model.y0.shape == self.param_model.total_y.shape)
            ):
                rf = rfactor(self.param_model.y0, self.param_model.total_y)
        return rf

    def output_summed_data_fit(self, save_fit=True, directory=None):
        """
        Save energy, summed data and fitting curve to a file.
        """
        directory = directory or os.path.dirname(self.hdf_path)

        xx = None
        if self.x0 is not None:
            a0, a1, a2 = (
                self.param_model.param_new["e_offset"]["value"],
                self.param_model.param_new["e_linear"]["value"],
                self.param_model.param_new["e_quadratic"]["value"],
            )
            xx = a0 + self.x0 * a1 + self.x0**2 * a2
        if save_fit:
            logger.info("Saving spectrum after total spectrum fitting.")
            if (xx is None) or (self.y0 is None) or (self.fit_y is None):
                msg = "Not enough data to save spectrum/fit data. Total spectrum fitting was not run."
                raise RuntimeError(msg)
            data = np.array([self.x0, self.y0, self.fit_y])
        else:
            logger.info("Saving spectrum based on loaded or estimated parameters.")
            if (xx is None) or (self.y0 is None) or (self.param_model.total_y is None):
                msg = "Not enough data to save spectrum/fit data based on loaded or estimated parameters."
                raise RuntimeError(msg)
            data = np.array([xx, self.y0, self.param_model.total_y])

        output_fit_name = self.data_title + "_summed_spectrum_fit.txt"
        fpath = os.path.join(directory, output_fit_name)
        np.savetxt(fpath, data.T)
        logger.info(f"Spectrum fit data is saved to file '{fpath}'")

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
            pixel_fit = "nnls"
        elif self.pixel_fit_method == 1:
            pixel_fit = "nonlinear"

        logger.info(
            "-------- Fitting of single pixels starts (incident_energy "
            f"{self.param_model.param_new['coherent_sct_energy']['value']} keV) --------"
        )
        t0 = time.time()
        self.pixel_fit_info = "Pixel fitting is in process."

        # app.processEvents()
        self.result_map, calculation_info = single_pixel_fitting_controller(
            self.io_model.data_all,
            self.param_model.param_new,
            method=pixel_fit,
            pixel_bin=pixel_bin,
            raise_bg=raise_bg,
            comp_elastic_combine=comp_elastic_combine,
            linear_bg=linear_bg,
            use_snip=use_snip,
            bin_energy=bin_energy,
        )

        t1 = time.time()
        logger.info("Time used for pixel fitting is : {}".format(t1 - t0))

        #  get fitted spectrum and save them to figs
        if self.save_point is True:
            self.pixel_fit_info = "Saving output ..."
            # app.processEvents()
            elist = calculation_info["fit_name"]
            matv = calculation_info["regression_mat"]
            results = calculation_info["results"]
            # fit_range = calculation_info['fit_range']
            x = calculation_info["energy_axis"]
            x = (
                self.param_model.param_new["e_offset"]["value"]
                + self.param_model.param_new["e_linear"]["value"] * x
                + self.param_model.param_new["e_quadratic"]["value"] * x**2
            )
            data_fit = calculation_info["input_data"]
            data_sel_indices = calculation_info["data_sel_indices"]

            p1 = [self.point1v, self.point1h]
            p2 = [self.point2v, self.point2h]

            if self.point2v > 0 or self.point2h > 0:
                prefix_fname = os.path.basename(self.hdf_path).split(".")[0]
                output_folder = os.path.join(os.path.dirname(self.hdfpath), prefix_fname + "_pixel_fit")
                if os.path.exists(output_folder) is False:
                    os.mkdir(output_folder)
                save_fitted_fig(
                    x,
                    matv,
                    results[:, :, 0 : len(elist)],
                    p1,
                    p2,
                    data_fit,
                    data_sel_indices,
                    self.param_model.param_new,
                    output_folder,
                    use_snip=use_snip,
                )

            # the output movie are saved as the same name
            # need to define specific name for prefix, to be updated
            # save_fitted_as_movie(x, matv, results[:, :, 0:len(elist)],
            #                      p1, p2,
            #                      data_fit, self.param_dict,
            #                      os.path.dirname(self.hdf_name), prefix=prefix_fname, use_snip=use_snip)
            logger.info("Done with saving fitting plots.")
        try:
            self.save2Dmap_to_hdf(calculation_info=calculation_info, pixel_fit=pixel_fit)
            self.pixel_fit_info = "Pixel fitting is done!"
            # app.processEvents()
        except ValueError:
            logger.warning("Fitting result can not be saved to h5 file.")
        except IOError as ex:
            logger.warning(f"{ex}")
        logger.info("-------- Fitting of single pixels is done! --------")

    def save_pixel_fitting_to_db(self):
        """Save fitting results to analysis store"""
        from .data_to_analysis_store import save_data_to_db

        doc = {}
        doc["param"] = self.param_model.param_new
        doc["exp"] = self.io_model.data
        doc["fitted"] = self.fit_y
        save_data_to_db(self.runid, self.result_map, doc)

    def save2Dmap_to_hdf(self, *, calculation_info=None, pixel_fit="nnls"):
        """
        Save fitted 2D map of elements into hdf file after fitting is done. User
        can choose to interpolate the image based on x,y position or not.

        Parameters
        ----------
        pixel_fit : str
            If nonlinear is chosen, more information needs to be saved.
        """

        prefix_fname = os.path.basename(self.hdf_path).split(".")[0]
        if len(prefix_fname) == 0:
            prefix_fname = "tmp"

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
        if pixel_fit == "nonlinear":
            error_map = calculation_info["error_map"]
            save_fitdata_to_hdf(
                self.hdf_path,
                error_map,
                datapath=inner_path,
                data_saveas="xrf_fit_error",
                dataname_saveas="xrf_fit_error_name",
            )

    # TODO: the function 'calculate_roi_sum" should be converted to work with
    #   DASK arrays and HDF5 datasets if needed, otherwise it can be removed.
    #   The function was called at the exit of dialog box 'OutputSetup'
    #   (button 'Output Setup') in the Fit tab. This option is eliminated
    #   from the new interface.
    """
    def calculate_roi_sum(self):
        if self.roi_sum_opt['status'] is True:
            low = int(self.roi_sum_opt['low']*100)
            high = int(self.roi_sum_opt['high']*100)
            logger.info('ROI sum at range ({}, {})'.format(self.roi_sum_opt['low'],
                                                           self.roi_sum_opt['high']))
            sumv = np.sum(self.io_model.data_all[:, :, low:high], axis=2)
            self.result_map['ROI'] = sumv
            # save to hdf again, this is not optimal
            self.save2Dmap_to_hdf()
    """

    def get_latest_single_pixel_fitting_data(self):
        r"""
        Returns the latest results of single pixel fitting. The returned results include
        the name of the scaler (None if no scaler is selected) and the dictionary of
        the computed XRF maps for selected emission lines.
        """

        # Find the selected scaler name. Scaler is None if not scaler is selected.
        scaler_name = None
        if self.scaler_index > 0:
            scaler_name = self.scaler_keys[self.scaler_index - 1]

        # Result map. If 'None', then no fitting results exists.
        #   Single pixel fitting must be run
        result_map = self.result_map.copy()

        return result_map, scaler_name

    def get_selected_fitted_map_data(self):
        """
        Returns fitting data selected in XRF Maps tab.

        Returned channel name should be 'sum', 'det1', 'det2' etc. If the selected
        dataset contains maps computed based on ROIs or only positions and scalers,
        then returned channel name is "".
        """
        image_title = self.img_title
        plotted_dict = self.dict_to_plot.copy()

        # Recover channel name
        channel_name = ""  # No name (dictionary with position or scaler data is selected)
        if image_title.endswith("_fit"):
            m = re.search(r"_det\d+_fit$", image_title)
            if m:
                channel_name = m.group(0).split("_")[1].lower()
            else:
                channel_name = "sum"

        # Find the selected scaler name. Scaler is None if not scaler is selected.
        scaler_name = None
        if self.scaler_index > 0:
            scaler_name = self.scaler_keys[self.scaler_index - 1]

        return plotted_dict, channel_name, scaler_name

    def output_2Dimage(
        self,
        *,
        results_path=None,
        dataset_name=None,
        scaler_name=None,
        interpolate_on=None,
        quant_norm_on=None,
        param_quant_analysis=None,
        file_format="tiff",
    ):
        """Read data from h5 file and save them into either tiff or txt.

        Parameters
        ----------
        results_path: str
            path to the root directory where data is to be saved. Data will be saved
            in subdirectories inside this directory.
        dataset_name: str
            name of the dataset (key in the dictionary `self.io_model.img_dict`
        scaler_name: str
            name of the scaler, must exist in `self.io_model.img_dict`
        interpolate_on: bool
            turns interpolation of images to uniform grid ON or OFF
        quant_norm_on: bool
            turns quantitative normalization of images on or off
        param_quant_analysis: ParamQuantitativeAnalysis
            class that contains methods for quantitative normalization
        file_format: str
            output file format, supported values: "tiff" or "txt"
        """
        supported_file_formats = ("tiff", "txt")
        file_format = file_format.lower()
        if file_format not in supported_file_formats:
            raise ValueError(
                f"The value 'file_format={file_format}' is not supported. "
                f"Supported values: {supported_file_formats} "
            )

        if file_format == "tiff":
            dir_prefix = "output_tiff_"
        else:
            dir_prefix = "output_txt_"

        dataset_dict = self.io_model.img_dict[dataset_name]

        _post_name_folder = "_".join(self.data_title.split("_")[:-1])
        output_n = dir_prefix + _post_name_folder
        output_dir = os.path.join(results_path, output_n)

        # self.io_model.img_dict contains ALL loaded datasets, including a separate "positions" dataset
        if "positions" in self.io_model.img_dict:
            positions_dict = self.io_model.img_dict["positions"]
        else:
            positions_dict = {}

        if scaler_name is not None:
            logger.info(f"*** NORMALIZED data is saved. Scaler: '{scaler_name}' ***")

        # Scalers are located in a separate dataset in 'img_dict'. They are also referenced
        #   in each '_fit' dataset (and in the selected dataset 'self.dict_to_plot')
        #   The list of scaler names is used to avoid attaching the detector channel name
        #   to file names that contain scaler data (scalers typically do not depend on
        #   the selection of detector channels.
        scaler_dsets = [_ for _ in self.io_model.img_dict.keys() if re.search(r"_scaler$", _)]
        if scaler_dsets:
            scaler_name_list = list(self.io_model.img_dict[scaler_dsets[0]].keys())
        else:
            scaler_name_list = None

        output_data(
            output_dir=output_dir,
            interpolate_to_uniform_grid=interpolate_on,
            dataset_name=dataset_name,
            quant_norm=quant_norm_on,
            param_quant_analysis=param_quant_analysis,
            dataset_dict=dataset_dict,
            positions_dict=positions_dict,
            file_format=file_format,
            scaler_name=scaler_name,
            scaler_name_list=scaler_name_list,
        )

    def save_result(self, fname=None):
        """
        Save fitting results.

        Parameters
        ----------
        fname : str, optional
            name of output file
        """
        if not fname:
            fname = self.data_title + "_out.txt"
        filepath = os.path.join(os.path.dirname(self.hdf_path), fname)

        area_list = []
        for v in list(self.fit_result.params.keys()):
            if "ka1_area" in v or "la1_area" in v or "ma1_area" in v or "amplitude" in v:
                area_list.append(v)
        try:
            with open(filepath, "w") as myfile:
                myfile.write("\n {:<10} \t {} \t {}".format("name", "summed area", "error in %"))
                for k, v in self.comps.items():
                    if k == "background":
                        continue
                    for name in area_list:
                        if k.lower() in name.lower():
                            std_error = self.fit_result.params[name].stderr
                            if std_error is None:
                                # Do not print 'std_error' if it is not computed by lmfit
                                errorv_s = ""
                            else:
                                errorv = std_error / (self.fit_result.params[name].value + 1e-8)
                                errorv *= 100
                                errorv = np.round(errorv, 3)
                                errorv_s = f"{errorv}%"
                            myfile.write("\n {:<10} \t {} \t {}".format(k, np.round(np.sum(v), 3), errorv_s))
                myfile.write("\n\n")

                # Print the report from lmfit
                # Remove strings (about 50%) on the variables that stayed at initial value
                report = lmfit.fit_report(self.fit_result, sort_pars=True)
                report = report.split("\n")
                report = [s for s in report if "at initial value" not in s and "##" not in s]
                report = "\n".join(report)
                myfile.write(report)

                logger.warning("Results are saved to {}".format(filepath))
        except FileNotFoundError:
            print("Summed spectrum fitting results are not saved.")

    def update_element_info(self):
        # need to clean list first, in order to refresh the list in GUI
        self.selected_index = 0
        self.elementinfo_list = []

        logger.info("The full list for fitting is {}".format(self.param_model.element_list))


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
            e_temp = e.split("_")[0]
            intensity = 0
            for k, v in components.items():
                if (e_temp in k) and (e not in k):
                    intensity += v
            new_components[e] = intensity
        elif "user" in e.lower():
            for k, v in components.items():
                if e in k:
                    new_components[e] = v
        else:
            comp_name = "pileup_" + e.replace("-", "_") + "_"  # change Si_K-Si_K to Si_K_Si_K
            new_components[e] = components[comp_name]

    # add background and elastic
    new_components["background"] = background
    new_components["compton"] = components["compton"]
    new_components["elastic"] = components["elastic_"]
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
    return {k: v[name] for k, v in param_new.items() if k != "non_fitting_values"}


def define_param_bound_type(param, strategy_list=["adjust_element2, adjust_element3"], b_type="fixed"):
    param_new = copy.deepcopy(param)
    for k, v in param_new.items():
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
        for i in np.arange(d_shape[0] - 1):
            for j in np.arange(d_shape[1] - 1):
                new_data[i, j, :] += new_data[i + 1, j, :] + new_data[i, j + 1, :] + new_data[i + 1, j + 1, :]
        new_data[:-1, :-1, :] /= nearest_n

    if nearest_n == 9:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                new_data[1:-1, 1:-1, :] += data[1 + i : d_shape[0] - 1 + i, 1 + j : d_shape[1] - 1 + j, :]

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
        d_shape = np.array([data.shape[0], data.shape[1]]) / bin_size

        data_new = np.zeros([d_shape[0], d_shape[1]])
        for i in np.arange(d_shape[0]):
            for j in np.arange(d_shape[1]):
                data_new[i, j] = np.sum(
                    data[i * bin_size : i * bin_size + bin_size, j * bin_size : j * bin_size + bin_size]
                )
    elif data.ndim == 3:
        d_shape = np.array([data.shape[0], data.shape[1]]) / bin_size

        data_new = np.zeros([d_shape[0], d_shape[1], data.shape[2]])
        for i in np.arange(d_shape[0]):
            for j in np.arange(d_shape[1]):
                data_new[i, j, :] = np.sum(
                    data[i * bin_size : i * bin_size + bin_size, j * bin_size : j * bin_size + bin_size, :],
                    axis=(0, 1),
                )
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
        conv_f = [1.0 / 2, 1.0 / 2]
    if width == 3:
        conv_f = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[1]):
            data_new[i, j, :] = np.convolve(data_new[i, j, :], conv_f, mode="same")

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
            new_len = data.shape[0] / bin_step
            m1 = data[::2, :]
            m2 = data[1::2, :]
            if sum_data is True:
                return (m1[:new_len, :] + m2[:new_len, :]) / bin_step
            else:
                return m1[:new_len, :]
        elif bin_step == 3:
            new_len = data.shape[0] / bin_step
            m1 = data[::3, :]
            m2 = data[1::3, :]
            m3 = data[2::3, :]
            if sum_data is True:
                return (m1[:new_len, :] + m2[:new_len, :] + m3[:new_len, :]) / bin_step
            else:
                return m1[:new_len, :]
        elif bin_step == 4:
            new_len = data.shape[0] / bin_step
            m1 = data[::4, :]
            m2 = data[1::4, :]
            m3 = data[2::4, :]
            m4 = data[3::4, :]
            if sum_data is True:
                return (m1[:new_len, :] + m2[:new_len, :] + m3[:new_len, :] + m4[:new_len, :]) / bin_step
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
        new_len = data.shape[2] / 2
        new_data1 = data[:, :, ::2]
        new_data2 = data[:, :, 1::2]
        if sum_data is True:
            return (new_data1[:, :, :new_len] + new_data2[:, :, :new_len]) / bin_step
        else:
            return new_data1[:, :, :new_len]
    elif bin_step == 3:
        new_len = data.shape[2] / 3
        new_data1 = data[:, :, ::3]
        new_data2 = data[:, :, 1::3]
        new_data3 = data[:, :, 2::3]
        if sum_data is True:
            return (new_data1[:, :, :new_len] + new_data2[:, :, :new_len] + new_data3[:, :, :new_len]) / bin_step
        else:
            return new_data1[:, :, :new_len]
    elif bin_step == 4:
        new_len = data.shape[2] / 4
        new_data1 = data[:, :, ::4]
        new_data2 = data[:, :, 1::4]
        new_data3 = data[:, :, 2::4]
        new_data4 = data[:, :, 3::4]
        if sum_data is True:
            return (
                new_data1[:, :, :new_len]
                + new_data2[:, :, :new_len]
                + new_data3[:, :, :new_len]
                + new_data4[:, :, :new_len]
            ) / bin_step
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
    sse = np.sum((y - y_cal) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - sse / sst


def calculate_area(e_select, matv, results, param, first_peak_area=False):
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
        get overall peak area or only the first peak area, such as Ar_Ka1

    Returns
    -------
    dict :
        dict of each 2D elemental distribution
    """
    total_list = e_select + ["snip_bkg", "r_factor", "sel_cnt", "total_cnt"]
    mat_sum = np.sum(matv, axis=0)

    if len(total_list) != results.shape[2]:
        logger.error(
            f"The number of features in the list ({len(total_list)} "
            f"is not equal to the number of features in the 'results' array "
            f"({results.shape[2]}). This issue needs to be investigated,"
            f"since some of the generated XRF maps may be invalid."
        )

    result_map = dict()
    for i, eline in enumerate(total_list):
        if i < len(e_select):
            # We are processing the component due to emission line (and may be
            #   additional constant spectrum representing background)
            if first_peak_area is not True:
                result_map.update({total_list[i]: results[:, :, i] * mat_sum[i]})
            else:
                if total_list[i] not in K_LINE + L_LINE + M_LINE:
                    ratio_v = 1
                else:
                    ratio_v = get_branching_ratio(total_list[i], param["coherent_sct_energy"]["value"])
                result_map.update({total_list[i]: results[:, :, i] * mat_sum[i] * ratio_v})
        else:
            # We are just copying additional computed data
            result_map.update({eline: results[:, :, i]})

    return result_map


def save_fitted_fig(
    x_v, matv, results, p1, p2, data_all, data_sel_indices, param_dict, result_folder, use_snip=False
):
    """
    Save single pixel fitting results to figs.
    `data_all` can be numpy array, Dask array or RawHDF5Dataset.
    """
    logger.info(f"Saving plots of the fitted data to file. Selection: {tuple(p1)} .. {tuple(p2)}")

    # Convert the 'data_all', which can be numpy array, Dask array or
    #   RawHDF5Dataset into Dask array, so that it could be treated uniformly
    data_all_dask, file_obj = prepare_xrf_map(data_all)
    # Selection (indices of the processed interval) of `data_all` along axis 2
    d_start, d_stop = data_sel_indices
    # 'file_obj' must remain alive until the function exits

    low_limit_v = 0.5

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    max_v = da.max(data_all_dask[p1[0] : p2[0], p1[1] : p2[1], d_start:d_stop]).compute()

    fitted_sum = None
    for m in range(p1[0], p2[0]):
        for n in range(p1[1], p2[1]):
            data_y = data_all_dask[m, n, d_start:d_stop].compute()

            fitted_y = np.sum(matv * results[m, n, :], axis=1)
            if use_snip is True:
                bg = snip_method_numba(
                    data_y,
                    param_dict["e_offset"]["value"],
                    param_dict["e_linear"]["value"],
                    param_dict["e_quadratic"]["value"],
                    width=param_dict["non_fitting_values"]["background_width"],
                )
                fitted_y += bg

            if fitted_sum is None:
                fitted_sum = fitted_y
            else:
                fitted_sum += fitted_y
            ax.cla()
            ax.set_title("Single pixel fitting for point ({}, {})".format(m, n))
            ax.set_xlabel("Energy [keV]")
            ax.set_ylabel("Counts")
            ax.set_ylim(low_limit_v, max_v * 2)

            ax.semilogy(x_v, data_y, label="exp", linestyle="", marker=".")
            ax.semilogy(x_v, fitted_y, label="fit")

            ax.legend()
            output_path = os.path.join(result_folder, "data_out_" + str(m) + "_" + str(n) + ".png")
            plt.savefig(output_path)

    ax.cla()
    sum_y = da.sum(data_all_dask[p1[0] : p2[0], p1[1] : p2[1], d_start:d_stop], axis=(0, 1)).compute()
    ax.set_title("Summed spectrum from point ({},{}) to ({},{})".format(p1[0], p1[1], p2[0], p2[1]))
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_ylim(low_limit_v, np.max(sum_y) * 2)
    ax.semilogy(x_v, sum_y, label="exp", linestyle="", marker=".")
    ax.semilogy(x_v, fitted_sum, label="fit", color="red")

    ax.legend()
    fit_sum_name = "pixel_sum_" + str(p1[0]) + "-" + str(p1[1]) + "_" + str(p2[0]) + "-" + str(p2[1]) + ".png"
    output_path = os.path.join(result_folder, fit_sum_name)
    plt.savefig(output_path)

    logger.info(f"Fitted data is saved to the directory '{result_folder}'")


def save_fitted_as_movie(
    x_v, matv, results, p1, p2, data_all, param_dict, result_folder, prefix=None, use_snip=False, dpi=150
):
    """
    Create movie to save single pixel fitting resutls.
    """
    total_n = data_all.shape[1] * p2[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_aspect("equal")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    max_v = np.max(data_all[p1[0] : p2[0], p1[1] : p2[1], :])
    ax.set_ylim([0, 1.1 * max_v])

    (l1,) = ax.plot(x_v, x_v, label="exp", linestyle="-", marker=".")
    (l2,) = ax.plot(x_v, x_v, label="fit", color="red", linewidth=2)

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

        fitted_y = np.sum(matv * results[m, n, :], axis=1)
        if use_snip is True:
            bg = snip_method_numba(
                data_y,
                param_dict["e_offset"]["value"],
                param_dict["e_linear"]["value"],
                param_dict["e_quadratic"]["value"],
                width=param_dict["non_fitting_values"]["background_width"],
            )
            fitted_y += bg

        ax.set_title("Single pixel fitting for point ({}, {})".format(m, n))
        # ax.set_ylim(low_limit_v, max_v*2)
        l1.set_ydata(data_y)
        l2.set_ydata(fitted_y)
        return l1, l2

    writer = animation.writers["ffmpeg"](fps=30)
    ani = animation.FuncAnimation(fig, update_img, plist)
    if prefix:
        output_file = prefix + "_pixel.mp4"
    else:
        output_file = "fit_pixel.mp4"
    output_p = os.path.join(result_folder, output_file)
    ani.save(output_p, writer=writer, dpi=dpi)


def fit_per_line_nnls(row_num, data, matv, param, use_snip, num_data, num_feature):
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
            bg = snip_method_numba(
                data[i, :],
                param["e_offset"]["value"],
                param["e_linear"]["value"],
                param["e_quadratic"]["value"],
                width=param["non_fitting_values"]["background_width"],
            )
            y = data[i, :] - bg
            bg_sum = np.sum(bg)

        else:
            y = data[i, :]

        result, res = nnls_fit(y, matv, weights=None)
        sst = np.sum((y - np.mean(y)) ** 2)
        if not math.isclose(sst, 0, abs_tol=1e-20):
            r2_adjusted = 1 - res / (num_data - num_feature - 1) / (sst / (num_data - 1))
        else:
            # This happens if all elements of 'y' are equal (most likely == 0)
            r2_adjusted = 0
        result = list(result) + [bg_sum, r2_adjusted]
        out.append(result)
    return np.array(out)


def fit_pixel_multiprocess_nnls(exp_data, matv, param, use_snip=False, lambda_reg=0.0):
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

    logger.info("cpu count: {}".format(num_processors_to_use))

    # TODO: find better permanent solution for issue with multiprocessing on Catalina MacOS
    # =======================================================================================
    # The following is a temporary patch for the issue with multiprocessing on Catalina MacOS.
    # The issue is solved by disabling multiprocessing if current OS is Catalina MacOS.
    # The code will change when the better solution is found.

    # Check if the OS is MacOS Catalina (10.15) or older
    disable_multiprocessing = False
    try:
        os_version = platform.mac_ver()[0]
        # 'os_version' is an empty string if the system is not MacOS
        # os_version has format similar to '10.15.3', MacOS Catalina is 10.15
        if os_version and (LooseVersion(os_version) >= LooseVersion("10.15")):
            disable_multiprocessing = True
    except Exception as ex:
        logger.error(f"Error occurred while checking the version of MacOS: {ex}")

    if disable_multiprocessing:
        pool = multiprocessing.pool.ThreadPool(num_processors_to_use)
        logger.warning(
            "Multiprocessing is currently not supported when running PyXRF in MacOS Catalina. "
            "Computations are executed in multithreading mode instead."
        )
    else:
        pool = multiprocessing.Pool(num_processors_to_use)
        logger.info("Computations are executed in multiprocessing mode.")

    n_data, n_feature = matv.shape

    if lambda_reg > 0:
        logger.info("nnls fit with regularization term, lambda={}".format(lambda_reg))
        diag_m = np.diag(np.ones(n_feature)) * np.sqrt(lambda_reg)
        matv = np.concatenate((matv, diag_m), axis=0)
        exp_tmp = np.zeros([exp_data.shape[0], exp_data.shape[1], n_feature])
        exp_data = np.concatenate((exp_data, exp_tmp), axis=2)

    result_pool = [
        pool.apply_async(fit_per_line_nnls, (n, exp_data[n, :, :], matv, param, use_snip, n_data, n_feature))
        for n in range(exp_data.shape[0])
    ]

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
    return np.sum(vals["a{}".format(i)] * reg_mat[:, i] for i in range(len(vals)))


def residual_nonlinear_fit(pars, x, data=None, reg_mat=None):
    return spectrum_nonlinear_fit(pars, x, reg_mat) - data


def fit_pixel_nonlinear_per_line(row_num, data, x0, param, reg_mat, use_snip):  # c_weight, fit_num, ftol):

    # c_weight = 1
    # fit_num = 100
    # ftol = 1e-3

    elist = param["non_fitting_values"]["element_list"].split(", ")
    elist = [e.strip(" ") for e in elist]

    # LinearModel = lmfit.Model(simple_spectrum_fun_for_nonlinear)
    # for i in np.arange(reg_mat.shape[0]):
    #     LinearModel.set_param_hint('a'+str(i), value=0.1, min=0, vary=True)

    logger.info("Row number at {}".format(row_num))
    out = []
    snip_bg = 0
    for i in range(data.shape[0]):
        if use_snip is True:
            bg = snip_method_numba(
                data[i, :],
                param["e_offset"]["value"],
                param["e_linear"]["value"],
                param["e_quadratic"]["value"],
                width=param["non_fitting_values"]["background_width"],
            )
            y0 = data[i, :] - bg
            snip_bg = np.sum(bg)
        else:
            y0 = data[i, :]

        fit_params = lmfit.Parameters()
        for i in range(reg_mat.shape[1]):
            fit_params.add("a" + str(i), value=1.0, min=0, vary=True)

        result = lmfit.minimize(
            residual_nonlinear_fit, fit_params, args=(x0,), kws={"data": y0, "reg_mat": reg_mat}
        )

        # result = MS.model_fit(x0, y0,
        #                       weights=1/np.sqrt(c_weight+y0),
        #                       maxfev=fit_num,
        #                       xtol=ftol, ftol=ftol, gtol=ftol)
        # namelist = list(result.keys())
        temp = {}
        temp["value"] = [result.params[v].value for v in list(result.params.keys())]
        temp["err"] = [result.params[v].stderr for v in list(result.params.keys())]
        temp["snip_bg"] = snip_bg
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
    logger.info("cpu count: {}".format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    # fit_params = lmfit.Parameters()
    # for i in range(reg_mat.shape[1]):
    #     fit_params.add('a'+str(i), value=1.0, min=0, vary=True)

    result_pool = [
        pool.apply_async(fit_pixel_nonlinear_per_line, (n, data[n, :, :], x, param, reg_mat, use_snip))
        for n in range(data.shape[0])
    ]

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
    area_dict["snip_bg"] = np.zeros([len(fit_results), len(fit_results[0])])
    weights_mat = np.zeros([len(fit_results), len(fit_results[0]), len(error_dict)])

    for i in range(len(fit_results)):
        for j in range(len(fit_results[0])):
            for m, v in enumerate(area_dict.keys()):
                if v == "snip_bg":
                    area_dict[v][i, j] = fit_results[i][j]["snip_bg"]
                else:
                    area_dict[v][i, j] = fit_results[i][j]["value"][m]
                    error_dict[v][i, j] = fit_results[i][j]["err"][m]
                    weights_mat[i, j, m] = fit_results[i][j]["value"][m]

    for i, v in enumerate(area_dict.keys()):
        if v == "snip_bg":
            continue
        area_dict[v] *= mat_sum[i]
        error_dict[v] *= mat_sum[i]

    return area_dict, error_dict, weights_mat


def single_pixel_fitting_controller(
    input_data,
    parameter,
    incident_energy=None,
    method="nnls",
    pixel_bin=0,
    raise_bg=0,
    comp_elastic_combine=False,
    linear_bg=False,
    use_snip=True,
    bin_energy=1,
    dask_client=None,
):
    """
    Parameters
    ----------
    input_data: array
        3D array of spectrum
    parameter: dict
        parameter for fitting
    incident_energy: float, optional
        incident beam energy in KeV
    method: str, optional
        fitting method, default as nnls
    pixel_bin: int, optional
        bin pixel as 2by2, or 3by3
    raise_bg: int, optional
        add a constant value to each spectrum, better for fitting
    comp_elastic_combine: bool, optional
        combine elastic and compton as one component for fitting
    linear_bg: bool, optional
        use linear background instead of snip
    use_snip: bool, optional
        use snip method to remove background
    bin_energy: int, optional
        bin spectrum with given value
    dask_client: dask.distributed.Client
        Dask client object. If None, then Dask client is created automatically.
        If a batch of files is processed, then creating Dask client and
        passing the reference to it to the processing functions will save
        execution time: `client = Client(processes=True, silence_logs=logging.ERROR)`

    Returns
    -------
    result_map : dict
        of elemental map for given elements
    calculation_info : dict
        dict of fitting information
    """
    param = copy.deepcopy(parameter)
    if incident_energy is not None:
        param["coherent_sct_energy"]["value"] = incident_energy

    n_bin_low, n_bin_high = get_energy_bin_range(
        num_energy_bins=input_data.shape[2],
        low_e=param["non_fitting_values"]["energy_bound_low"]["value"],
        high_e=param["non_fitting_values"]["energy_bound_high"]["value"],
        e_offset=param["e_offset"]["value"],
        e_linear=param["e_linear"]["value"],
    )

    n_bin = np.arange(n_bin_low, n_bin_high)

    # calculate matrix for regression analysis
    elist = param["non_fitting_values"]["element_list"].split(", ")
    elist = [e.strip(" ") for e in elist]
    e_select, matv, e_area = construct_linear_model(n_bin, param, elist)

    # The initial list of elines may contain lines that are not activated for the incident beam
    #   energy. This always happens for at least one line when batches of XRF scans obtained for
    #   the range of beam energies are fitted for the same selection of emission lines to generate
    #   XANES amps. In such experiments, the line of interest is typically not activated at the
    #   lower energies of the band. It is impossible to include non-activated lines in the linear
    #   model. In order to make processing results consistent throughout the batch (contain the
    #   same set of emission lines), the non-activated lines are represented by maps filled with zeros.
    elist_non_activated = list(set(elist) - set(e_select))
    if elist_non_activated:
        logger.warning(
            "Some of the emission lines in the list are not activated: "
            f"{elist_non_activated} at {param['coherent_sct_energy']['value']} keV."
        )

    if comp_elastic_combine is True:
        e_select = e_select[:-1]
        e_select[-1] = "comp_elastic"

        matv_old = np.array(matv)
        matv = matv_old[:, :-1]
        matv[:, -1] += matv_old[:, -1]

    if linear_bg is True:
        e_select.append("const_bkg")

        matv_old = np.array(matv)
        matv = np.ones([matv_old.shape[0], matv_old.shape[1] + 1])
        matv[:, :-1] = matv_old

    logger.info("Matrix used for linear fitting has components: {}".format(e_select))

    def _log_unsupported_option(option):
        logger.warning(
            f"Option '{option}' is enabled. This option is not supported "
            f"and will be ignored. Disable the option to eliminate this warning."
        )

    if raise_bg > 0:
        _log_unsupported_option(f"raise_bg == {raise_bg}")

    # add const background, so nnls works better for values above zero
    # if raise_bg > 0:
    #     exp_data += raise_bg

    if pixel_bin > 1:
        _log_unsupported_option(f"pixel_bin == {pixel_bin}")

    # bin data based on nearest pixels, only two options
    # if pixel_bin in [4, 9]:
    #     logger.info('Bin pixel data with parameter: {}'.format(pixel_bin))
    #     exp_data = bin_data_spacial(exp_data, bin_size=int(np.sqrt(pixel_bin)))
    #     # exp_data = bin_data_pixel(exp_data, nearest_n=pixel_bin)  # return a copy of data

    if bin_energy > 1:
        _log_unsupported_option(f"pixel_bin == {pixel_bin}")
    # bin data based on energy spectrum
    # if bin_energy in [2, 3]:
    #     exp_data = conv_expdata_energy(exp_data, width=bin_energy)

    # make matrix smaller for single pixel fitting
    matv /= input_data.shape[0] * input_data.shape[1]
    # save matrix to analyze collinearity
    # np.save('mat.npy', matv)
    error_map = None

    if method != "nnls":
        logger.warning(f"Fitting using '{method}' is not supported: 'nnls' method will be used instead.")

    logger.info("Fitting method: non-negative least squares")

    snip_param = {
        "e_offset": param["e_offset"]["value"],
        "e_linear": param["e_linear"]["value"],
        "e_quadratic": param["e_quadratic"]["value"],
        "b_width": param["non_fitting_values"]["background_width"],
    }
    results = fit_xrf_map(
        data=input_data,
        data_sel_indices=(n_bin_low, n_bin_high),
        matv=matv,
        snip_param=snip_param,
        use_snip=use_snip,
        chunk_pixels=5000,
        n_chunks_min=4,
        progress_bar=TerminalProgressBar("NNLS fitting"),
        client=dask_client,
    )

    # output area of dict
    result_map = calculate_area(e_select, matv, results, param, first_peak_area=False)

    # Alternative fitting method (nonlinear fit). Very slow and nobody seems to be using it
    # logger.info('Fitting method: nonlinear least squares')
    # matrix_norm = exp_data.shape[0]*exp_data.shape[1]
    # fit_results = fit_pixel_multiprocess_nonlinear(exp_data, x, param, matv/matrix_norm,
    #                                               use_snip=use_snip)
    # result_map, error_map, results = get_area_and_error_nonlinear_fit(e_select,
    #                                                                  fit_results,
    #                                                                  matv/matrix_norm)

    # Generate 'zero' maps for the emission lines that were not activated
    for eline in elist_non_activated:
        result_map[eline] = np.zeros(shape=input_data.shape[0:2])

    calculation_info = dict()
    if error_map is not None:
        calculation_info["error_map"] = error_map

    calculation_info["fit_name"] = e_select
    calculation_info["regression_mat"] = matv
    calculation_info["results"] = results
    calculation_info["fit_range"] = (n_bin_low, n_bin_high)
    calculation_info["energy_axis"] = n_bin
    # Used to be 'exp_data'(selected data), now it is the full dataset,
    #   which can be ndarray, Dask array or RawHDF5Dataset. In order
    #   to get the selected set, 'input_data' must be sliced along axis2
    #   using 'fit_range' values.
    calculation_info["input_data"] = input_data
    calculation_info["data_sel_indices"] = (n_bin_low, n_bin_high)

    return result_map, calculation_info


def get_energy_bin_range(num_energy_bins, low_e, high_e, e_offset, e_linear):
    """
    Find the bin numbers in the range `0 .. num_energy_bins-1` that correspond
    to the selected energy range `low_e` .. `high_e`.

    The range `(n_low:n_high)` includes the bin with energies `low_e` .. `high_e`
    (note that `n_high` is not part of the range).

    Parameters
    ----------
    num_energy_bins : int
        number of energy bins
    low_e : float
        low energy bound, in keV
    high_e : float
        high energy bound, in keV
    e_offset : float
        offset term in energy calibration (energy for bin #0), keV
    e_linear : float
        linear term in energy calibration

    Returns
    -------
    n_low: int
        the number of the bin that corresponds to `low_e`
    n_high: int
        the number of the bin above the one that corresponds to `high_e`
    """

    v_low = (low_e - e_offset) / e_linear
    v_high = (high_e - e_offset) / e_linear + 1

    def _get_index(v):
        return int(np.clip(v, a_min=0, a_max=num_energy_bins))

    n_low, n_high = _get_index(v_low), _get_index(v_high)

    return n_low, n_high


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

    name, line = elemental_line.split("_")
    e = Element(name)
    transition_lines = TRANSITIONS_LOOKUP[line.upper()]

    sum_v = 0
    for v in transition_lines:
        sum_v += e.cs(energy)[v]
    ratio_v = e.cs(energy)[transition_lines[0]] / sum_v
    return ratio_v


def roi_sum_calculation(dir_path, file_prefix, fileID, element_dict, interpath):
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
    num_str = "{:03d}".format(fileID)
    # logger.info('File number is {}'.format(fileID))
    filename = file_prefix + num_str
    file_path = os.path.join(dir_path, filename)
    with h5py.File(file_path, "r") as f:
        data = f[interpath][:]

    result_map = dict()
    # for v in element_dict.keys():
    #     result_map[v] = np.zeros([datas[0], datas[1]])

    for k, v in element_dict.items():
        result_map[k] = np.sum(data[:, :, v[0] : v[1]], axis=2)

    return result_map


def roi_sum_multi_files(
    dir_path, file_prefix, start_i, end_i, element_dict, interpath="entry/instrument/detector/data"
):
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
    logger.info("cpu count: {}".format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    result_pool = [
        pool.apply_async(roi_sum_calculation, (dir_path, file_prefix, m, element_dict, interpath))
        for m in range(start_i, end_i + 1)
    ]

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
    if "pileup" in elemental_line:
        return "-"
    elif "_K" in elemental_line:
        name_label = "ka1"
        ename = elemental_line.split("_")[0]
    elif "_L" in elemental_line:
        name_label = "la1"
        ename = elemental_line.split("_")[0]
    elif "_M" in elemental_line:
        name_label = "ma1"
        ename = elemental_line.split("_")[0]
    else:
        return "-"

    e = Element(ename)
    sumv = 0
    for line_name in list(e.csb(eng).keys()):
        if name_label[0] in line_name:
            sumv += e.csb(eng)[line_name]
    if norm is True:
        return np.around(sumv / e.csb(eng)[name_label], round_n)
    else:
        return np.around(sumv, round_n)
