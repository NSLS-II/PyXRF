from __future__ import absolute_import

import os
from ..model.fileio import FileIOModel
from ..model.lineplot import LinePlotModel  # , SettingModel
from ..model.guessparam import GuessParamModel
from ..model.draw_image import DrawImageAdvanced
from ..model.draw_image_rgb import DrawImageRGB
from ..model.fit_spectrum import Fit1D
from ..model.setting import SettingModel
from ..model.param_data import param_data

import logging
logger = logging.getLogger()


def _get_defaults():

    # Set working directory to current working directory (if PyXRF is started from shell)
    working_directory = os.getcwd()
    logger.info(f"Starting PyXRF in the current working directory '{working_directory}'")

    default_parameters = param_data
    defaults = {'working_directory': working_directory,
                'default_parameters': default_parameters}
    return defaults


def pyxrf_startup():
    """
    Run the sequence of actions needed to initialize PyXRF modules.
    """

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    defaults = _get_defaults()
    io_model = FileIOModel(**defaults)
    param_model = GuessParamModel(**defaults)
    plot_model = LinePlotModel(param_model=param_model)
    fit_model = Fit1D(param_model=param_model, io_model=io_model, **defaults)
    setting_model = SettingModel(**defaults)
    img_model_adv = DrawImageAdvanced()
    img_model_rgb = DrawImageRGB()

    # Initialization needed to eliminate program crash
    plot_model.roi_dict = setting_model.roi_dict

    # send working directory changes to different models
    io_model.observe('working_directory', fit_model.result_folder_changed)
    io_model.observe('working_directory', setting_model.result_folder_changed)
    io_model.observe('selected_file_name', fit_model.data_title_update)
    io_model.observe('selected_file_name', plot_model.exp_label_update)
    io_model.observe('selected_file_name', setting_model.data_title_update)

    # send the same file to fit model, as fitting results need to be saved
    io_model.observe('file_name', fit_model.filename_update)
    io_model.observe('file_name', plot_model.plot_exp_data_update)
    io_model.observe('file_name', setting_model.filename_update)
    io_model.observe('runid', fit_model.runid_update)

    # send exp data to different models
    io_model.observe('data', plot_model.exp_data_update)
    io_model.observe('data', param_model.exp_data_update)
    io_model.observe('data', fit_model.exp_data_update)
    io_model.observe('data_all', fit_model.exp_data_all_update)
    io_model.observe('img_dict', fit_model.img_dict_update)
    io_model.observe('data_sets', fit_model.data_sets_update)
    io_model.observe('img_dict', setting_model.img_dict_update)

    # send fitting param of summed spectrum to param_model
    io_model.observe('param_fit', param_model.param_from_db_update)

    # send img dict to img_model for visualization
    io_model.observe('img_dict', img_model_adv.data_dict_update)
    io_model.observe('img_dict', img_model_rgb.data_dict_update)

    io_model.observe('incident_energy_set', plot_model.set_incident_energy)
    io_model.observe('incident_energy_set', img_model_adv.set_incident_energy)

    img_model_adv.observe('scaler_name_index', fit_model.scaler_index_update)

    img_model_adv.observe('dict_to_plot', fit_model.dict_to_plot_update)
    img_model_adv.observe('img_title', fit_model.img_title_update)
    img_model_adv.observe('quantitative_normalization', fit_model.quantitative_normalization_update)
    img_model_adv.observe('param_quant_analysis', fit_model.param_quant_analysis_update)

    param_model.observe('energy_bound_high_buf', fit_model.energy_bound_high_update)
    param_model.observe('energy_bound_low_buf', fit_model.energy_bound_low_update)
    param_model.observe('energy_bound_high_buf', plot_model.energy_bound_high_update)
    param_model.observe('energy_bound_low_buf', plot_model.energy_bound_low_update)

    logger.info('pyxrf started.')
