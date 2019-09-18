from __future__ import absolute_import
import enaml
from enaml.qt.qt_application import QtApplication

import os
# import numpy as np
from atom.api import Atom, Str
from .model.fileio import FileIOModel
from .model.lineplot import LinePlotModel  # , SettingModel
from .model.guessparam import GuessParamModel
from .model.draw_image import DrawImageAdvanced
from .model.draw_image_rgb import DrawImageRGB
from .model.fit_spectrum import Fit1D
from .model.setting import SettingModel
from .model.param_data import param_data
# import json

import logging
logger = logging.getLogger()

with enaml.imports():
    from .view.main_window import XRFGui


def get_defaults():

    sub_folder = 'data_analysis'
    working_directory = os.path.join(os.path.expanduser('~'), sub_folder)
    # Ideally, if directory does not exist, it should be created, but for now
    #     we just set working directory to user directory
    if not os.path.exists(working_directory):
        working_directory = os.path.expanduser('~')

    # output_directory = working_directory
    default_parameters = param_data
    defaults = {'working_directory': working_directory,
                # 'output_directory': output_directory,
                'default_parameters': default_parameters}
    return defaults


class LogModel(Atom):
    logtext = Str()


class GuiHandler(logging.Handler):
    def __init__(self, model=None):
        super(GuiHandler, self).__init__()
        if model is None:
            model = LogModel()
        self.model = model

    def handle(self, record):
        self.model.logtext += self.format(record) + '\n'


def run():

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    app = QtApplication()
    defaults = get_defaults()
    io_model = FileIOModel(**defaults)
    param_model = GuessParamModel(**defaults)
    plot_model = LinePlotModel()
    fit_model = Fit1D(**defaults)
    setting_model = SettingModel(**defaults)
    img_model_adv = DrawImageAdvanced()
    img_model_rgb = DrawImageRGB()

    # Output log to gui, turn off for now
    # error at mac, works fine on linux
    # so log info only outputs to terminal for now.
    guihandler = GuiHandler()
    guihandler.setLevel(logging.INFO)
    guihandler.setFormatter(formatter)
    logger.addHandler(guihandler)

    # send working directory changes to different models
    io_model.observe('working_directory', fit_model.result_folder_changed)
    io_model.observe('selected_file_name', fit_model.data_title_update)
    io_model.observe('selected_file_name', plot_model.exp_label_update)

    # send the same file to fit model, as fitting results need to be saved
    io_model.observe('file_name', fit_model.filename_update)
    io_model.observe('file_name', plot_model.plot_exp_data_update)
    io_model.observe('runid', fit_model.runid_update)

    # send exp data to different models
    io_model.observe('data', plot_model.exp_data_update)
    io_model.observe('data', param_model.exp_data_update)
    io_model.observe('data', fit_model.exp_data_update)
    io_model.observe('data_all', fit_model.exp_data_all_update)
    io_model.observe('img_dict', fit_model.img_dict_update)

    # send fitting param of summed spectrum to param_model
    io_model.observe('param_fit', param_model.param_from_db_update)

    # send img dict to img_model for visualization
    io_model.observe('img_dict', img_model_adv.data_dict_update)
    io_model.observe('img_dict', img_model_rgb.data_dict_update)

    img_model_adv.observe('scaler_name_index', fit_model.scaler_index_update)

    param_model.observe('energy_bound_high_buf', fit_model.energy_bound_high_update)
    param_model.observe('energy_bound_low_buf', fit_model.energy_bound_low_update)

    #  set default parameters
    # io_model.observe('default_parameters', plot_model.parameters_update)
    # param_model.observe('param_new', plot_model.parameters_update)
    # fit_model.observe('param_dict', param_model.param_changed)

    #  send exp data to SettingModel for roi sum
    #  got warning message
    # io_model.observe('data_sets', setting_model.data_sets_update)
    logger.info('pyxrf started.')
    xrfview = XRFGui(io_model=io_model,
                     param_model=param_model,
                     plot_model=plot_model,
                     fit_model=fit_model,
                     setting_model=setting_model,
                     img_model_adv=img_model_adv,
                     img_model_rgb=img_model_rgb,
                     logmodel=guihandler.model)

    xrfview.show()
    app.start()


if __name__ == "__main__":
    run()
