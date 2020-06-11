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


class GlobalProcessingClasses:

    def __init__(self):
        self.defaults = None
        self.io_model = None
        self.param_model = None
        self.plot_model = None
        self.fit_model = None
        self.setting_model = None
        self.img_model_adv = None
        self.img_model_rgb = None

    def _get_defaults(self):

        # Set working directory to current working directory (if PyXRF is started from shell)
        working_directory = os.getcwd()
        logger.info(f"Starting PyXRF in the current working directory '{working_directory}'")

        default_parameters = param_data
        defaults = {'working_directory': working_directory,
                    'default_parameters': default_parameters}
        return defaults

    def initialize(self):
        """
        Run the sequence of actions needed to initialize PyXRF modules.

        """

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        defaults = self._get_defaults()
        self.io_model = FileIOModel(**defaults)
        self.param_model = GuessParamModel(**defaults)
        self.plot_model = LinePlotModel(param_model=self.param_model)
        self.fit_model = Fit1D(param_model=self.param_model, io_model=self.io_model, **defaults)
        self.setting_model = SettingModel(**defaults)
        self.img_model_adv = DrawImageAdvanced()
        self.img_model_rgb = DrawImageRGB()

        # Initialization needed to eliminate program crash
        self.plot_model.roi_dict = self.setting_model.roi_dict

        # send working directory changes to different models
        self.io_model.observe('working_directory', self.fit_model.result_folder_changed)
        self.io_model.observe('working_directory', self.setting_model.result_folder_changed)
        self.io_model.observe('selected_file_name', self.fit_model.data_title_update)
        self.io_model.observe('selected_file_name', self.plot_model.exp_label_update)
        self.io_model.observe('selected_file_name', self.setting_model.data_title_update)

        # send the same file to fit model, as fitting results need to be saved
        self.io_model.observe('file_name', self.fit_model.filename_update)
        self.io_model.observe('file_name', self.plot_model.plot_exp_data_update)
        self.io_model.observe('file_name', self.setting_model.filename_update)
        self.io_model.observe('runid', self.fit_model.runid_update)

        # send exp data to different models
        self.io_model.observe('data', self.plot_model.exp_data_update)
        self.io_model.observe('data', self.param_model.exp_data_update)
        self.io_model.observe('data', self.fit_model.exp_data_update)
        self.io_model.observe('data_all', self.fit_model.exp_data_all_update)
        self.io_model.observe('img_dict', self.fit_model.img_dict_update)
        self.io_model.observe('data_sets', self.fit_model.data_sets_update)
        self.io_model.observe('img_dict', self.setting_model.img_dict_update)

        # send fitting param of summed spectrum to param_model
        self.io_model.observe('param_fit', self.param_model.param_from_db_update)

        # send img dict to img_model for visualization
        self.io_model.observe('img_dict', self.img_model_adv.data_dict_update)
        self.io_model.observe('img_dict', self.img_model_rgb.data_dict_update)

        self.io_model.observe('incident_energy_set', self.plot_model.set_incident_energy)
        self.io_model.observe('incident_energy_set', self.img_model_adv.set_incident_energy)

        self.img_model_adv.observe('scaler_name_index', self.fit_model.scaler_index_update)

        self.img_model_adv.observe('dict_to_plot', self.fit_model.dict_to_plot_update)
        self.img_model_adv.observe('img_title', self.fit_model.img_title_update)
        self.img_model_adv.observe('quantitative_normalization', self.fit_model.quantitative_normalization_update)
        self.img_model_adv.observe('param_quant_analysis', self.fit_model.param_quant_analysis_update)

        self.param_model.observe('energy_bound_high_buf', self.fit_model.energy_bound_high_update)
        self.param_model.observe('energy_bound_low_buf', self.fit_model.energy_bound_low_update)
        self.param_model.observe('energy_bound_high_buf', self.plot_model.energy_bound_high_update)
        self.param_model.observe('energy_bound_low_buf', self.plot_model.energy_bound_low_update)

        logger.info('pyxrf started.')

    def open_data_file(self, file_path):

        self.io_model.data_ready = False
        # only load one file
        # 'temp' is used to reload the same file, otherwise file_name will not update
        self.io_model.file_name = 'temp'
        f_dir, f_name = os.path.split(file_path)
        self.io_model.working_directory = f_dir

        # The following statement initiates file loading. It may raise exceptions
        try:
            self.io_model.file_name = f_name
        except Exception:
            # Clear file name or scan id from window title. This does not update
            #   the displayed title.
            self.io_model.window_title_clear()
            raise
        else:
            self.plot_model.parameters = self.param_model.param_new
            self.setting_model.parameters = self.param_model.param_new
            self.setting_model.data_sets = self.io_model.data_sets
            self.fit_model.data_sets = self.io_model.data_sets
            self.fit_model.fit_img = {}  # clear dict in fitmodel to rm old results

            # Change window title (include file name). This does not update the visible title,
            #   only the text attribute of 'io_model' class.
            self.io_model.window_title_set_file_name(f_name)

            if not self.io_model.incident_energy_available:
                msg = ("Incident energy is not available in scan metadata and must be set manually. "
                       "Incident energy may be set by changing 'Incident energy, keV' parameter "
                       "in the dialog boxes opened using 'Find Automatically...' ('Find Elements "
                       "in sample' or 'General...' ('General Settings for Fitting Alogirthm') "
                       "buttons in 'Model' tab.")
            else:
                msg = ""

            return msg
