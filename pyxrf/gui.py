# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

__author__ = 'Li Li'
import enaml
from enaml.qt.qt_application import QtApplication

import os
import numpy as np
import logging

from pyxrf.model.fileio import FileIOModel
from pyxrf.model.lineplot import LinePlotModel #, SettingModel
from pyxrf.model.guessparam import GuessParamModel
from pyxrf.model.draw_image import DrawImageAdvanced
from pyxrf.model.fit_spectrum import Fit1D
from pyxrf.model.setting import SettingModel
import json

with enaml.imports():
    from pyxrf.view.main_window import XRFGui


def get_defaults():

    sub_folder = 'xrf_data'  # + '/xspress3'
    working_directory = os.path.join(os.path.expanduser('~'),
                                     'data_analysis', sub_folder)
    output_directory = working_directory

    # grab the default parameter file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    temp = current_dir.split('/')[:-1]
    temp.append('configs')
    param_dir = '/'.join(temp)
    default_parameter_file = os.path.join(
        param_dir, 'xrf_parameter.json')
    with open(default_parameter_file, 'r') as json_data:
        default_parameters = json.load(json_data)

    defaults = {'working_directory': working_directory,
                'output_directory': output_directory,
                'default_parameters': default_parameters}
    return defaults


def run():
    LOG_F = 'log_example.out'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO,
                        #filename=LOG_F,
                        filemode='w')

    app = QtApplication()
    defaults = get_defaults()

    io_model = FileIOModel(**defaults)
    param_model = GuessParamModel(**defaults)
    plot_model = LinePlotModel()
    fit_model = Fit1D(**defaults)
    setting_model = SettingModel()
    img_model_adv = DrawImageAdvanced()

    # send working directory changes to the fit_model
    #io_model.observe('working_directory', fit_model.result_folder_changed)
    io_model.observe('output_directory', fit_model.result_folder_changed)
    io_model.observe('output_directory', param_model.result_folder_changed)

    xrfview = XRFGui(io_model=io_model,
                     param_model=param_model,
                     plot_model=plot_model,
                     fit_model=fit_model,
                     setting_model=setting_model,
                     img_model_adv=img_model_adv)

    xrfview.show()
    app.start()


if __name__ == "__main__":
    run()
