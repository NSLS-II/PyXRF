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

import os
import numpy as np
import enaml
from enaml.qt.qt_application import QtApplication
from bubblegum.xrf.model.fileio import FileIOModel
from bubblegum.xrf.model.lineplot import LinePlotModel
from bubblegum.xrf.model.guessparam import GuessParamModel
import json
from pprint import pprint

def get_defaults():

    working_directory = os.path.join(os.path.expanduser('~'), 'Downloads')
    data_file = 'NSLS_X27.txt'
    data_path = os.path.join(working_directory, data_file)
    # grab the default parameter file
    default_parameter_file = os.path.join(os.path.expanduser('~'), '.bubblegum',
                                      'xrf_parameter_default.json')
    with open(default_parameter_file, 'r') as json_data:
        default_parameters = json.load(json_data)

    pprint(default_parameters)
    # see if there is a user parameter file
    user_parameter_file = os.path.join(os.path.expanduser('~'), '.bubblegum',
                                      'xrf_parameter_user.json')
    try:
        with open(user_parameter_file, 'r') as json_data:
            user = json.load(json_data)

        default_parameters.update(user)
    except IOError:
        # user file doesn't exist
        pass

    pprint(default_parameters)



    defaults = {'working_directory': working_directory,
                'data_file': data_file,
                'data_path': data_path,
                'default_parameters': default_parameters,
    }

    return defaults


def run():
    app = QtApplication()
    with enaml.imports():
        from bubblegum.xrf.view.main_window import XRFGui

    defaults = get_defaults()

    xrfview = XRFGui()

    xrfview.io_model = FileIOModel(**defaults)
    xrfview.param_model = GuessParamModel(**defaults)
    xrfview.plot_model = LinePlotModel()

    xrfview.show()
    app.start()


if __name__ == "__main__":
    run()
