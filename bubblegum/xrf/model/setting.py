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

import six
import h5py
import numpy as np
import copy
import os
from collections import OrderedDict

from atom.api import (Atom, Str, observe, Typed,
                      Dict, List, Int, Enum, Float, Bool)

from skxray.constants.api import XrfElement as Element

import logging
logger = logging.getLogger(__name__)


class SettingModel(Atom):
    """
    Advanced setting for roi sum.

    Attributes
    ----------
    element_input : str
        element names
    element_list : list
        list of strings
    roi_dict : dict
        dict of class object
    """
    element_input = Str()
    element_list = List()
    #element_list = Typed(object)
    roi_dict = OrderedDict()
    parameters = Dict()

    data_sets = Dict()



    def __init__(self):
        self.element_list = []

    @observe('element_input')
    def _update_element(self, change):
        #if change['type'] == 'create':
        #    return
        if ',' in self.element_input:
            element_list = [v.strip(' ') for v in self.element_input.split(',')]
        else:
            element_list = [v for v in self.element_input.split(' ')]
        self.update_roi(element_list)
        self.element_list = element_list
        #if len(self.element_list):
        #    self.update_roi()

    def update_roi(self, element_list):
        if not len(element_list):
            return
        self.roi_dict.clear()
        for v in element_list:
            if '_K' in v:
                temp = v.split('_')[0]
                e = Element(temp)
                val = int(e.emission_line['ka1']*1000)
            elif '_L' in v:
                temp = v.split('_')[0]
                e = Element(temp)
                val = int(e.emission_line['la1']*1000)
            elif '_M' in v:
                temp = v.split('_')[0]
                e = Element(temp)
                val = int(e.emission_line['ma1']*1000)

            delta_v = int(self.get_sigma(val/1000.)*1000)
            roi = ROIModel(line_val=val, left_val=val-delta_v*2,
                           right_val=val+delta_v*2,
                           step=1,
                           show_plot=False)
            self.roi_dict.update({v: roi})

    def get_sigma(self, energy, epsilon=2.96):
        print('param: {}'.format(self.parameters.keys()))
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt((self.parameters['fwhm_offset'].value/temp_val)**2 +
                       energy*epsilon*self.parameters['fwhm_fanoprime'].value)






class ROIModel(Atom):
    """
    This class defines basic data structure for roi setup.

    Attributes
    ----------
    line_val : float
        emission energy of primary line
    left_val : float
        left boundary
    right_val : float
        right boundary
    step : float
        min step value to change
    show_plot : bool
        option to plot
    """
    line_val = Int()
    left_val = Int()
    right_val = Int()
    step = Int(1)
    show_plot = Bool(False)

    @observe('left_val')
    def _value_update(self, change):
        print('left value is changed {}'.format(change))
