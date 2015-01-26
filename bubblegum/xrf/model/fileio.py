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
import logging
logger = logging.getLogger(__name__)

from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum


class FileIOModel(Atom):
    """
    This class focuses on file input and output.

    Attributes
    ----------
    working_directory : str
    file_names : list
        list of loaded files
    data_file : str
    file_path : str
    data : array
        Experiment data.
    load_status : str
        Description of file loading status
    file_opt : int
        Define which file to choose
    data_obj : dict
        data object
    data_dict : dict
        Dict has filename as key and group data as value.
    """
    working_directory = Str()
    data_file = Str()
    file_names = List()
    file_path = Str()
    data = Typed(np.ndarray)
    load_status = Str()
    file_opt = Int()
    data_obj = Typed(object)
    data_dict = Dict()
    img_dict = Dict()

    def __init__(self,
                 working_directory=None,
                 data_file=None, *args, **kwargs):
        if working_directory is None:
            working_directory = os.path.expanduser('~')

        with self.suppress_notifications():
            self.working_directory = working_directory
            self.data_file = data_file
        # load the data file
        #self.load_data()

    # @observe('working_directory', 'data_file')
    # def path_changed(self, changed):
    #     if changed['type'] == 'create':
    #         return
    #     self.load_data()
#>>>>>>> eric_autofit

    @observe('data')
    def data_changed(self, changed):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

    @observe('file_names')
    def update_more_data(self, change):
        for fname in self.file_names:
            try:
                self.file_path = os.path.join(self.working_directory, fname)
                f = h5py.File(self.file_path, 'r')
                data = f['MAPS']
                # dict has filename as key and group data as value
                self.data_dict.update({fname: data})
            except ValueError:
                continue
            #f.close()
        self.get_roi_data()

    @observe('file_opt')
    def choose_file(self, changed):
        print('option is {}'.format(self.file_opt))
        if self.file_opt == 0:
            return
        self.data_file = self.file_names[self.file_opt-1]
        self.data_obj = self.data_dict[str(self.data_file)]
        # calculate the summed intensity, this should be included in data already
        self.data = np.sum(self.data_obj['mca_arr'], axis=(1, 2))

    def get_roi_data(self):
        """
        Get roi sum data from data_dict.
        """
        for k, v in six.iteritems(self.data_dict):
            roi_dict = {d[0]: d[1] for d in zip(v['channel_names'], v['XRF_roi'])}
            self.img_dict.update({str(k): {'roi_sum': roi_dict}})
        print('keys: {}'.format(self.img_dict.keys()))


plot_as = ['Summed', 'Point', 'Roi']


class DataSelection(Atom):

    filename = Str()
    plot_choice = Enum(*plot_as)
    point = List()
    roi = List()