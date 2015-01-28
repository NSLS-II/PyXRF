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

from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum

import logging
logger = logging.getLogger(__name__)

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
    data_dict : dict
        Dict has filename as key and group data as value.
    """
    working_directory = Str()
    data_file = Str()
    file_names = List()
    file_path = Str()
    data = Typed(np.ndarray)
    load_status = Str()
    data_dict = Dict()
    img_dict = Dict()
    data_sets = OrderedDict() #Typed(OrderedDict)

    def __init__(self,
                 working_directory=None,
                 data_file=None, *args, **kwargs):
        if working_directory is None:
            working_directory = os.path.expanduser('~')

        with self.suppress_notifications():
            self.working_directory = working_directory
            self.data_file = data_file
        #self.data_sets = OrderedDict()
        # load the data file
        #self.load_data()

    @observe('file_names')
    def update_more_data(self, change):
        self.data_sets.clear()
        self.file_names.sort()
        print('file name: {}'.format(self.file_names))
        for fname in self.file_names:
            try:
                self.file_path = os.path.join(self.working_directory, fname)
                f = h5py.File(self.file_path, 'r')
                data = f['MAPS']
                # dict has filename as key and group data as value
                self.data_dict.update({fname: data})
                print('filenames are updated')
                DS = DataSelection(filename=fname,
                                   raw_data=np.asarray(data['mca_arr']))
                self.data_sets.update({fname: DS})
            except ValueError:
                continue
        print('ordered keys: {}'.format(self.data_sets.keys()))
        #self.get_roi_data()

    def get_roi_data(self):
        """
        Get roi sum data from data_dict.
        """
        for k, v in six.iteritems(self.data_dict):
            roi_dict = {d[0]: d[1] for d in zip(v['channel_names'], v['XRF_roi'])}
            self.img_dict.update({str(k): {'roi_sum': roi_dict}})

    #def set_file_for_plot(self):


plot_as = ['Summed', 'Point', 'Roi']


class DataSelection(Atom):

    filename = Str()
    plot_choice = Enum(*plot_as)
    point1 = Str('0, 0')
    point2 = Str('0, 0')
    roi = List()
    raw_data = Typed(np.ndarray)
    data = Typed(np.ndarray)
    plot_index = Int(0)

    @observe('plot_index', 'point1', 'point2')
    def _update_roi(self, change):
        if self.plot_index == 0:
            return
        elif self.plot_index == 1:
            self.data = self.get_sum()
            print('spec is calculated at step {} as {}'.format(change['value'],
                                                               np.sum(self.data)))
        elif self.plot_index == 2:
            SC = SpectrumCalculator(self.raw_data, pos1=self.point1)
            self.data = SC.get_spectrum()
            print('spec is calculated at step {} as {}'.format(change['value'],
                                                               np.sum(self.data)))
        else:
            SC = SpectrumCalculator(self.raw_data,
                                    pos1=self.point1,
                                    pos2=self.point2)
            self.data = SC.get_spectrum()
            print('spec is calculated at step {} as {}'.format(change['value'],
                                                               np.sum(self.data)))

    def get_sum(self):
        SC = SpectrumCalculator(self.raw_data)
        return SC.get_spectrum()


class SpectrumCalculator(object):

    def __init__(self, data,
                 pos1=None, pos2=None):
        self.data = data
        if pos1:
            self.pos1 = self._parse_pos(pos1)
        else:
            self.pos1 = None
        if pos2:
            self.pos2 = self._parse_pos(pos2)
        else:
            self.pos2 = None

    def _parse_pos(self, pos):
        if isinstance(pos, list):
            return pos
        return [float(v) for v in pos.split(', ')]

    def get_spectrum(self):
        if not self.pos1 and not self.pos2:
            return np.sum(self.data, axis=(1, 2))
        elif self.pos1 and not self.pos2:
            #if self.pos1[0] >= self.data.shape[1]:
            #    return np.sum(self.data, axis=(1, 2))
            return self.data[:, self.pos1[0], self.pos1[1]]
        else:
            return np.sum(self.data[:, self.pos1[0]:self.pos2[0], self.pos1[1]:self.pos2[1]],
                          axis=(1, 2))
