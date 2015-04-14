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

from __future__ import (absolute_import, division,
                        print_function)

__author__ = 'Li Li'

import six
import h5py
import numpy as np
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
    data : array
        Experiment data.
    load_status : str
        Description of file loading status
    data_dict : dict
        Dict has filename as key and group data as value.
    """
    working_directory = Str()
    file_names = List()
    file_path = Str()
    data = Typed(np.ndarray)
    load_status = Str()
    data_dict = Dict()
    img_dict = Dict()
    img_dict_flat = Dict()

    file_channel_list = List()

    data_sets = Typed(OrderedDict)
    data_sets_fit = Typed(OrderedDict)

    def __init__(self, **kwargs):
        self.working_directory = kwargs['working_directory']
        #with self.suppress_notifications():
        #    self.working_directory = working_directory
            #self.data_file = data_file

    @observe('file_names')
    def update_more_data(self, change):
        self.file_channel_list = []
        self.file_names.sort()
        logger.info('Loaded files : {}'.format(self.file_names))

        self.data_dict, self.data_sets_fit, self.data_sets = read_hdf_HXN(self.working_directory,
                                                                          self.file_names)

        self.file_channel_list = self.data_sets.keys()

    def get_roi_data(self):
        """
        Get roi sum data from data_dict.
        """
        # for k, v in six.iteritems(self.data_dict):
        #     roi_dict = {d[0]: d[1] for d in zip(v['channel_names'], v['XRF_roi'])}
        #     self.img_dict.update({str(k): {'roi_sum': roi_dict}})
        #
        #     self.img_dict_flat.update({str(k).split('.')[0]+'_roi_sum': roi_dict})
        pass


def get_roi_sum(namelist, data_range, data):
    data_temp = dict()
    for i in range(len(namelist)):
        lowv = data_range[i, 0]
        highv = data_range[i, 1]
        data_sum = np.sum(data[:, :, lowv: highv], axis=2)
        data_temp.update({namelist[i].replace(' ', '_'): data_sum})
    return data_temp


def read_hdf_HXN(working_directory,
                 file_names, channel_num=8):
    """
    Data IO for HXN temporary datasets. This might be changed later.

    Parameters
    ----------
    working_directory : str
        path folder
    file_names : list
        list of chosen files
    channel_num : int, optional
        detector channel number

    Returns
    -------
    data_dict : dict
        with fitting data
    data_sets_fit : dict
        summed data of all channels, file dependent, for fitting purpose
    data_sets : dict
        data from all channels, for plotting purpose
    """
    data_dict = OrderedDict()
    data_sets_fit = OrderedDict()
    data_sets = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 1

    for fname in file_names:
        try:
            file_path = os.path.join(working_directory, fname)
            f = h5py.File(file_path, 'r+')
            #data = f['MAPS']
            #data = f['xrfmap']
            data = f['entry/instrument']
            exp_data = np.asarray(data['detector/data'])
            logger.info('File : {} with total counts {}'.format(fname, np.sum(exp_data)))

            #exp_data = np.reshape(exp_data, [2, 4, 4096])
            # dict has filename as key and group data as value

            data_dict[fname] = data

            fname = fname.split('.')[0]

            #
            exp_data = exp_data[:, :, :-bad_point_cut]

            # for fitting
            DS = DataSelection(filename=fname,
                               raw_data=exp_data)

            data_sets_fit.update({fname: DS})

            #plot each channel, for plotting purposes
            data_sets.update({fname: DS})
            for i in range(channel_num):
                file_channel = fname+'_channel_'+str(i+1)
                exp_data_new = np.reshape(exp_data[0, i, :],
                                          [1, 1, exp_data[0, i, :].size])
                DS = DataSelection(filename=file_channel,
                                   raw_data=exp_data_new)
                data_sets.update({file_channel: DS})

            # get roi sum data
            #roi_result = get_roi_sum(data[detID]['roi_name'].value,
            #                         data[detID]['roi_limits'].value,
            #                         data[detID]['counts'])
            #self.img_dict_flat.update({fname.split('.')[0]+'_roi': roi_result})

            # read fitting results
            # if 'xrf_fit' in data[detID]:
            #     fit_result = get_fit_data(data[detID]['xrf_fit_name'].value,
            #                               data[detID]['xrf_fit'].value)
            #     self.img_dict_flat.update({fname.split('.')[0]+'_fit': fit_result})

        except ValueError:
            continue
    return data_dict, data_sets_fit, data_sets


def get_fit_data(namelist, data):
    """
    Read fit data from h5 file. This is to be moved to filestore part.

    Parameters
    ---------
    namelist : list
        list of str for element lines
    data : array
        3D array of fitting results
    """
    data_temp = dict()
    for i in range(len(namelist)):
        data_temp.update({namelist[i]: data[i, :, :]})
    return data_temp
    #self.img_dict_flat.update({fname.split('.')[0]: data_temp})


plot_as = ['Sum', 'Point', 'Roi']


class DataSelection(Atom):
    """
    Attributes
    ----------
    filename : str
    plot_choice : enum
        methods ot plot
    point1 : str
        starting position
    point2 : str
        ending position
    roi : list
    raw_data : array
        experiment 3D data
    data : array
    plot_index : int
        plot data or not, sum or roi or point
    """
    filename = Str()
    plot_choice = Enum(*plot_as)
    point1 = Str('0, 0')
    point2 = Str('0, 0')
    raw_data = Typed(np.ndarray)
    data = Typed(np.ndarray)
    plot_index = Int(0)
    fit_name = Str()
    fit_data = Typed(np.ndarray)

    @observe('plot_index', 'point1', 'point2')
    def _update_roi(self, change):
        if self.plot_index == 0:
            return
        elif self.plot_index == 1:
            self.data = self.get_sum()
        elif self.plot_index == 2:
            SC = SpectrumCalculator(self.raw_data, pos1=self.point1)
            self.data = SC.get_spectrum()
        else:
            SC = SpectrumCalculator(self.raw_data,
                                    pos1=self.point1,
                                    pos2=self.point2)
            self.data = SC.get_spectrum()

    def get_sum(self):
        SC = SpectrumCalculator(self.raw_data)
        return SC.get_spectrum()


class SpectrumCalculator(object):
    """
    Calculate summed spectrum according to starting and ending positions.

    Attributes
    ----------
    data : array
        3D array of experiment data
    pos1 : str
        starting position
    pos2 : str
        ending position
    """

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
        return [int(v) for v in pos.split(',')]

    def get_spectrum(self):
        if not self.pos1 and not self.pos2:
            return np.sum(self.data, axis=(0, 1))
        elif self.pos1 and not self.pos2:
            print('shape: {}'.format(self.data.shape))
            print('pos1: {}'.format(self.pos1))
            return self.data[self.pos1[0], self.pos1[1], :]
            #return self.data[:, self.pos1[0], self.pos1[1]]
        else:
            return np.sum(self.data[self.pos1[0]:self.pos2[0], self.pos1[1]:self.pos2[1], :],
                          axis=(0, 1))
            #return np.sum(self.data[:, self.pos1[0]:self.pos2[0], self.pos1[1]:self.pos2[1]],
            #              axis=(1, 2))


