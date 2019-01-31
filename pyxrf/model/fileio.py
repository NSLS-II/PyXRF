# ######################################################################
# Copyright (c) 2014-, Brookhaven Science Associates, Brookhaven       #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                       #
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
                        print_function, unicode_literals)

__author__ = 'Li Li'

import six
import sys
import h5py
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import json
import time
import skimage.io as sio
from PIL import Image
import copy
import glob
import ast
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import multiprocessing
from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum, Float, Bool

import logging
logger = logging.getLogger()

import warnings
warnings.filterwarnings('ignore')

sep_v = os.sep

try:
    config_path = '/etc/pyxrf/pyxrf.json'
    with open(config_path, 'r') as beamline_pyxrf:
        beamline_config_pyxrf = json.load(beamline_pyxrf)
        beamline_name = beamline_config_pyxrf['beamline_name']
    if beamline_name == 'HXN':
        from pyxrf.db_config.hxn_db_config import db
    elif beamline_name == 'SRX':
        from pyxrf.db_config.srx_db_config import db
    else:
        db = None
        print('Beamline Database is not used in pyxrf.')
except IOError:
    db = None
    print('Beamline Database is not used in pyxrf.')

try:
    import suitcase.hdf5 as sc
except ImportError:
    pass


class FileIOModel(Atom):
    """
    This class focuses on file input and output.

    Attributes
    ----------
    working_directory : str
        current working path
    file_name : str
        name of loaded file
    load_status : str
        Description of file loading status
    data_sets : dict
        dict of experiment data, 3D array
    img_dict : dict
        Dict of 2D arrays, such as 2D roi pv or fitted data
    """
    working_directory = Str()
    file_name = Str()
    file_path = Str()
    load_status = Str()
    data_sets = Typed(OrderedDict)
    img_dict = Dict()

    file_channel_list = List()

    runid = Int(-1)
    h_num = Int(1)
    v_num = Int(1)
    fname_from_db = Str()

    file_opt = Int()
    data = Typed(np.ndarray)
    data_all = Typed(np.ndarray)
    selected_file_name = Str()
    #file_name = Str()
    mask_data = Typed(object)
    mask_name = Str()
    mask_opt = Int(0)
    load_each_channel = Bool(False)

    p1_row = Int(-1)
    p1_col = Int(-1)
    p2_row = Int(-1)
    p2_col = Int(-1)

    def __init__(self, **kwargs):
        self.working_directory = kwargs['working_directory']
        self.mask_data = None

    @observe(str('file_name'))
    def update_more_data(self, change):
        if change['value'] == 'temp':
            # 'temp' is used to reload the same file
            return

        self.file_channel_list = []
        logger.info('File is loaded: %s' % (self.file_name))

        # focus on single file only
        self.img_dict, self.data_sets = file_handler(self.working_directory,
                                                     self.file_name,
                                                     load_each_channel=self.load_each_channel)
        self.file_channel_list = list(self.data_sets.keys())
        self.file_opt = 1  # use summed data as default

    @observe(str('runid'))
    def _update_fname(self, change):
        self.fname_from_db = 'scan_'+str(self.runid)+'.h5'

    def load_data_runid(self):
        """
        Load data according to runID number.

        requires databroker
        """
        if db is None:
            raise RuntimeError("databroker is not installed. This function "
                               "is disabled.  To install databroker, see "
                               "https://nsls-ii.github.io/install.html")
        if self.h_num != 0 and self.v_num != 0:
            datashape = [self.v_num, self.h_num]

        self.file_name = self.fname_from_db
        fpath = os.path.join(self.working_directory, self.file_name)
        config_file = os.path.join(self.working_directory, 'pv_config.json')
        db_to_hdf_config(fpath, self.runid,
                         datashape, config_file)

    @observe(str('file_opt'))
    def choose_file(self, change):
        if self.file_opt == 0:
            return

        # selected file name from all channels
        # controlled at top level gui.py startup
        try:
            self.selected_file_name = self.file_channel_list[self.file_opt-1]
        except IndexError:
            pass

        # passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[self.selected_file_name].raw_data
        # get summed data or based on mask
        self.data = self.data_sets[self.selected_file_name].get_sum()

    def apply_mask(self):
        """Apply mask with different options.
        """
        if self.mask_opt == 2:
            # load mask data
            if len(self.mask_name) > 0:
                mask_file = os.path.join(self.working_directory,
                                         self.mask_name)
                try:
                    if 'npy' in mask_file:
                        self.mask_data = np.load(mask_file)
                    elif 'txt' in mask_file:
                        self.mask_data = np.loadtxt(mask_file)
                    else:
                        self.mask_data = np.array(Image.open(mask_file))
                except IOError:
                    logger.error('Mask file cannot be loaded.')

                for k in six.iterkeys(self.img_dict):
                    if 'fit' in k:
                        self.img_dict[k][self.mask_name] = self.mask_data
        else:
            self.mask_data = None
            data_s = self.data_all.shape
            if self.mask_opt == 1:
                valid_opt = False
                # define square mask region
                if self.p1_row>=0 and self.p1_col>=0 and self.p1_row<data_s[0] and self.p1_col<data_s[1]:
                    self.data_sets[self.selected_file_name].point1 = [self.p1_row, self.p1_col]
                    logger.info('Starting position is {}.'.format([self.p1_row, self.p1_col]))
                    valid_opt = True
                    if self.p2_row>self.p1_row and self.p2_col>self.p1_col and self.p2_row<data_s[0] and self.p2_col<data_s[1]:
                        self.data_sets[self.selected_file_name].point2 = [self.p2_row, self.p2_col]
                        logger.info('Ending position is {}.'.format([self.p2_row, self.p2_col]))
                if valid_opt is False:
                    logger.info('The positions are not valid. No mask is applied.')
            else:
                self.data_sets[self.selected_file_name].delete_points()
                logger.info('Do not apply mask.')

        # passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[self.selected_file_name].raw_data
        # get summed data or based on mask
        self.data = self.data_sets[self.selected_file_name].get_sum(self.mask_data)


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
    #point1 = Str('0, 0')
    #point2 = Str('0, 0')
    point1 = List()
    point2 = List()
    raw_data = Typed(np.ndarray)
    data = Typed(np.ndarray)
    plot_index = Int(0)
    fit_name = Str()
    fit_data = Typed(np.ndarray)

    @observe(str('plot_index'))
    def _update_roi(self, change):
        if self.plot_index == 0:
            return
        elif self.plot_index == 1:
            self.data = self.get_sum()

    def delete_points(self):
        self.point1 = []
        self.point2 = []

    def get_sum(self, mask=None):
        if len(self.point1)==0 and len(self.point2)==0:
            SC = SpectrumCalculator(self.raw_data)
            return SC.get_spectrum(mask=mask)
        else:
            SC = SpectrumCalculator(self.raw_data,
                                    pos1=self.point1,
                                    pos2=self.point2)
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
        self.pos1 = pos1
        self.pos2 = pos2

    def get_spectrum(self, mask=None):
        """
        Get roi sum from point positions, or from mask file.
        """
        if mask is None:
            if not self.pos1 and not self.pos2:
                return np.sum(self.data, axis=(0, 1))
            elif self.pos1 and not self.pos2:
                return self.data[self.pos1[0], self.pos1[1], :]
            else:
                return np.sum(self.data[self.pos1[0]:self.pos2[0],
                                        self.pos1[1]:self.pos2[1], :],
                              axis=(0, 1))
        else:
            spectrum_sum = np.zeros(self.data.shape[2])
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    if mask[i,j] > 0:
                        spectrum_sum += self.data[i, j, :]
            return spectrum_sum


def file_handler(working_directory, file_name, load_each_channel=True, spectrum_cut=3000):
    # send information on GUI level later !
    get_data_nsls2 = True
    try:
        if get_data_nsls2 is True:
            return read_hdf_APS(working_directory, file_name,
                                spectrum_cut=spectrum_cut,
                                load_each_channel=load_each_channel)
        else:
            return read_MAPS(working_directory,
                             file_name, channel_num=1)
    except IOError as e:
        logger.error("I/O error({0}): {1}".format(e.errno, e.strerror))
        logger.error('Please select .h5 file')
    except:
        logger.error("Unexpected error:", sys.exc_info()[0])
        raise


def fetch_data_from_db(runid):
    """
    Read data from database.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        ID for given experimental measurement

    Returns
    -------
    data : pandas.core.frame.DataFrame
        data frame with keys as given PV names.
    """

    #hdr = db[runid]
    # headers = db.find_headers(scan_id=runid)
    # head_list = sorted(headers, key=lambda x: x.start_time)
    # hdr = head_list[-1]
    # # events = db.fetch_events(hdr, fill=False)
    # # num_events = len(list(events))
    # # print('%s events found' % num_events)
    # ev = db.fetch_events(hdr)
    #
    # events = []
    # for idx, event in enumerate(ev):
    #     if idx % 1000 == 0:
    #         print('event %s loaded' % (idx+1))
    #     events.append(event)
    #
    # muxer = dm.from_events(events)
    # data = muxer.to_sparse_dataframe()
    fields = ['xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3',
              'ssx[um]', 'ssy[um]', 'ssx', 'ssy', 'sclr1_ch3', 'sclr1_ch4']
    d = db.get_table(db[runid], fields=fields)
    return d


def read_runid(runid, c_list, dshape=None):
    """
    Read data from databroker.

    .. note:: Requires the databroker package from NSLS2

    .. note:: Not currently used in the gui

    Parameters
    ----------
    runid : int
        ID for given experimental measurement
    c_list : list
        channel list

    Returns
    -------
    data_dict : dict
        with fitting data
    data_sets : dict
        data from each channel and channel summed
    """
    data_dict = OrderedDict()
    data_sets = OrderedDict()

    # in case inputid is -1
    if runid == -1:
        hdr = db[-1]
        runid = hdr.scan_id

    data = fetch_data_from_db(runid)

    exp_keys = list(data.keys())

    sumv = None

    for c_name in c_list:
        channel_data = data[c_name]
        new_data = np.zeros([1, len(channel_data), len(channel_data[0])])

        for i in xrange(len(channel_data)):
            channel_data[i][pd.isnull(channel_data[i])] = 0
            new_data[0, i, :] = channel_data[i]

        file_channel = 'run_'+str(runid)+'_'+c_name
        DS = DataSelection(filename=file_channel,
                           raw_data=new_data)
        data_sets[file_channel] = DS

        if sumv is None:
            sumv = np.array(new_data)
        else:
            sumv += new_data

    file_channel = 'run_'+str(runid)
    DS = DataSelection(filename=file_channel,
                       raw_data=sumv)
    data_sets[file_channel] = DS

    temp = {}
    for v in exp_keys:
        if v not in c_list:
            # clean up nan data, should be done in lower level
            data[v][pd.isnull(data[v])] = 0
            pv_data = np.array(data[v])
            temp[v] = pv_data.reshape(dshape)
    data_dict['Run'+str(runid)+'_roi'] = temp

    return data_dict, data_sets


def read_xspress3_data(file_path):
    """
    Data IO for xspress3 format.

    Parameters
    ----------
    working_directory : str
        path folder
    file_name : str

    Returns
    -------
    data_output : dict
        with data from each channel
    """
    data_output = {}

    #file_path = os.path.join(working_directory, file_name)
    with h5py.File(file_path, 'r') as f:
        data = f['entry/instrument']

        # data from channel summed
        exp_data = np.asarray(data['detector/data'])
        xval = np.asarray(data['NDAttributes/NpointX'])
        yval = np.asarray(data['NDAttributes/NpointY'])

    # data size is (ysize, xsize, num of frame, num of channel, energy channel)
    exp_data = np.sum(exp_data, axis=2)
    num_channel = exp_data.shape[2]
    # data from each channel
    for i in range(num_channel):
        channel_name = 'channel_'+str(i+1)
        data_output.update({channel_name: exp_data[:, :, i, :]})

    # change x,y to 2D array
    xval = xval.reshape(exp_data.shape[0:2])
    yval = yval.reshape(exp_data.shape[0:2])

    data_output.update({'x_pos': xval})
    data_output.update({'y_pos': yval})

    return data_output


def flip_data(input_data, subscan_dims=None):
    """
    Flip 2D or 3D array. The flip happens on the second index of shape.
    .. warning :: This function mutates the input values.

    Parameters
    ----------
    input_data : 2D or 3D array.

    Returns
    -------
    flipped data
    """
    new_data = np.asarray(input_data)
    data_shape = input_data.shape
    if len(data_shape) == 2:
        if subscan_dims is None:
            new_data[1::2, :] = new_data[1::2, ::-1]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :] = new_data[start:end:2, ::-1]
                i += ny

    if len(data_shape) == 3:
        if subscan_dims is None:
            new_data[1::2, :, :] = new_data[1::2, ::-1, :]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :, :] = new_data[start:end:2, ::-1, :]
                i += ny
    return new_data


def output_data(fpath, output_folder,
                file_format='tiff', norm_name=None, use_average=True):
    """
    Read data from h5 file and transfer them into txt.

    Parameters
    ----------
    fpath : str
        path to h5 file
    output_folder : str
        which folder to save those txt file
    file_format : str, optional
        tiff or txt
    norm_name : str, optional
        if given, normalization will be performed.
    use_average : Bool, optional
        when normalization, multiply mean value of denomenator,
        i.e., norm_data = data1/data2 * np.mean(data2)
    """

    with h5py.File(fpath, 'r') as f:
        tmp = output_folder.split(sep_v)[-1]
        name_append = tmp.split('_')[-1]
        if name_append.isdigit():
            name_append = '_'+name_append
        else:
            name_append = ''
        detlist = list(f['xrfmap'].keys())
        fit_output = {}

        for detname in detlist:
            # fitted data
            if 'xrf_fit' in f['xrfmap/'+detname]:
                fit_data = f['xrfmap/'+detname+'/xrf_fit']
                fit_name = f['xrfmap/'+detname+'/xrf_fit_name']
                fit_name = helper_decode_list(fit_name)
                for i in np.arange(len(fit_name)):
                    fit_output[detname+'_'+fit_name[i]] = np.asarray(fit_data[i, :, :])
            # fitted error
            if 'xrf_fit_error' in f['xrfmap/'+detname]:
                error_data = f['xrfmap/'+detname+'/xrf_fit_error']
                error_name = f['xrfmap/'+detname+'/xrf_fit_error_name']
                error_name = helper_decode_list(error_name)

                for i in np.arange(len(error_name)):
                    fit_output[detname+'_'+error_name[i]+'_error'] = np.asarray(error_data[i, :, :])

        # ic data
        if 'scalers' in f['xrfmap']:
            ic_data = f['xrfmap/scalers/val']
            ic_name = f['xrfmap/scalers/name']
            ic_name = helper_decode_list(ic_name)
            for i in np.arange(len(ic_name)):
                fit_output[ic_name[i]] = np.asarray(ic_data[:, :, i])

        # position data
        if 'positions' in f['xrfmap']:
            pos_name = f['xrfmap/positions/name']
            pos_name = helper_decode_list(pos_name)
            for i, n in enumerate(pos_name):
                fit_output[n] = np.asarray(f['xrfmap/positions/pos'].value[i, :])

    # more data from suitcase part
    data_sc = {}
    #data_sc = retrieve_data_from_hdf_suitcase(fpath)
    if len(data_sc) != 0:
        fit_output.update(data_sc)

    #save data
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    if norm_name is not None:
        ic_v = fit_output[str(norm_name)]
        norm_sign = '_norm'
        for k, v in six.iteritems(fit_output):
            if 'pos' in k or 'r2' in k:
                continue
            ave = 1.0
            if use_average == True:
                ave = np.mean(ic_v)
            v = v/ic_v * ave

            _fname = k + name_append + norm_sign
            if file_format == 'tiff':
                fname = os.path.join(output_folder, _fname + '.tiff')
                sio.imsave(fname, v.astype(np.float32))
            elif file_format == 'txt':
                fname = os.path.join(output_folder, _fname + '.txt')
                np.savetxt(fname, v.astype(np.float32))
            else:
                pass

    for k, v in six.iteritems(fit_output):
        _fname = k + name_append
        if file_format == 'tiff':
            fname = os.path.join(output_folder, _fname + '.tiff')
            sio.imsave(fname, v.astype(np.float32))
        elif file_format == 'txt':
            fname = os.path.join(output_folder, _fname + '.txt')
            np.savetxt(fname, v.astype(np.float32))
        else:
            pass


def read_hdf_APS(working_directory,
                 file_name, spectrum_cut=3000,
                 load_summed_data=True,
                 load_each_channel=True):
    """
    Data IO for files similar to APS Beamline 13 data format.
    This might be changed later.

    Parameters
    ----------
    working_directory : str
        path folder
    file_name : str
        selected h5 file
    spectrum_cut : int, optional
        only use spectrum from, say 0, 3000
    load_summed_data : bool, optional
        load summed spectrum or not
    load_each_channel : bool, optional
        load data from each channel or not
    other_list : list, optional
        data dumped from suitcase

    Returns
    -------
    data_dict : dict
        with fitting data
    data_sets : dict
        data from each channel and channel summed, a dict of DataSelection objects
    """
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    file_path = os.path.join(working_directory, file_name)

    # defined in other_list in config file
    try:
        dict_sc = retrieve_data_from_hdf_suitcase(file_path)
    except:
        dict_sc = {}

    with h5py.File(file_path, 'r+') as f:
        data = f['xrfmap']
        fname = file_name.split('.')[0]
        if load_summed_data is True:
            try:
                # data from channel summed
                exp_data = np.array(data['detsum/counts'][:, :, 0:spectrum_cut])
                logger.warning('We use spectrum range from 0 to {}'.format(spectrum_cut))
                logger.info('Exp. data from h5 has shape of: {}'.format(exp_data.shape))

                fname_sum = fname+'_sum'
                DS = DataSelection(filename=fname_sum,
                                   raw_data=exp_data)

                data_sets[fname_sum] = DS
                logger.info('Data of detector sum is loaded.')
            except KeyError:
                print('No data is loaded for detector sum.')

        if 'scalers' in data:
            det_name = data['scalers/name']
            temp = {}
            for i, n in enumerate(det_name):
                if not isinstance(n, six.string_types):
                    n = n.decode()
                temp[n] = data['scalers/val'].value[:, :, i]
            img_dict[fname+'_scaler'] = temp
            # also dump other data from suitcase if required
            if len(dict_sc) != 0:
                img_dict[fname+'_scaler'].update(dict_sc)

        if 'positions' in data:
            pos_name = data['positions/name']
            temp = {}
            for i, n in enumerate(pos_name):
                if not isinstance(n, six.string_types):
                    n = n.decode()
                temp[n] = data['positions/pos'].value[i, :]
            img_dict['positions'] = temp

        # find total channel:
        channel_num = 0
        for v in list(data.keys()):
            if 'det' in v:
                channel_num = channel_num+1
        channel_num = channel_num-1  # do not consider det_sum

        # data from each channel
        if load_each_channel:
            for i in range(1, channel_num+1):
                det_name = 'det'+str(i)
                file_channel = fname+'_det'+str(i)
                try:
                    exp_data_new = np.array(data[det_name+'/counts'][:, :, 0:spectrum_cut])
                    DS = DataSelection(filename=file_channel,
                                       raw_data=exp_data_new)
                    data_sets[file_channel] = DS
                    logger.info('Data from detector channel {} is loaded.'.format(i))
                except KeyError:
                    print('No data is loaded for {}.'.format(det_name))

                if 'xrf_fit' in data[det_name]:
                    try:
                        fit_result = get_fit_data(data[det_name]['xrf_fit_name'].value,
                                                  data[det_name]['xrf_fit'].value)
                        img_dict.update({file_channel+'_fit': fit_result})
                        # also include scaler data
                        if 'scalers' in data:
                            img_dict[file_channel+'_fit'].update(img_dict[fname+'_scaler'])
                    except IndexError:
                        logger.info('No fitting data is loaded for channel {}.'.format(i))

        if 'roimap' in data:
            if 'sum_name' in data['roimap']:
                det_name = data['roimap/sum_name']
                temp = {}
                for i, n in enumerate(det_name):
                    temp[n] = data['roimap/sum_raw'].value[:, :, i]
                    # bad points on first one
                    try:
                        temp[n][0, 0] = temp[n][1, 0]
                    except IndexError:
                        temp[n][0, 0] = temp[n][0, 1]
                img_dict[fname+'_roi'] = temp
                # also include scaler data
                if 'scalers' in data:
                    img_dict[fname+'_roi'].update(img_dict[fname+'_scaler'])

            if 'det_name' in data['roimap']:
                det_name = data['roimap/det_name']
                temp = {}
                for i, n in enumerate(det_name):
                    temp[n] = data['roimap/det_raw'].value[:, :, i]
                    try:
                        temp[n][0, 0] = temp[n][1, 0]
                    except IndexError:
                        temp[n][0, 0] = temp[n][0, 1]
                img_dict[fname+'_roi_each'] = temp

        # read fitting results from summed data
        if 'xrf_fit' in data['detsum']:
            try:
                fit_result = get_fit_data(data['detsum']['xrf_fit_name'].value,
                                          data['detsum']['xrf_fit'].value)
                img_dict.update({fname+'_fit': fit_result})
                if 'scalers' in data:
                    img_dict[fname+'_fit'].update(img_dict[fname+'_scaler'])
            except (IndexError, KeyError):
                logger.info('No fitting data is loaded for channel summed data.')


    return img_dict, data_sets


def retrieve_data_from_hdf_suitcase(fpath):
    """
    Retrieve data from suitcase part in hdf file.
    Data name is defined in config file.
    """
    data_dict = {}
    with h5py.File(fpath, 'r+') as f:
        other_data_list = [v for v in f.keys() if v!='xrfmap']
        if len(other_data_list) > 0:
            f_hdr = f[other_data_list[0]].attrs['start']
            if not isinstance(f_hdr, six.string_types):
                f_hdr = f_hdr.decode('utf-8')
            start_doc = ast.literal_eval(f_hdr)
            other_data = f[other_data_list[0]+'/primary/data']

            if start_doc['beamline_id'] == 'HXN':
                current_dir = os.path.dirname(os.path.realpath(__file__))
                config_file = 'hxn_pv_config.json'
                config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
                with open(config_path, 'r') as json_data:
                    config_data = json.load(json_data)
                extra_list = config_data['other_list']
                fly_type = start_doc.get('fly_type', None)
                subscan_dims = start_doc.get('subscan_dims', None)

                if 'dimensions' in start_doc:
                    datashape = start_doc['dimensions']
                elif 'shape' in start_doc:
                    datashape = start_doc['shape']
                else:
                    logger.error('No dimension/shape is defined in hdr.start.')

                datashape = [datashape[1], datashape[0]]  # vertical first, then horizontal
                for k in extra_list:
                    #k = k.encode('utf-8')
                    if k not in other_data.keys():
                        continue
                    _v = np.array(other_data[k])
                    v = _v.reshape(datashape)
                    if fly_type in ('pyramid',):
                        # flip position the same as data flip on det counts
                        v = flip_data(v, subscan_dims=subscan_dims)
                    data_dict[k] = v
    return data_dict


def read_MAPS(working_directory,
              file_name, channel_num=1):
    data_dict = OrderedDict()
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 0

    fit_val = None
    fit_v_pyxrf = None

    file_path = os.path.join(working_directory, file_name)
    print('file path is {}'.format(file_path))

    with h5py.File(file_path, 'r+') as f:

        data = f['MAPS']
        fname = file_name.split('.')[0]

        # for 2D MAP
        #data_dict[fname] = data

        # raw data
        exp_data = data['mca_arr'][:]

        # data from channel summed
        roi_channel = data['channel_names'].value
        roi_val = data['XRF_roi'][:]

        scaler_names = data['scaler_names'].value
        scaler_val = data['scalers'][:]

        try:
            # data from fit
            fit_val = data['XRF_fits'][:]
        except KeyError:
            logger.info('No fitting from MAPS can be loaded.')

        try:
            fit_data = f['xrfmap/detsum']
            fit_v_pyxrf = fit_data['xrf_fit'][:]
            fit_n_pyxrf = fit_data['xrf_fit_name'].value
            print(fit_n_pyxrf)
        except KeyError:
            logger.info('No fitting from pyxrf can be loaded.')

    exp_shape = exp_data.shape
    exp_data = exp_data.T
    exp_data = np.rot90(exp_data, 1)
    logger.info('File : {} with total counts {}'.format(fname,
                                                        np.sum(exp_data)))
    DS = DataSelection(filename=fname,
                       raw_data=exp_data)
    data_sets.update({fname: DS})

    # save roi and fit into dict

    temp_roi = {}
    temp_fit = {}
    temp_scaler = {}
    temp_pos = {}

    for i, name in enumerate(roi_channel):
        temp_roi[name] = np.flipud(roi_val[i, :, :])
    img_dict[fname+'_roi'] = temp_roi

    if fit_val is not None:
        for i, name in enumerate(roi_channel):
            temp_fit[name] = fit_val[i, :, :]
        img_dict[fname+'_fit_MAPS'] = temp_fit

    cut_bad_col = 1
    if fit_v_pyxrf is not None:
        for i, name in enumerate(fit_n_pyxrf):
            temp_fit[name] = fit_v_pyxrf[i, :, cut_bad_col:]
        img_dict[fname+'_fit'] = temp_fit

    for i, name in enumerate(scaler_names):
        if name == 'x_coord':
            temp_pos['x_pos'] = np.flipud(scaler_val[i, :, :])
        elif name == 'y_coord':
            temp_pos['y_pos'] = np.flipud(scaler_val[i, :, :])
        else:
            temp_scaler[name] = np.flipud(scaler_val[i, :, :])
    img_dict[fname+'_scaler'] = temp_scaler
    img_dict['positions'] = temp_pos

    # read fitting results
    # if 'xrf_fit' in data[detID]:
    #     fit_result = get_fit_data(data[detID]['xrf_fit_name'].value,
    #                               data[detID]['xrf_fit'].value)
    #     img_dict.update({fname+'_fit': fit_result})

    return img_dict, data_sets


def get_roi_sum(namelist, data_range, data):
    data_temp = dict()
    for i in range(len(namelist)):
        lowv = data_range[i, 0]
        highv = data_range[i, 1]
        data_sum = np.sum(data[:, :, lowv: highv], axis=2)
        data_temp.update({namelist[i]: data_sum})
        #data_temp.update({namelist[i].replace(' ', '_'): data_sum})
    return data_temp


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
    for i,v in enumerate(namelist):
        if not isinstance(v, six.string_types):
            v = v.decode()
        data_temp.update({v: data[i, :, :]})
    return data_temp


def read_hdf_to_stitch(working_directory, filelist,
                       shape, ignore_file=None):
    """
    Read fitted results from each hdf file, and stitch them together.

    Parameters
    ----------
    working_directory : str
        folder with all the h5 files and also the place to save output
    filelist : list of str
        names for all the h5 files
    shape : list or tuple
        shape defines how to stitch all the h5 files. [veritcal, horizontal]
    ignore_file : list of str
        to be implemented

    Returns
    -------
    dict :
        combined results from each h5 file
    """
    out = {}
    shape_v = {}
    horizontal_v = 0
    vertical_v = 0
    h_index = np.zeros(shape)
    v_index = np.zeros(shape)

    for i, file_name in enumerate(filelist):
        img, _ = read_hdf_APS(working_directory, file_name,
                              load_summed_data=False, load_each_channel=False)
        tmp_shape = img['positions']['x_pos'].shape
        m = i // shape[1]
        n = i % shape[1]

        if n == 0:
            h_step = 0

        h_index[m][n] = h_step
        v_index[m][n] = m * tmp_shape[0]
        h_step += tmp_shape[1]

        if i<shape[1]:
            horizontal_v += tmp_shape[1]
        if i%shape[1] == 0:
            vertical_v += tmp_shape[0]
        if i == 0:
            out = copy.deepcopy(img)

    data_tmp = np.zeros([vertical_v, horizontal_v])

    for k, v in six.iteritems(out):
        for m, n in six.iteritems(v):
            v[m] = np.array(data_tmp)

    for i, file_name in enumerate(filelist):
        img, _ = read_hdf_APS(working_directory, file_name,
                              load_summed_data=False, load_each_channel=False)

        tmp_shape = img['positions']['x_pos'].shape
        m = i // shape[1]
        n = i % shape[1]
        h_i = h_index[m][n]
        v_i = v_index[m][n]

        keylist = ['fit', 'scaler', 'position']

        for key_name in keylist:
            fit_key0, = [v for v in list(out.keys()) if key_name in v]
            fit_key, = [v for v in list(img.keys()) if key_name in v]
            for k, v in six.iteritems(img[fit_key]):
                out[fit_key0][k][v_i:v_i+tmp_shape[0], h_i:h_i+tmp_shape[1]] = img[fit_key][k]

    return out


def make_hdf_stitched(working_directory, filelist, fname,
                      shape):
    """
    Read fitted results from each hdf file, stitch them together and save to
    a new h5 file.

    Parameters
    ----------
    working_directory : str
        folder with all the h5 files and also the place to save output
    filelist : list of str
        names for all the h5 files
    fname : str
        name of output h5 file
    shape : list or tuple
        shape defines how to stitch all the h5 files. [veritcal, horizontal]
    """
    print('Reading data from each hdf file.')
    fpath = os.path.join(working_directory, fname)
    out = read_hdf_to_stitch(working_directory, filelist, shape)

    result = {}
    img_shape = None
    for k, v in six.iteritems(out):
        for m, n in six.iteritems(v):
            if img_shape is None:
                img_shape = n.shape
            result[m] = n.ravel()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = 'srx_pv_config.json'
    config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)

    print('Saving all the data into one hdf file.')
    write_db_to_hdf(fpath, result,
                    img_shape,
                    det_list=config_data['xrf_detector'],
                    pos_list=('x_pos', 'y_pos'),
                    scaler_list=config_data['scaler_list'],
                    base_val=config_data['base_value'])  #base value shift for ic


    fitkey, = [v for v in list(out.keys()) if 'fit' in v]
    save_fitdata_to_hdf(fpath, out[fitkey])

    print('Done!')


def get_data_from_folder_helper(working_directory, foldername,
                                filename, flip_h=False):
    """
    Read fitted data from given folder.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    foldername : string
        folder name of given fitting result
    filename : string
        given element
    flip_h : bool
        x position is saved in a wrong way, so we may want to flip left right on the data,
        to be removed.

    Returns
    -------
    2D array
    """
    fpath = os.path.join(working_directory, foldername, filename)
    if 'txt' in filename:
        data = np.loadtxt(fpath)
    elif 'tif' in filename:
        data = np.array(Image.open(fpath))

    # x position is saved in a wrong way
    if flip_h == True:
        data = np.fliplr(data)
    return data


def get_data_from_multiple_folders_helper(working_directory, folderlist,
                                          filename, flip_h=False):
    """
    Read given element from fitted results in multiple folders.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    folderlist : list
        list of folder names saving fitting result
    filename : string
        given element
    flip_h : bool
        x position is saved in a wrong way, so we may want to flip left right on the data,
        to be removed.

    Returns
    -------
    2D array
    """
    output = np.array([])
    for foldername in folderlist:
        result = get_data_from_folder_helper(working_directory, foldername,
                                             filename, flip_h=flip_h)
        output = np.concatenate([output, result.ravel()])
    return output


def stitch_fitted_results(working_directory, folderlist, output=None):
    """
    Stitch fitted data from multiple folders. Output stiched results as 1D array.

    Parameters
    ----------
    working_directory : string
        overall folder path where multiple fitting results are saved
    folderlist : list
        list of folder names saving fitting result
    output : string, optional
        output folder name to save all the stiched results.
    """

    # get all filenames
    fpath = os.path.join(working_directory, folderlist[0], '*')
    pathlist = [name for name in glob.glob(fpath)]
    filelist = [name.split(sep_v)[-1] for name in pathlist]
    out = {}
    for filename in filelist:
        if 'x_pos' in filename:
            flip_h = True
        else:
            flip_h = False
        data = get_data_from_multiple_folders_helper(working_directory, folderlist,
                                                     filename, flip_h=flip_h)
        out[filename.split('.')[0]]=data

    if output is not None:
        outfolder = os.path.join(working_directory, output)
        if os.path.exists(outfolder) is False:
            os.mkdir(outfolder)
        for k, v in out.items():
            outpath = os.path.join(outfolder, k+'_stitched.txt')
            np.savetxt(outpath, v)
    return out


def save_fitdata_to_hdf(fpath, data_dict,
                        datapath='xrfmap/detsum',
                        data_saveas='xrf_fit',
                        dataname_saveas='xrf_fit_name'):
    """
    Add fitting results to existing h5 file. This is to be moved to filestore.

    Parameters
    ----------
    fpath : str
        path of the hdf5 file
    data_dict : dict
        dict of array
    datapath : str
        path inside h5py file
    data_saveas : str, optional
        name in hdf for data array
    dataname_saveas : str, optional
        name list in hdf to explain what the saved data mean
    """
    f = h5py.File(fpath, 'a')
    try:
        dataGrp = f.create_group(datapath)
    except ValueError:
        dataGrp=f[datapath]

    data = []
    namelist = []
    for k, v in six.iteritems(data_dict):
        if not isinstance(k, six.string_types):
            k = k.decode()
        namelist.append(k)
        data.append(v)

    if data_saveas in dataGrp:
        del dataGrp[data_saveas]

    data = np.asarray(data)
    ds_data = dataGrp.create_dataset(data_saveas, data=data)
    ds_data.attrs['comments'] = ' '

    if dataname_saveas in dataGrp:
        del dataGrp[dataname_saveas]

    if not isinstance(dataname_saveas, six.string_types):
        dataname_saveas = dataname_saveas.decode()
    namelist = np.array(namelist).astype('|S9')
    name_data = dataGrp.create_dataset(dataname_saveas, data=namelist)
    name_data.attrs['comments'] = ' '

    f.close()


def read_xspress(file_name):
    """
    Data from xspress file format.

    Parameters
    ----------
    file_name : str
        file path

    Returns
    -------
    array :
        data with shape [2D_size1, 2D_size2, num_frame, num_channel, num_energy_channel]
    """

    file_path = os.path.join(file_name)
    f = h5py.File(file_path, 'r+')
    data = f['entry/instrument/detector/data']

    return np.array(data)


def write_data_to_hdf(fpath, data, bin_frame=True, channel_n=4):
    """
    Assume data is obained from databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : array
        data from data broker
    bin_frame : bool, optional
        true when data has multiple frames per point
    channel_n : int, optional
        number of detector channels
    """

    if bin_frame is True:
        data = np.sum(data, 2)

    interpath = 'xrfmap'
    f = h5py.File(fpath, 'a')

    for i in range(channel_n):
        detname = 'det'+str(i+1)
        try:
            dataGrp = f.create_group(interpath+'/'+detname)
        except ValueError:
            dataGrp = f[interpath+'/'+detname]

        if 'counts' in dataGrp:
            del dataGrp['counts']
        ds_data = dataGrp.create_dataset('counts', data=data[:, :, i, :])
        ds_data.attrs['comments'] = 'Experimental data from channel ' + str(i)

    # summed data
    try:
        dataGrp = f.create_group(interpath+'/detsum')
    except ValueError:
        dataGrp = f[interpath+'/detsum']

    if 'counts' in dataGrp:
        del dataGrp['counts']
    ds_data = dataGrp.create_dataset('counts', data=np.sum(data, axis=2))
    ds_data.attrs['comments'] = 'Experimental data from channel sum'

    f.close()


def transfer_xspress(fpath, output_path):
    """
    Transfer xspress h5 file to file which can be taken by pyxrf.

    Parameters
    ----------
    fpath : str
        input file path
    output_path : str
        path to save output file
    """
    d = read_xspress(fpath)
    write_data_to_hdf(output_path, d)


def write_db_to_hdf(fpath, data, datashape,
                    det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
                    pos_list=('zpssx[um]', 'zpssy[um]'),
                    scaler_list=('sclr1_ch3', 'sclr1_ch4'),
                    fly_type=None, subscan_dims=None, base_val=None):
    """
    Assume data is obained from databroker, and save the data to hdf file.
    This function can handle stopped/aborted scans.

    .. note:: This function should become part of suitcase

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    """
    interpath = 'xrfmap'
    with h5py.File(fpath, 'a') as f:

        sum_data = None
        new_v_shape = datashape[0]  # to be updated if scan is not completed
        spectrum_len = 4096  # standard

        for n, c_name in enumerate(det_list):
            if c_name in data:
                detname = 'det'+str(n+1)
                dataGrp = f.create_group(interpath+'/'+detname)

                logger.info('read data from %s' % c_name)
                channel_data = data[c_name]

                # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
                new_v_shape = len(channel_data) // datashape[1]

                new_data = np.vstack(channel_data)
                new_data = new_data[:new_v_shape*datashape[1], :]

                new_data = new_data.reshape([new_v_shape, datashape[1],
                                             len(channel_data[1])])
                if new_data.shape[2] != spectrum_len:
                    # merlin detector has spectrum len 2048
                    # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                    new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                    new_tmp[:,:,:new_data.shape[2]] = new_data
                    new_data = new_tmp
                if fly_type in ('pyramid',):
                    new_data = flip_data(new_data, subscan_dims=subscan_dims)

                if sum_data is None:
                    sum_data = new_data
                else:
                    sum_data += new_data
                ds_data = dataGrp.create_dataset('counts', data=new_data, compression='gzip')
                ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)

        # summed data
        dataGrp = f.create_group(interpath+'/detsum')

        if sum_data is not None:
            sum_data = sum_data.reshape([new_v_shape, datashape[1],
                                         spectrum_len])
            ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel sum'

        # position data
        dataGrp = f.create_group(interpath+'/positions')

        pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                     datashape)

        for i in range(len(pos_names)):
            if 'x' in pos_names[i]:
                pos_names[i] = 'x_pos'
            elif 'y' in pos_names[i]:
                pos_names[i] = 'y_pos'

        # need to change shape to sth like [2, 100, 100]
        data_temp = np.zeros([pos_data.shape[2], pos_data.shape[0], pos_data.shape[1]])
        for i in range(pos_data.shape[2]):
            data_temp[i,:,:] = pos_data[:,:,i]

        if fly_type in ('pyramid',):
            for i in range(data_temp.shape[0]):
                # flip position the same as data flip on det counts
                data_temp[i,:,:] = flip_data(data_temp[i,:,:], subscan_dims=subscan_dims)

        dataGrp.create_dataset('name', data=helper_encode_list(pos_names))
        dataGrp.create_dataset('pos', data=data_temp[:,:new_v_shape,:])

        # scaler data
        dataGrp = f.create_group(interpath+'/scalers')

        scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                           datashape)

        if fly_type in ('pyramid',):
            scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)

        dataGrp.create_dataset('name', data=helper_encode_list(scaler_names))

        if base_val is not None:  # base line shift for detector, for SRX
            base_val = np.array([base_val])
            if len(base_val) == 1:
                scaler_data = np.abs(scaler_data - base_val)
            else:
                for i in scaler_shape.shape[2]:
                    scaler_data[:,:,i] = np.abs(scaler_data[:,:,i] - base_val[i])

        dataGrp.create_dataset('val', data=scaler_data[:new_v_shape,:])


def map_data2D(data, datashape,
               det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
               pos_list=('zpssx[um]', 'zpssy[um]'),
               scaler_list=('sclr1_ch3', 'sclr1_ch4'),
               fly_type=None, subscan_dims=None, spectrum_len=4096):
    """
    Data is obained from databroker. Transfer items from data to a dictionay of
    numpy array, which has 2D shape same as scanning area.

    This function can handle stopped/aborted scans. Raster scan (snake scan) is
    also considered.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    fly_type : string or optional
        raster scan (snake scan) or normal
    subscan_dims : 1D array or optional
        used at HXN, 2D of a large area is split into small area scans
    spectrum_len : int, optional
        standard spectrum length

    Returns
    -------
    dict of numpy array
    """
    data_output = {}
    sum_data = None
    new_v_shape = datashape[0]  # updated if scan is not completed

    for n, c_name in enumerate(det_list):
        if c_name in data:
            detname = 'det'+str(n+1)
            logger.info('read data from %s' % c_name)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]
            new_data = np.vstack(channel_data)
            new_data = new_data[:new_v_shape*datashape[1], :]
            new_data = new_data.reshape([new_v_shape, datashape[1],
                                         len(channel_data[1])])
            if new_data.shape[2] != spectrum_len:
                # merlin detector has spectrum len 2048
                # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                new_tmp[:,:,:new_data.shape[2]] = new_data
                new_data = new_tmp
            if fly_type in ('pyramid',):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)
            data_output[detname] = new_data

    # scanning position data
    pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                 datashape)
    for i in range(len(pos_names)):
        if 'x' in pos_names[i]:
            pos_names[i] = 'x_pos'
        elif 'y' in pos_names[i]:
            pos_names[i] = 'y_pos'
    if fly_type in ('pyramid',):
        for i in range(pos_data.shape[2]):
            # flip position the same as data flip on det counts
            pos_data[:, :, i] = flip_data(pos_data_temp[:, :, i], subscan_dims=subscan_dims)
    for i, v in enumerate(pos_names):
        data_output[v] = pos_data[:, :, i]

    # scaler data
    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                       datashape)
    if fly_type in ('pyramid',):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)
    for i, v in enumerate(scaler_names):
        data_output[v] = scaler_data[:, :, i]
    return data_output


def write_db_to_hdf_base(fpath, data, num_det=3, create_each_det=True):
    """
    Data is obained based on databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : dict
        fluorescence data with scaler value and positions
    num_det : int
        number of detector
    create_each_det : Bool, optional
        if number of point is too large, only sum data is saved in h5 file
    """
    interpath = 'xrfmap'
    sum_data = None

    with h5py.File(fpath, 'a') as f:
        if create_each_det is True:
            for n in range(num_det):
                detname = 'det' + str(n+1)
                new_data = data[detname]

                if sum_data is None:
                    sum_data = new_data
                else:
                    sum_data += new_data

                dataGrp = f.create_group(interpath+'/'+detname)
                ds_data = dataGrp.create_dataset('counts', data=new_data, compression='gzip')
                ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)
        else:
            sum_data = data['det_sum']

        # summed data
        if sum_data is not None:
            dataGrp = f.create_group(interpath+'/detsum')
            ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel sum'

        # add positions
        if 'pos_names' in data:
            dataGrp = f.create_group(interpath+'/positions')
            pos_names = data['pos_names']
            pos_data = data['pos_data']
            dataGrp.create_dataset('name', data=helper_encode_list(pos_names))
            dataGrp.create_dataset('pos', data=pos_data)

        # scaler data
        if 'scaler_data' in data:
            dataGrp = f.create_group(interpath+'/scalers')
            scaler_names = data['scaler_names']
            scaler_data = data['scaler_data']
            dataGrp.create_dataset('name', data=helper_encode_list(scaler_names))
            dataGrp.create_dataset('val', data=scaler_data)


def helper_encode_list(data, data_type='utf-8'):
    return [d.encode(data_type) for d in data]


def helper_decode_list(data, data_type='utf-8'):
    return [d.decode(data_type) for d in data]


def get_name_value_from_db(name_list, data, datashape):
    """
    Get name and data from db.
    """
    pos_names = []
    pos_data = np.zeros([datashape[0], datashape[1], len(name_list)])
    for i, v in enumerate(name_list):
        posv = np.zeros(datashape[0]*datashape[1])  # keep shape unchanged, so stopped/aborted run can be handled.
        data[v] = np.asarray(data[v])  # in case data might be list
        posv[:data[v].shape[0]] = np.asarray(data[v])
        pos_data[:, :, i] = posv.reshape([datashape[0], datashape[1]])
        pos_names.append(str(v))
    return pos_names, pos_data


def _make_hdf(fpath, runid, full_data=True,
              create_each_det=False, save_scalar=True,
              num_end_lines_excluded=None):
    """
    Save the data from databroker to hdf file.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    fpath: str
        path to save hdf file
    runid : int
        id number for given run
    full_data : bool, optional
        save baseline data and all other information if True
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines
    """
    hdr_tmp = db[-1]
    print('Loading data from database.')

    if hdr_tmp.start.beamline_id == 'HXN':
        hdr = db[runid]

        start_doc = hdr['start']
        if 'dimensions' in start_doc:
            datashape = start_doc.dimensions
        elif 'shape' in start_doc:
            datashape = start_doc.shape
        else:
            logger.error('No dimension/shape is defined in hdr.start.')

        datashape = [datashape[1], datashape[0]]  # vertical first, then horizontal
        fly_type = start_doc.get('fly_type', None)
        subscan_dims = start_doc.get('subscan_dims', None)

        if 'motors' in hdr.start:
            pos_list = hdr.start.motors
        elif 'axes' in hdr.start:
            pos_list = hdr.start.axes
        else:
            pos_list = ['zpssx[um]', 'zpssy[um]']

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = 'hxn_pv_config.json'
        config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
        with open(config_path, 'r') as json_data:
            config_data = json.load(json_data)

        mercury_det = ['mercury1_mca_spectrum']
        keylist =  hdr.descriptors[0].data_keys.keys()
        #xspress3_det = config_data['xrf_detector']
        xspress3_det = [v for v in keylist if 'xspress3' in v]  # find xspress3 det with key word matching
        if mercury_det[0] in keylist:
            det_list = xspress3_det + mercury_det
        elif not len(xspress3_det) and mercury_det[0] in keylist:
            det_list = mercury_det
        else:
            det_list = xspress3_det

        scaler_list_all = config_data['scaler_list']

        all_keys = hdr.descriptors[0].data_keys.keys()
        scaler_list = [v for v in scaler_list_all if v in all_keys]

        fields = det_list + scaler_list + pos_list
        data = db.get_table(hdr, fill=True)

        print('Saving data to hdf file.')
        write_db_to_hdf(fpath, data, datashape,
                        det_list=det_list, pos_list=pos_list,
                        scaler_list=scaler_list,
                        fly_type=fly_type, subscan_dims=subscan_dims)

        # use suitcase to save baseline data, and scaler data from primary
        tmp = set()
        for descriptor in hdr.descriptors:
            # no 3D vector data
            xs3 = [key for key in descriptor.data_keys.keys() if 'xspress3' in key]
            tmp.update(xs3)
            tmp.add('merlin1')
        fds = sc.filter_fields(hdr, tmp)
        if full_data == True:
            sc.export(hdr, fpath, db.mds, fields=fds, use_uid=False)

    elif hdr_tmp.start.beamline_id == 'xf05id' or str(hdr_tmp.start.beamline_id) == 'SRX':
        _make_hdf_srx(fpath, runid, create_each_det=False,
                      save_scalar=save_scalar,
                      num_end_lines_excluded=num_end_lines_excluded)
        print('Done!')
    else:
        print("Databroker is not setup for this beamline")

    free_memory_from_handler()


def _make_hdf_srx(fpath, runid, create_each_det=False,
                  save_scalar=True, num_end_lines_excluded=None):
    """
    Save the data from databroker to hdf file for SRX beamline.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    fpath: str
        path to save hdf file
    runid : int
        id number for given run
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines
    """
    hdr = db[runid]
    spectrum_len = 4096
    start_doc = hdr['start']
    plan_n = start_doc.get('plan_name')
    if 'fly' not in plan_n: # not fly scan
        datashape = start_doc['shape']   # vertical first then horizontal
        fly_type = None

        snake_scan = start_doc.get('snaking')
        if snake_scan[1] == True:
            fly_type = 'pyramid'

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = 'srx_pv_config.json'
        config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
        with open(config_path, 'r') as json_data:
            config_data = json.load(json_data)

        try:
            data = db.get_table(hdr, fill=True, convert_times=False)
        except IndexError:
            total_len = get_total_scan_point(hdr) - 2

            evs, _ = zip(*zip(get_events(hdr, fill=True), range(total_len)))

            namelist = config_data['xrf_detector'] +hdr.start.motors +config_data['scaler_list']

            dictv = {v:[] for v in namelist}

            for e in evs:
                for k,v in six.iteritems(dictv):
                    dictv[k].append(e.data[k])

            data = pd.DataFrame(dictv, index=np.arange(1, total_len+1)) # need to start with 1

        #express3 detector name changes in databroker
        xrf_detector_names = config_data['xrf_detector']
        if xrf_detector_names[0] not in data.keys():
            xrf_detector_names = ['xs_channel'+str(i) for i in range(1,4)]
        print('Saving data to hdf file.')
        write_db_to_hdf(fpath, data,
                        datashape,
                        det_list=xrf_detector_names,
                        pos_list=hdr.start.motors,
                        scaler_list=config_data['scaler_list'],
                        fly_type=fly_type,
                        base_val=config_data['base_value'])  #base value shift for ic
        if 'xs2' in hdr.start.detectors:
            print('Saving data to hdf file for second xspress3 detector.')
            tmp = fpath.split('.')
            fpath1 = '.'.join([tmp[0]+'_1', tmp[1]])
            write_db_to_hdf(fpath1, data,
                            datashape,
                            det_list=config_data['xrf_detector2'],
                            pos_list=hdr.start.motors,
                            scaler_list=config_data['scaler_list'],
                            fly_type=fly_type,
                            base_val=config_data['base_value'])  #base value shift for ic
    else:
        # srx fly scan
        if save_scalar is True:
            scaler_list = ['i0', 'time', 'i0_time', 'time_diff']
            xpos_name = 'enc1'
            ypos_name = 'hf_stage_y'
        # Added by AMK to allow flying of single element on xs2
        if 'E_tomo' in start_doc['scaninfo']['type']:
            num_det = 1
            ypos_name = 'e_tomo_y'
        else:
            num_det = 3
        vertical_fast = False  # assuming fast on x as default
        if num_end_lines_excluded is None:
            datashape = [start_doc['shape'][1], start_doc['shape'][0]]   # vertical first then horizontal, assuming fast scan on x
        else:
            datashape = [start_doc['shape'][1]-num_end_lines_excluded, start_doc['shape'][0]]
        if 'fast_axis' in hdr.start.scaninfo:
            if hdr.start.scaninfo['fast_axis'] in ('VER', 'DET2VER'):  # fast scan along vertical, y is fast scan, x is slow
                xpos_name = 'enc1'
                ypos_name = 'hf_stage_x'
                if 'E_tomo' in start_doc['scaninfo']['type']:
                    ypos_name = 'e_tomo_x'
                vertical_fast = True
                #datashape = [start_doc['shape'][0], start_doc['shape'][1]]   # fast vertical scan put shape[0] as vertical direction

        new_shape = datashape + [spectrum_len]
        total_points = datashape[0]*datashape[1]

        new_data = {}
        data = {}
        e = db.get_events(hdr, fill=True, stream_name='stream0')

        if save_scalar is True:
            new_data['scaler_names'] = scaler_list
            scaler_tmp = np.zeros([datashape[0], datashape[1], len(scaler_list)])
            if vertical_fast is True:  # data shape only has impact on scalar data
                scaler_tmp = np.zeros([datashape[1], datashape[0], len(scaler_list)])
            for v in scaler_list+[xpos_name]:
                data[v] = np.zeros([datashape[0], datashape[1]])

        if create_each_det is False:
            new_data['det_sum'] = np.zeros(new_shape)
        else:
            for i in range(num_det):
                new_data['det'+str(i+1)] = np.zeros(new_shape)

        for m,v in enumerate(e):
            if m < datashape[0]:
                if save_scalar is True:
                    for n in scaler_list[:-1]+[xpos_name]:
                        min_len = min(v.data[n].size, datashape[1])
                        data[n][m, :min_len] = v.data[n][:min_len]
                        if min_len < datashape[1]:  # position data or i0 has shorter length than fluor data
                            len_diff = datashape[1] - min_len
                            interp_list = (v.data[n][-1]-v.data[n][-3])/2*np.arange(1,len_diff+1) + v.data[n][-1]
                            data[n][m, min_len:datashape[1]] = interp_list
                if create_each_det is False:
                    for i in range(num_det):
                        new_data['det_sum'][m,:v.data['fluor'].shape[0],:] += v.data['fluor'][:,i,:]
                else:
                    for i in range(num_det):  # in case the data length in each line is different
                        new_data['det'+str(i+1)][m,:v.data['fluor'].shape[0],:] = v.data['fluor'][:,i,:]

        if vertical_fast is True: # need to transpose the data, as we scan y first
            if create_each_det is False:
                new_data['det_sum'] = np.transpose(new_data['det_sum'], axes=(1,0,2))
            else:
                for i in range(num_det):
                    new_data['det'+str(i+1)] = np.transpose(new_data['det'+str(i+1)], axes=(1,0,2))

        if save_scalar is True:
            if vertical_fast is False:
                for i,v in enumerate(scaler_list[:-1]):
                    scaler_tmp[:, :, i] = data[v]
                scaler_tmp[:, :-1, -1] = np.diff(data['time'], axis=1)
                scaler_tmp[:, -1, -1] = data['time'][:, -1] - data['time'][:, -2]
            else:
                for i,v in enumerate(scaler_list[:-1]):
                    scaler_tmp[:, :, i] = data[v].T
                data_t = data['time'].T
                scaler_tmp[:-1, :, -1] = np.diff(data_t, axis=0)
                scaler_tmp[-1, :, -1] = data_t[-1, :] - data_t[-2, :]
            new_data['scaler_data'] = scaler_tmp
            x_pos = data[xpos_name]

        # get y position data
        if save_scalar is True:
            data1 = db.get_table(hdr, fill=True, stream_name='primary')
            if num_end_lines_excluded is not None:
                data1 = data1[:datashape[0]]
            if ypos_name not in data1.keys() and 'E_tomo' not in start_doc['scaninfo']['type']:
                ypos_name = 'hf_stage_z'        #vertical along z
            y_pos0 = np.hstack(data1[ypos_name])
            if len(y_pos0) >= x_pos.shape[0]:  # y position is more than actual x pos, scan not finished?
                y_pos = y_pos0[:x_pos.shape[0]]
                x_tmp = np.ones(x_pos.shape[1])
                xv, yv = np.meshgrid(x_tmp, y_pos)
                # need to change shape to sth like [2, 100, 100]
                data_tmp = np.zeros([2, x_pos.shape[0], x_pos.shape[1]])
                data_tmp[0,:,:] = x_pos
                data_tmp[1,:,:] = yv
                new_data['pos_data'] = data_tmp
                new_data['pos_names'] = ['x_pos', 'y_pos']
                if vertical_fast is True: # need to transpose the data, as we scan y first
                    data_tmp = np.zeros([2, x_pos.shape[1], x_pos.shape[0]]) # fast scan on y has impact for scalar data
                    data_tmp[1,:,:] = x_pos.T
                    data_tmp[0,:,:] = yv.T
                    new_data['pos_data'] = data_tmp
            else:
                print('x,y positions are not saved.')
        # output to file
        print('Saving data to hdf file.')
        if create_each_det is False:
            create_each_det = False
        else:
            create_each_det = True
        write_db_to_hdf_base(fpath, new_data, num_det=num_det,
                             create_each_det=create_each_det)


def get_data_per_event(n, data, e, det_num):
    db.fill_event(e)
    min_len = e.data['fluor'].shape[0]
    for i in range(det_num):
        data[n, :min_len, :] += e.data['fluor'][:,i,:]


def get_data_parallel(data, elist, det_num):
    num_processors_to_use = multiprocessing.cpu_count()-2

    print('cpu count: {}'.format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    result_pool = [
        pool.apply_async(get_data_per_event, (n, data, e, det_num))
        for n, e in enumerate(elist)]

    results = [r.get() for r in result_pool]

    pool.terminate()
    pool.join()


def get_total_scan_point(hdr):
    """
    Find the how many data points are recorded. This number may not equal to the total number
    defined at the start of the scan due to scan stop or abort.
    """
    evs = get_events(hdr)
    n = 0
    try:
        for e in evs:
            n = n+1
    except IndexError:
        pass
    return n


def make_hdf(start, end=None, fname=None,
             prefix='scan2D_', full_data=True,
             create_each_det=False, save_scalar=True,
             num_end_lines_excluded=None):
    """
    Transfer multiple h5 files.

    Parameters
    ---------
    start : int
        start run id
    end : int, optional
        end run id
    fname : string
        path to save file when start equals to end, in this case only
        one file is transfered.
    prefix : str, optional
        prefix name of the file
    full_data : bool, optional
        save baseline data and all other information if True
    db : databroker
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines. Used at SRX beamline.
    """
    if end is None:
        end = start

    if end == start:
        if fname is None:
            fname = prefix+str(start)+'.h5'
        _make_hdf(fname, start, full_data=full_data,
                  create_each_det=create_each_det,
                  save_scalar=save_scalar,
                  num_end_lines_excluded=num_end_lines_excluded)  # only transfer one file
    else:
        datalist = range(start, end+1)
        for v in datalist:
            filename = prefix+str(v)+'.h5'
            try:
                _make_hdf(filename, v, full_data=full_data,
                          create_each_det=create_each_det,
                          save_scalar=save_scalar,
                          num_end_lines_excluded=num_end_lines_excluded)
                print('{} is created. \n'.format(filename))
            except:
                print('Can not transfer scan {}. \n'.format(v))


def free_memory_from_handler():
    """Quick way to set 3D dataset at handler to None to release memory.
    """
    for h in db.fs._handler_cache.values():
        setattr(h, '_dataset', None)
    print('Memory is released.')


def export_hdf(runid, fname, xrf=False):
    """Wrapper around suitcase.hdf5.export function.
    Use can choose to select xrf data or not.

    Parameters
    ---------
    runid : int
        run id
    fname : str
        output name
    xrf : Bool, optional
        save xrf data or not, default as not saving.
    """
    hdr = db[runid]

    if xrf is False:
        tmp = set()
        for descriptor in hdr.descriptors:
            xs3 = [key for key in descriptor.data_keys.keys() if 'xspress3' in key]
            roi = [key for key in descriptor.data_keys.keys() if 'Det' in key]
            tmp.update(xs3)
            tmp.update(roi)
        fds = sc.filter_fields(hdr, tmp)
        sc.export(hdr, fname, db.mds, fields=fds, use_uid=False)
    else:
        sc.export(hdr, fname, db.mds, use_uid=False)


def export_to_view(fpath, output_name=None, output_folder='', namelist=None):
    """
    Output fitted data to tablet data for visulization.

    Parameters
    ----------
    fpath : str
        input file path, file is pyxrf h5 file
    output_name : str
        output file name
    otuput_folder : str, optional
        default as current working folder
    namelist : list, optional
        list of elemental names
    """
    with h5py.File(fpath, 'r') as f:
        d = f['xrfmap/detsum/xrf_fit'][:]
        d = d.reshape([d.shape[0], -1])
        elementlist = f['xrfmap/detsum/xrf_fit_name'][:]
        elementlist = helper_decode_list(elementlist)

        xy = f['xrfmap/positions/pos'][:]
        xy =  xy.reshape([xy.shape[0], -1])
        xy_name = ['X', 'Y']

        names = xy_name + elementlist
        data = np.concatenate((xy, d), axis=0)

    data_dict = OrderedDict()
    if namelist is None:
        for i, k in enumerate(names):
            if 'Userpeak' in k or 'r2_adjust' in k:
                continue
            data_dict.update({k: data[i,:]})
    else:
        for i, k in enumerate(names):
            if k in namelist or k in xy_name:
                data_dict.update({k: data[i,:]})

    df = pd.DataFrame(data_dict)
    if output_name is None:
        fname = fpath.split(sep_v)[-1]
        output_name = fname.split('.')[0] + '_fit_view.csv'

    outpath = os.path.join(output_folder, output_name)
    print('{} is created.'.format(outpath))
    df.to_csv(outpath, index=False)


def export1d(runid, name=None):
    """
    Export all PVs to a file. Do not talk to filestore.

    Parameters
    ----------
    name : str or optional
        name for the file
    runid : int
        run number
    """
    t = db.get_table(db[runid], fill=False)
    if name is None:
        name = 'scan_'+str(runid)+'.txt'
    t.to_csv(name)


def get_header(fname):
    """
    helper function to extract header in spec file.
    .. warning :: This function works fine for spec file format
    from Canadian light source. Others may need to be tested.

    Parameters
    ----------
    fname : spec file name
    """
    mydata = []
    with open(fname, 'r') as f:
        for v in f:   # iterate the file
            mydata.append(v)
            _sign = '#'
            _sign = _sign.encode('utf-8')
            if _sign not in v:
                break
    header_line = mydata[-2]  # last line is space
    n = [v.strip() for v in header_line[1:].split('\t') if v.strip()!='']
    return n


def combine_data_to_recon(element_list, datalist, working_dir, norm=True,
                          file_prefix='scan2D_', ic_name='sclr1_ch4',
                          expand_r=2, internal_path='xrfmap/detsum'):
    """
    Combine 2D data to 3D array for reconstruction.

    Parameters
    ----------
    element_list : list
        list of elements
    datalist : list
        list of run number
    working_dir : str
    norm : bool, optional
        normalization or not
    file_prefix : str, optional
        prefix name for h5 file
    ic_name : str
        ion chamber name for normalization
    expand_r: int
        expand initial array to a larger size to include each 2D image easily,
        as each 2D image may have different size. Crop the 3D array back to a proper size in the end.
    internal_path : str, optional
        inside path to get fitting data in h5 file

    Returns
    -------
    dict of 3d array with each array's shape like [num_sequences, num_row, num_col]
    """
    element3d = {}
    for element_name in element_list:
        element3d[element_name] = None

    max_h = 0
    max_v = 0
    for i, v in enumerate(datalist):
        filename = file_prefix+str(v)+'.h5'
        filepath = os.path.join(working_dir, filename)
        with h5py.File(filepath, 'r+') as f:
            dataset = f[internal_path]
            try:
                data_all = dataset['xrf_fit'].value
                data_name = dataset['xrf_fit_name'].value
                data_name = helper_decode_list(data_name)
            except KeyError:
                print('Need to do fitting first.')
            scaler_dataset = f['xrfmap/scalers']
            scaler_v = scaler_dataset['val'].value
            scaler_n = scaler_dataset['name'].value
            scaler_n = helper_decode_list(scaler_n)

        data_dict = {}
        for name_i, name_v in enumerate(data_name):
            data_dict[name_v] = data_all[name_i, :, :]
        if norm is True:
            scaler_dict = {}
            for s_i, s_v in enumerate(scaler_n):
                scaler_dict[s_v] = scaler_v[:, :, s_i]

        for element_name in element_list:
            data = data_dict[element_name]
            if norm is True:
                normv = scaler_dict[ic_name]
                data = data/normv
            if element3d[element_name] is None:
                element3d[element_name] = np.zeros([len(datalist),
						    data.shape[0]*expand_r,
						    data.shape[1]*expand_r])
            element3d[element_name][i, :data.shape[0], :data.shape[1]] = data

        max_h = max(max_h, data.shape[0])
        max_v = max(max_v, data.shape[1])

    for k, v in element3d.items():
        element3d[k] = v[:,:max_h, :max_v]
    return element3d


def h5file_for_recon(element_dict, angle, runid=None, filename=None):
    """
    Save fitted 3d elemental data into h5 file for reconstruction use.

    Parameters
    ----------
    element_dict : dict
        elements 3d data after normalization
    angle : list
        angle information
    runid : list or optional
        run ID
    filename : str
    """

    if filename is None:
        filename = 'xrf3d.h5'
    with h5py.File(filename) as f:
        d_group = f.create_group('element_data')
        for k, v in element_dict.items():
            sub_g = d_group.create_group(k)
            sub_g.create_dataset('data', data=np.asarray(v),
                                 compression='gzip')
            sub_g.attrs['comments'] = 'normalized fluorescence data for {}'.format(k)
        angle_g = f.create_group('angle')
        angle_g.create_dataset('data', data=np.asarray(angle))
        angle_g.attrs['comments'] = 'angle information'
        if runid is not None:
            runid_g = f.create_group('runid')
            runid_g.create_dataset('data', data=np.asarray(runid))
            runid_g.attrs['comments'] = 'run id information'


def create_movie(data, fname='demo.mp4', dpi=100, cmap='jet',
                 clim=None, fig_size=(6,8), fps=20, data_power=1, angle=None, runid=None):
    """
    Transfer 3d array into a movie.

    Parameters
    ----------
    data : 3d array
        data shape is [num_sequences, num_row, num_col]
    fname : string, optional
        name to save movie
    dpi : int, optional
        resolution of the movie
    cmap : string, optional
        color format
    clim : list, tuple, optional
        [low, high] value to define plotting range
    fig_size : list, tuple, optional
        size (horizontal size, vertical size) of each plot
    fps : int, optional
        frame per second
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(np.zeros([data.shape[1], data.shape[2]]),
                   cmap=cmap, interpolation='nearest')

    fig.set_size_inches(fig_size)
    fig.tight_layout()

    def update_img(n):
        tmp = data[n,:,:]
        im.set_data(tmp**data_power)
        if clim is not None:
            im.set_clim(clim)
        else:
            im.set_clim([0,np.max(data[n,:,:])])
        figname = ''
        if runid is not None:
            figname = 'runid: {} '.format(runid[n])
        if angle is not None:
            figname += 'angle: {}'.format(angle[n])
        #if len(figname) != 0:
        #    im.ax.set_title(figname)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,data.shape[0],interval=30)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(fname,writer=writer,dpi=dpi)


def print_image(fig):
    """
    Print function used at beamline only.
    """
    if db is not None:
        hdr = db[-1]
        if hdr.start.beamline_id == 'HXN':
            current_dir = os.path.dirname(os.path.realpath(__file__))
            config_file = 'hxn_pv_config.json'
            config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
            with open(config_path, 'r') as json_data:
                config_data = json.load(json_data)
            fpath = config_data.get('print_path', None)
            if fpath is None:
                fpath = '/home/xf03id/Desktop/temp.png'
            fig.savefig(fpath,  bbox_inches='tight', pad_inches=4)
            os.system(config_data['print_command'])
        else:
            print('Printer is not set up yet.')


def spec_to_hdf(wd, spec_file, spectrum_file, output_file, img_shape,
                ic_name=None, x_name=None, y_name=None):
    """
    Transform spec data to hdf file pyxrf can take. Using this function, users need to
    have two input files ready, sepc_file and spectrum_file, with explanation as below.

    .. warning :: This function should be better defined to take care spec file in general.
    The work in suitcase should also be considered. This function works fine for spec file format
    from Canadian light source. Others may need to be tested.

    Parameters
    ----------
    wd : str
        working directory for spec file, and created hdf
    spec_file : str
        spec txt data file
    spectrum_file : str
        fluorescence spectrum data file
    output_file : str
        the output h5 file for pyxrf
    img_shape : list or array
        the shape of two D scan, [num of row, num of column]
    ic_name : str
        the name of ion chamber for normalization, listed in spec file
    x_name : str
        x position name, listed in spec file
    y_name : str
        y position name, listed in spec file
    """
    # read scaler data from spec file
    spec_path = os.path.join(wd, spec_file)
    h = get_header(spec_path)
    spec_data = pd.read_csv(spec_path, names=h, sep='\t', comment='#', index_col=False)

    if ic_name is not None:
        scaler_name = [str(ic_name)]
        scaler_val = spec_data[scaler_name].values
        scaler_val = scaler_val.reshape(img_shape)
        scaler_data = np.zeros([img_shape[0], img_shape[1], 1])
        scaler_data[:,:,0] = scaler_val

    if x_name is not None and y_name is not None:
        xy_data = np.zeros([2, img_shape[0], img_shape[1]])
        xy_data[0, :, :] = spec_data[x_name].values.reshape(img_shape)
        xy_data[1, :, :] = spec_data[y_name].values.reshape(img_shape)
        xy_name = ['x_pos', 'y_pos']

    spectrum_path = os.path.join(wd, spectrum_file)
    sum_data0 = np.loadtxt(spectrum_path)
    sum_data = np.reshape(sum_data0, [sum_data0.shape[0], img_shape[0], img_shape[1]])
    sum_data = np.transpose(sum_data, axes=(1,2,0))

    interpath = 'xrfmap'

    fpath = os.path.join(wd, output_file)
    with h5py.File(fpath) as f:
        dataGrp = f.create_group(interpath+'/detsum')
        ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
        ds_data.attrs['comments'] = 'Experimental data from channel sum'

        if ic_name is not None:
            dataGrp = f.create_group(interpath+'/scalers')
            dataGrp.create_dataset('name', data=helper_encode_list(scaler_name))
            dataGrp.create_dataset('val', data=scaler_data)

        if x_name is not None and y_name is not None:
            dataGrp = f.create_group(interpath+'/positions')
            dataGrp.create_dataset('name', data=helper_encode_list(xy_name))
            dataGrp.create_dataset('pos', data=xy_data)
