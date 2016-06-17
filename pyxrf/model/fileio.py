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
                        print_function)

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

from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum, Float, Bool

import logging
logger = logging.getLogger(__name__)

try:
    from databroker.databroker import DataBroker as db, get_table, get_events
    # registers a filestore handler for the XSPRESS3 detector
except (ImportError, KeyError) as e:
    db = None
    logger.error('databroker is not available: %s', e)

try:
    # registers a filestore handler for the XSPRESS3 detector
    from hxntools import handlers as hxn_handlers
    from hxntools.handlers import register
    register()
except ImportError as e:
    logger.error('hxntools is not available from old version: %s', e)

try:
    import vortex_handler
except ImportError as e:
    logger.error('handler is not loaded.')


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
        #with self.suppress_notifications():
        #    self.working_directory = working_directory

    # @observe('working_directory')
    # def working_directory_changed(self, changed):
    #     # make sure that the output directory stays in sync with working
    #     # directory changes
    #     self.output_directory = self.working_directory

    @observe('file_name')
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
        self.file_channel_list = self.data_sets.keys()

    @observe('runid')
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

    @observe('file_opt')
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

    @observe('plot_index')
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
    d = get_table(db[runid], fields=fields)
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

    exp_keys = data.keys()

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


# def read_hdf_HXN(working_directory,
#                  file_names, channel_num=4):
#     """
#     Data IO for HXN temporary datasets. This might be changed later.
#
#     Parameters
#     ----------
#     working_directory : str
#         path folder
#     file_names : list
#         list of chosen files
#     channel_num : int, optional
#         detector channel number
#
#     Returns
#     -------
#     data_dict : dict
#         with fitting data
#     data_sets : dict
#         data from each channel and channel summed
#     """
#     data_dict = OrderedDict()
#     data_sets = OrderedDict()
#
#     # cut off bad point on the last position of the spectrum
#     bad_point_cut = 1
#
#     for fname in file_names:
#         try:
#             file_path = os.path.join(working_directory, fname)
#             f = h5py.File(file_path, 'r+')
#             data = f['entry/instrument']
#
#             fname = fname.split('.')[0]
#
#             # for 2D MAP???
#             data_dict[fname] = data
#
#             # data from channel summed
#             exp_data = np.asarray(data['detector/data'])
#             logger.info('File : {} with total counts {}'.format(fname,
#                                                                 np.sum(exp_data)))
#             exp_data = exp_data[:, :, :-bad_point_cut]
#             DS = DataSelection(filename=fname,
#                                raw_data=exp_data)
#             data_sets.update({fname: DS})
#
#             # data from each channel
#             for i in range(channel_num):
#                 file_channel = fname+'_channel_'+str(i+1)
#                 exp_data_new = np.reshape(exp_data[0, i, :],
#                                           [1, 1, exp_data[0, i, :].size])
#                 DS = DataSelection(filename=file_channel,
#                                    raw_data=exp_data_new)
#                 data_sets.update({file_channel: DS})
#
#         except ValueError:
#             continue
#     return data_dict, data_sets


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


# def read_log_file_srx(fpath, name='ch 2'):
#     """
#     Read given column value from log file.
#
#     Parameters
#     ----------
#     fpath : str
#         path to log file.
#     name : str
#         name of a given column
#
#     Returns
#     -------
#     1d array:
#         selected ic value.
#     """
#     line_num = 7
#     base_val = 8.5*1e-10
#
#     with open(fpath, 'r') as f:
#         lines = f.readlines()
#         col_names = lines[line_num].split('\t')
#         col_names = [v for v in col_names if len(v) != 0]
#
#     df = pd.read_csv(fpath,
#                      skiprows=line_num+1, skip_footer=1, sep='\s+', header=None)
#     df.columns = col_names
#     i0 = np.abs(np.array(df[name]) - base_val)
#     return i0


# def write_xspress3_data_to_hdf(fpath, data_dict):
#     """
#     Assume data is obained from databroker, and save the data to hdf file.
#
#     Parameters
#     ----------
#     fpath: str
#         path to save hdf file
#     data : dict
#         data_dict with data from each channel
#     """
#
#     interpath = 'xrfmap'
#     sum_data = None
#     channel_list = [k for k in six.iterkeys(data_dict) if 'channel' in k]
#     pos_names = [k for k in six.iterkeys(data_dict) if 'pos' in k]
#     scaler_names = [k for k in six.iterkeys(data_dict) if 'scaler' in k]
#
#     with h5py.File(fpath, 'a') as f:
#
#         for n in range(len(channel_list)):
#             detname = 'det'+str(n+1)
#             try:
#                 dataGrp = f.create_group(interpath+'/'+detname)
#             except ValueError:
#                 dataGrp = f[interpath+'/'+detname]
#
#             if sum_data is None:
#                 sum_data = data_dict[channel_list[n]]
#             else:
#                 sum_data += data_dict[channel_list[n]]
#
#             if 'counts' in dataGrp:
#                 del dataGrp['counts']
#             ds_data = dataGrp.create_dataset('counts', data=data_dict[channel_list[n]])
#             ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)
#
#         # summed data
#         try:
#             dataGrp = f.create_group(interpath+'/detsum')
#         except ValueError:
#             dataGrp = f[interpath+'/detsum']
#
#         if 'counts' in dataGrp:
#             del dataGrp['counts']
#         ds_data = dataGrp.create_dataset('counts', data=sum_data)
#         ds_data.attrs['comments'] = 'Experimental data from channel sum'
#
#         data_shape = sum_data.shape
#
#         # position data
#         try:
#             dataGrp = f.create_group(interpath+'/positions')
#         except ValueError:
#             dataGrp = f[interpath+'/positions']
#
#         pos_data = []
#         for k in pos_names:
#             pos_data.append(data_dict[k])
#         pos_data = np.array(pos_data)
#
#         if 'pos' in dataGrp:
#             del dataGrp['pos']
#
#         if 'name' in dataGrp:
#             del dataGrp['name']
#         dataGrp.create_dataset('name', data=pos_names)
#         dataGrp.create_dataset('pos', data=pos_data)
#
#         # scaler data
#         scaler_data = np.ones([data_shape[0], data_shape[1], len(scaler_names)])
#         for i in np.arange(len(scaler_names)):
#             scaler_data[:, :, i] = data_dict[scaler_names[i]]
#         scaler_data = np.array(scaler_data)
#         print('shape for scaler: {}'.format(scaler_data.shape))
#
#         try:
#             dataGrp = f.create_group(interpath+'/scalers')
#         except ValueError:
#             dataGrp = f[interpath+'/scalers']
#
#         if 'val' in dataGrp:
#             del dataGrp['val']
#
#         if 'name' in dataGrp:
#             del dataGrp['name']
#         dataGrp.create_dataset('name', data=scaler_names)
#         dataGrp.create_dataset('val', data=scaler_data)


# def xspress3_data_to_hdf(fpath_hdf5, fpath_log, fpath_out):
#     """
#     Assume data is obained from databroker, and save the data to hdf file.
#
#     Parameters
#     ----------
#     fpath_hdf5: str
#         path to raw hdf file
#     fpath_log: str
#         path to log file
#     fpath_out: str
#         path to save hdf file
#     """
#     data_dict = read_xspress3_data(fpath_hdf5)
#
#     data_ic = read_log_file_srx(fpath_log)
#     shapev = None
#
#     for k, v in six.iteritems(data_dict):
#         data_dict[k] = flip_data(data_dict[k])
#         if shapev is None:
#             shapev = data_dict[k].shape
#
#     data_ic = data_ic.reshape([shapev[0], shapev[1]])
#     print('data ic shape {}'.format(data_ic.shape))
#     data_ic = flip_data(data_ic)
#
#     data_dict['scalers_ch2'] = data_ic
#
#     write_xspress3_data_to_hdf(fpath_out, data_dict)


def output_data(fpath, output_folder, file_format='tiff'):
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
    """

    f = h5py.File(fpath, 'r')

    detlist = f['xrfmap'].keys()
    fit_output = {}

    for detname in detlist:
        # fitted data
        if 'xrf_fit' in f['xrfmap/'+detname]:
            fit_data = f['xrfmap/'+detname+'/xrf_fit']
            fit_name = f['xrfmap/'+detname+'/xrf_fit_name']

            for i in np.arange(len(fit_name)):
                fit_output[detname+'_'+fit_name[i]] = fit_data[i, :, :]
        # fitted error
        if 'xrf_fit_error' in f['xrfmap/'+detname]:
            error_data = f['xrfmap/'+detname+'/xrf_fit_error']
            error_name = f['xrfmap/'+detname+'/xrf_fit_error_name']

            for i in np.arange(len(error_name)):
                fit_output[detname+'_'+error_name[i]+'_error'] = error_data[i, :, :]

    # ic data
    if 'scalers' in f['xrfmap']:
        ic_data = f['xrfmap/scalers/val']
        ic_name = f['xrfmap/scalers/name']
        for i in np.arange(len(ic_name)):
            fit_output[ic_name[i]] = ic_data[:, :, i]

    # position data
    if 'positions' in f['xrfmap']:
        pos_name = f['xrfmap/positions/name']
        for i, n in enumerate(pos_name):
            fit_output[n] = f['xrfmap/positions/pos'].value[i, :]

    #save data
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    for k, v in six.iteritems(fit_output):
        if file_format == 'tiff':
            fname = os.path.join(output_folder, k+'.tiff')
            sio.imsave(fname, v.astype(np.float32))
        elif file_format == 'txt':
            fname = os.path.join(output_folder, k+'.txt')
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
                temp[n] = data['scalers/val'].value[:, :, i]
            img_dict[fname+'_scaler'] = temp

        # find total channel:
        channel_num = 0
        for v in data.keys():
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
            except IndexError:
                logger.info('No fitting data is loaded for channel summed data.')

        if 'positions' in data:
            pos_name = data['positions/name']
            temp = {}
            for i, n in enumerate(pos_name):
                #
                # !!! This should be cleaned up later.
                # !!! This is due to the messy in write_db_to_hdf function
                #
                if i==0:
                    temp[n] = np.fliplr(data['positions/pos'].value[i, :])
                else:
                    temp[n] = data['positions/pos'].value[i, :]
                    #temp[n] = np.flipud(data['positions/pos'].value[i, :])

            img_dict['positions'] = temp

    return img_dict, data_sets


def read_MAPS(working_directory,
              file_name, channel_num=1):
    data_dict = OrderedDict()
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 0

    fit_val = None

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
            pass

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
    for i in range(len(namelist)):
        data_temp.update({namelist[i]: data[i, :, :]})
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
            fit_key0, = [v for v in out.keys() if key_name in v]
            fit_key, = [v for v in img.keys() if key_name in v]
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
    config_path = '/'.join(current_dir.split('/')[:-2]+['configs', config_file])
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)

    print('Saving all the data into one hdf file.')
    write_db_to_hdf(fpath, result,
                    img_shape,
                    det_list=config_data['xrf_detector'],
                    #roi_dict=roi_dict,
                    pos_list=('x_pos', 'y_pos'),
                    scaler_list=config_data['scaler_list'],
                    base_val=config_data['base_value'])  #base value shift for ic


    fitkey, = [v for v in out.keys() if 'fit' in v]
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
    filelist = [name.split('/')[-1] for name in pathlist]
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
        namelist.append(str(k))
        data.append(v)

    if data_saveas in dataGrp:
        del dataGrp[data_saveas]

    data = np.array(data)
    ds_data = dataGrp.create_dataset(data_saveas, data=data)
    ds_data.attrs['comments'] = ' '

    if dataname_saveas in dataGrp:
        del dataGrp[dataname_saveas]

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


def write_db_to_hdf(fpath, data, datashape, get_roi_sum_sign=False,
                    det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
                    roi_dict={'Pt_9300_9600': ['Ch1 [9300:9600]', 'Ch2 [9300:9600]', 'Ch3 [9300:9600]']},
                    pos_list=('zpssx[um]', 'zpssy[um]'),
                    scaler_list=('sclr1_ch3', 'sclr1_ch4'),
                    fly_type=None, subscan_dims=None, base_val=None):
    """
    Assume data is obained from databroker, and save the data to hdf file.
    This function can handle stopped/aborted scans.

    !!! This function needs to be rewrite during beam shutdown. It is
    better to pass 2D/3D array instead of dataframe or dict.

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
    roi_dict : dict
        dict of roi pv names
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    """
    interpath = 'xrfmap'
    f = h5py.File(fpath, 'a')

    sum_data = None
    new_v_shape = datashape[0]  # to be updated if scan is not completed

    for n in range(len(det_list)):
        c_name = det_list[n]
        if c_name in data:
            detname = 'det'+str(n+1)
            try:
                dataGrp = f.create_group(interpath+'/'+detname)
            except ValueError:
                dataGrp = f[interpath+'/'+detname]

            logger.info('read data from %s' % c_name)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]
            #new_data = np.zeros([1, len(channel_data), len(channel_data[1])])
            new_data = np.zeros([1, new_v_shape*datashape[1], len(channel_data[1])])

            for i in range(new_v_shape*datashape[1]):
                #channel_data[i+1][pd.isnull(channel_data[i+1])] = 0
                new_data[0, i, :] = channel_data[i+1]

            new_data = new_data.reshape([new_v_shape, datashape[1],
                                         len(channel_data[1])])
            if fly_type in ('pyramid',):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)

            if sum_data is None:
                sum_data = new_data
            else:
                sum_data += new_data

            if 'counts' in dataGrp:
                del dataGrp['counts']
            ds_data = dataGrp.create_dataset('counts', data=new_data, compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)

    # summed data
    try:
        dataGrp = f.create_group(interpath+'/detsum')
    except ValueError:
        dataGrp = f[interpath+'/detsum']

    if sum_data is not None:
        sum_data = sum_data.reshape([new_v_shape, datashape[1],
                                     len(channel_data[1])])

        if 'counts' in dataGrp:
            del dataGrp['counts']
        ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
        ds_data.attrs['comments'] = 'Experimental data from channel sum'

    # position data
    try:
        dataGrp = f.create_group(interpath+'/positions')
    except ValueError:
        dataGrp = f[interpath+'/positions']

    pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                 datashape)

    for i in range(len(pos_names)):
        if 'x' in pos_names[i]:
            pos_names[i] = 'x_pos'
        elif 'y' in pos_names[i]:
            pos_names[i] = 'y_pos'

    if 'pos' in dataGrp:
        del dataGrp['pos']

    if 'name' in dataGrp:
        del dataGrp['name']

    # need to change shape to sth like [2, 100, 100]
    #
    # this needs to be corrected later. It is confusing to Reorganize position this way.
    #
    pos_data = pos_data.transpose()
    data_temp = np.zeros([pos_data.shape[0], pos_data.shape[2], pos_data.shape[1]])
    for i in range(pos_data.shape[0]):
        data_temp[i,:,:] = np.rot90(pos_data[i,:,:], k=3)

    if fly_type in ('pyramid',):
        for i in range(data_temp.shape[0]):
            # flip position the same as data flip on det counts
            data_temp[i,:,:] = flip_data(data_temp[i,:,:], subscan_dims=subscan_dims)

    dataGrp.create_dataset('name', data=pos_names)
    dataGrp.create_dataset('pos', data=data_temp[:,:new_v_shape,:])

    # scaler data
    try:
        dataGrp = f.create_group(interpath+'/scalers')
    except ValueError:
        dataGrp = f[interpath+'/scalers']

    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                       datashape)

    if fly_type in ('pyramid',):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)

    if 'val' in dataGrp:
        del dataGrp['val']

    if 'name' in dataGrp:
        del dataGrp['name']
    dataGrp.create_dataset('name', data=scaler_names)

    if base_val is not None:  # base line shift for detector, for SRX
        scaler_data = np.abs(scaler_data - base_val)

    dataGrp.create_dataset('val', data=scaler_data[:new_v_shape,:])

    # roi sum
    if get_roi_sum_sign:
        try:
            dataGrp = f.create_group(interpath+'/roimap')
        except ValueError:
            dataGrp = f[interpath+'/roimap']

        roi_data_all = np.zeros([datashape[0], datashape[1], len(roi_dict)])
        roi_name_list = []

        roi_data_all_ch = None
        roi_name_all_ch = []

        for i, (k, roi_list) in enumerate(six.iteritems(roi_dict)):
            count = 0
            if i == 0 and roi_data_all_ch is None:
                roi_data_all_ch = np.zeros([datashape[0], datashape[1],
                                            len(roi_dict)*len(roi_list)])

            roi_names, roi_data = get_name_value_from_db(roi_list, data,
                                                         datashape)
            roi_name_list.append(str(k))
            roi_data_all[:, :, i] = np.sum(roi_data, axis=2)

            for n in range(len(roi_list)):
                roi_name_all_ch.append(str(roi_list[n]))
                roi_data_all_ch[:, :, count] = roi_data[:, :, n]
                count += 1

        # summed data from each channel
        if 'sum_name' in dataGrp:
            del dataGrp['sum_name']

        if 'sum_raw' in dataGrp:
            del dataGrp['sum_raw']
        dataGrp.create_dataset('sum_name', data=roi_name_list)
        dataGrp.create_dataset('sum_raw', data=roi_data_all)

        # data from each channel
        if 'det_name' in dataGrp:
            del dataGrp['det_name']

        if 'det_raw' in dataGrp:
            del dataGrp['det_raw']
        dataGrp.create_dataset('det_name', data=roi_name_all_ch)
        dataGrp.create_dataset('det_raw', data=roi_data_all_ch)

        # if 'det_name' in dataGrp:
        #     del dataGrp['det_name']
        #
        # if 'det_raw' in dataGrp:
        #     del dataGrp['det_raw']
        #
        # dataGrp.create_dataset('det_raw', data=roi_data)
        # dataGrp.create_dataset('det_name', data=roi_names)

        logger.info('roi names: {}'.format(roi_names))

    f.close()


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


# def db_to_hdf(fpath, runid,
#               datashape,
#               det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
#               roi_dict={'Pt_9300_9600': ['Ch1 [9300:9600]', 'Ch2 [9300:9600]', 'Ch3 [9300:9600]']},
#               pos_list=('ssx[um]', 'ssy[um]'),
#               scaler_list=('sclr1_ch2', 'sclr1_ch3', 'sclr1_ch8')):
#     """
#     Read data from databroker, and save the data to hdf file.
#
#     .. note:: Requires the databroker package from NSLS2
#     .. note:: This should probably be moved to suitcase!
#
#     Parameters
#     ----------
#     fpath: str
#         path to save hdf file
#     runid : int
#         id number for given run
#     datashape : tuple or list
#         shape of two D image
#     det_list : list, tuple or optional
#         list of detector channels
#     roi_list : list, tuple, optional
#         list of roi pv names
#     pos_list : list, tuple or optional
#         list of pos pv
#     scaler_list : list, tuple, optional
#         list of scaler pv
#     """
#     data = fetch_data_from_db(runid)
#     write_db_to_hdf(fpath, data,
#                     datashape, det_list=det_list,
#                     roi_dict=roi_dict,
#                     pos_list=pos_list,
#                     scaler_list=scaler_list)


# def get_roi_keys(all_keys, beamline='HXN'):
#     """
#     Get roi dict in terms of beamlines.
#     """
#     if beamline == 'HXN':
#         return get_roi_keys_hxn(all_keys)
#
#
# def get_roi_keys_hxn(all_keys):
#     """
#     Reorganize detector names of roi detector.
#
#     Parameters
#     ----------
#     all_keys : list
#         pv names
#
#     Returns
#     -------
#     dict:
#         format as {'Ch0_sum': ['ch0_1', 'ch0_2', 'ch0_3']}
#     """
#     Ch1_list = sorted([v for v in all_keys if 'Det1' in v and 'xspress' not in v])
#     Ch2_list = sorted([v for v in all_keys if 'Det2' in v and 'xspress' not in v])
#     Ch3_list = sorted([v for v in all_keys if 'Det3' in v and 'xspress' not in v])
#
#     Ch_dict = {}
#     for i in range(len(Ch1_list)):
#         k = Ch1_list[i].replace('1', '_sum')
#         val = [Ch1_list[i], Ch2_list[i], Ch3_list[i]]
#         Ch_dict[k] = val
#     return Ch_dict


# def get_scaler_list_hxn(all_keys):
#     return [v for v in all_keys if 'sclr1_' in v]


def add_extral_fields_hdf(fpath, config_path):
    """Add extral information from databroker into h5 file.
    In progress.
    """
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)
    print(config_data.keys())

    interpath = 'xrfmap'
    with h5py.File(fpath, 'a') as f:

        try:
            dg = f.create_group(interpath+'/config')
        except ValueError:
            dg = f[interpath+'/config']

        sub_name_dict = {'upstream': 'upstream_name_list',
                         'microscope': 'microscope_name_list'}
        for sub_name, v in sub_name_dict.items():
            try:
                dg_sub = dg.create_group(sub_name)
            except ValueError:
                dg_sub = dg[sub_name]

            data_name = [str(val) for val in config_data[v]]

            # need to read data from PV here
            data_val = np.zeros(len(data_name))

            try:
                dg_sub.create_dataset('name', data=data_name)
            except RuntimeError:
                try:
                    dg_sub['name'][:] = data_name
                except TypeError:  # when the length of data items changes
                    dg_sub.__delitem__('name')
                    dg_sub.create_dataset('name', data=data_name)

            try:
                dg_sub.create_dataset('value', data=data_val)
            except RuntimeError:
                try:
                    dg_sub['value'][:] = data_val
                except TypeError:
                    dg_sub.__delitem__('value')
                    dg_sub.create_dataset('value', data=data_val)

        # sub_name = 'microscope'
        # try:
        #     dg_m = dg.create_group(sub_name)
        # except ValueError:
        #     dg_m = dg[sub_name]


def _make_hdf(fpath, runid):
    """
    Save the data from databroker to hdf file.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    fpath: str
        path to save hdf file
    runid : int
        id number for given run
    """
    hdr = db[runid]
    print('Loading data from database.')

    if hdr.start.beamline_id == 'HXN':
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

        fields = ['xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3',
                  'sclr1_ch3', 'sclr1_ch4'] + pos_list
        data = get_table(hdr, fields=fields, fill=True)

        print('Saving data to hdf file.')
        write_db_to_hdf(fpath, data, datashape,
                        pos_list=pos_list,
                        fly_type=fly_type, subscan_dims=subscan_dims)
        print('Done!')

    elif hdr.start.beamline_id == 'xf05id':
        start_doc = hdr['start']
        datashape = start_doc['shape']   # vertical first then horizontal
        fly_type = None

        # issues on py3
        snake_scan = start_doc.get('snaking')
    	if snake_scan[1] == True:
    	    fly_type = 'pyramid'

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = 'srx_pv_config.json'
        config_path = '/'.join(current_dir.split('/')[:-2]+['configs', config_file])
        with open(config_path, 'r') as json_data:
            config_data = json.load(json_data)

        try:
            data = get_table(hdr)
        except IndexError:
            # !!! this part needs to rewrite during shutdown. it is better to pass
            # !!! array data to write_db_to_hdf, instead of dataframe or dict
            # !!! this is not optimized, to be update soon!

            cut_num = 2
            spectrum_len = 4096
            total_len = datashape[0]*datashape[1]

            evs, _ = zip(*zip(get_events(hdr), range(total_len-cut_num)))

            namelist = config_data['xrf_detector'] +config_data['pos_list'] +config_data['scaler_list']

            dictv = {v:[] for v in namelist}

            for e in evs:
                for k,v in six.iteritems(dictv):
                    dictv[k].append(e.data[k])

            data = pd.DataFrame(dictv, index=np.arange(1, total_len-cut_num+1)) # need to start with 1

        print('Saving data to hdf file.')
        write_db_to_hdf(fpath, data,
                        datashape,
                        det_list=config_data['xrf_detector'],
                        #roi_dict=roi_dict,
                        pos_list=hdr.start.motors,
                        scaler_list=config_data['scaler_list'],
                        fly_type=fly_type,
                        base_val=config_data['base_value'])  #base value shift for ic
        print('Done!')

    else:
        print("Databroker is not setup for this beamline")


def make_hdf(start, end=None, fname=None, prefix='scan2D_'):
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
    """
    if end is None:
        end = start

    if end == start:
        if fname is None:
            fname = prefix+str(start)+'.h5'
        _make_hdf(fname, start)  # only transfer one file
    else:
        datalist = range(start, end+1)
        for v in datalist:
            filename = prefix+str(v)+'.h5'
            try:
                _make_hdf(filename, v)
                print('{} is created. \n'.format(filename))
            except:
                print('Can not transfer scan {}. \n'.format(v))


def export1d(runid):
    """
    Export all PVs to a file. Do not talk to filestore.

    Parameters
    ----------
    runid : int
        run number
    """
    t = get_table(db[runid], fill=False)
    name = 'scan_'+str(runid)+'.txt'
    t.to_csv(name)
