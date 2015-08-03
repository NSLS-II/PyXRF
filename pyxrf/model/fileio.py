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
import pandas as pd
import json

from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum, Float

import logging
logger = logging.getLogger(__name__)

try:
    from dataportal import DataBroker as db
    from dataportal import StepScan as ss
    from dataportal import DataMuxer as dm
    import hxntools.detectors
except ImportError, e:
    logger.warning('Modules not available: %s' % (e))


class FileIOModel(Atom):
    """
    This class focuses on file input and output.

    Attributes
    ----------
    working_directory : str
        current working path
    working_directory : str
        define where output files are saved
    file_names : list
        list of loaded files
    load_status : str
        Description of file loading status
    data_sets : dict
        dict of experiment data, 3D array
    img_dict : dict
        Dict of 2D arrays, such as 2D roi pv or fitted data
    """
    working_directory = Str()
    output_directory = Str()
    file_names = List()
    file_path = Str()
    #data = Typed(np.ndarray)
    load_status = Str()
    #data_dict = Dict()
    data_sets = Typed(OrderedDict)
    img_dict = Dict()

    file_channel_list = List()

    runid = Int(-1)
    h_num = Int(1)
    v_num = Int(1)
    fname_from_db = Str()

    def __init__(self, **kwargs):
        self.working_directory = kwargs['working_directory']
        self.output_directory = kwargs['output_directory']
        #with self.suppress_notifications():
        #    self.working_directory = working_directory

    @observe('working_directory')
    def working_directory_changed(self, changed):
        # make sure that the output directory stays in sync with working
        # directory changes
        self.output_directory = self.working_directory

    @observe('file_names')
    def update_more_data(self, change):
        self.file_channel_list = []
        self.file_names.sort()
        logger.info('Files are loaded: %s' % (self.file_names))

        self.img_dict, self.data_sets = file_handler(self.working_directory,
                                                     self.file_names)

        self.file_channel_list = self.data_sets.keys()

    @observe('runid')
    def _update_fname(self, change):
        self.fname_from_db = 'scan_'+str(self.runid)+'.h5'

    def load_data_runid(self):
        """
        Load data according to runID number.
        """
        if self.h_num != 0 and self.v_num != 0:
            datashape = [self.v_num, self.h_num]

        fname = self.fname_from_db
        fpath = os.path.join(self.working_directory, fname)
        config_file = os.path.join(self.working_directory, 'pv_config.json')
        db_to_hdf_config(fpath, self.runid,
                         datashape, config_file)
        self.file_names = [fname]


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
        else:
            return np.sum(self.data[self.pos1[0]:self.pos2[0], self.pos1[1]:self.pos2[1], :],
                          axis=(0, 1))


def file_handler(working_directory, file_names):
    # be alter: to be update, temporary use!!!
    if '.h5' in file_names[0]:
        #logger.info('Load APS 13IDE data format.')
        return read_hdf_APS(working_directory, file_names)
    elif 'bnp' in file_names[0]:
        logger.info('Load APS 2IDE data format.')
        read_MAPS(working_directory, file_names)
    elif '.npy' in file_names[0]:
        # temporary use
        read_numpy_data(working_directory, file_names)
    else:
        read_hdf_HXN(working_directory, file_names)


def fetch_data_from_db(runid):
    """
    Read data from database.

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
    headers = db.find_headers(scan_id=runid)
    head_list = sorted(headers, key=lambda x: x.start_time)
    hdr = head_list[-1]
    # events = db.fetch_events(hdr, fill=False)
    # num_events = len(list(events))
    # print('%s events found' % num_events)
    ev = db.fetch_events(hdr)

    events = []
    for idx, event in enumerate(ev):
        if idx % 1000 == 0:
            print('event %s loaded' % (idx+1))
        events.append(event)

    muxer = dm.from_events(events)
    data = muxer.to_sparse_dataframe()
    return data


def read_runid(runid, c_list, dshape=None):
    """
    Read data from databroker.

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
        print(c_name)
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
            print(v)
            # clean up nan data, should be done in lower level
            data[v][pd.isnull(data[v])] = 0
            pv_data = np.array(data[v])
            temp[v] = pv_data.reshape(dshape)
    data_dict['Run'+str(runid)+'_roi'] = temp

    return data_dict, data_sets


def read_hdf_HXN(working_directory,
                 file_names, channel_num=4):
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
    data_sets : dict
        data from each channel and channel summed
    """
    data_dict = OrderedDict()
    data_sets = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 1

    for fname in file_names:
        try:
            file_path = os.path.join(working_directory, fname)
            f = h5py.File(file_path, 'r+')
            data = f['entry/instrument']

            fname = fname.split('.')[0]

            # for 2D MAP???
            data_dict[fname] = data

            # data from channel summed
            exp_data = np.asarray(data['detector/data'])
            logger.info('File : {} with total counts {}'.format(fname,
                                                                np.sum(exp_data)))
            exp_data = exp_data[:, :, :-bad_point_cut]
            DS = DataSelection(filename=fname,
                               raw_data=exp_data)
            data_sets.update({fname: DS})

            # data from each channel
            for i in range(channel_num):
                file_channel = fname+'_channel_'+str(i+1)
                exp_data_new = np.reshape(exp_data[0, i, :],
                                          [1, 1, exp_data[0, i, :].size])
                DS = DataSelection(filename=file_channel,
                                   raw_data=exp_data_new)
                data_sets.update({file_channel: DS})

        except ValueError:
            continue
    return data_dict, data_sets


def read_hdf_APS(working_directory,
                 file_names, channel_num=3):
    """
    Data IO for files similar to APS Beamline 13 data format.
    This might be changed later.

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
    data_sets : dict
        data from each channel and channel summed
    """
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    for fname in file_names:
        try:
            file_path = os.path.join(working_directory, fname)
            #with h5py.File(file_path, 'r+') as f:
            f = h5py.File(file_path, 'r+')
            data = f['xrfmap']

            fname = fname.split('.')[0]

            # data from channel summed
            exp_data = data['detsum/counts']
            #exp_data = np.asarray(exp_data[:, angle_cut:-angle_cut, :-spectrum_cut])
            exp_data = np.array(exp_data)
            print('raw data from h5: {}'.format(exp_data.shape))
            # for line plot
            try:
                exp_data[0, 0, :] = exp_data[1, 0, :]
            except IndexError:
                exp_data[0, 0, :] = exp_data[0, 1, :]
            #roi_name = data['detsum']['roi_name'].value
            #roi_value = data['detsum']['roi_limits'].value

            # temp!!!
            # fitname = list(data['detsum']['xrf_fit_name'].value)
            # ename = 'Pt_L'
            # indv = fitname.index(ename)
            # fitv = data['detsum']['xrf_fit'].value[indv,:,:]
            # cutv = 99.25
            # ind_array = fitv>cutv
            # for i in range(fitv.shape[0]):
            #     for j in range(fitv.shape[1]):
            #         if ind_array[i, j] == False:
            #             exp_data[i, j, :] *= 0

            fname_sum = fname+'_sum'
            DS = DataSelection(filename=fname_sum,
                               raw_data=exp_data)

            data_sets[fname_sum] = DS
            logger.info('Data of detector sum is loaded.')

            # data from each channel
            for i in range(1, channel_num+1):
                det_name = 'det'+str(i)
                file_channel = fname+'_channel_'+str(i)
                exp_data_new = data[det_name+'/counts']
                exp_data_new = np.array(exp_data_new)
                try:
                    exp_data_new[0, 0, :] = exp_data_new[1, 0, :]
                except IndexError:
                    exp_data_new[0, 0, :] = exp_data_new[0, 1, :]
                DS = DataSelection(filename=file_channel,
                                   raw_data=exp_data_new)
                data_sets[file_channel] = DS
                logger.info('Data from detector channel {} is loaded.'.format(i))

            if 'roimap' in data:
                if 'sum_name' in data['roimap']:
                    det_name = data['roimap/sum_name']
                    temp = {}
                    for i, n in enumerate(det_name):
                        temp[n] = data['roimap/sum_raw'].value[:, :, i]
                        try:
                            temp[n][0, 0] = temp[n][1, 0]
                        except IndexError:
                            temp[n][0, 0] = temp[n][0, 1]
                    img_dict[fname+'_roi'] = temp

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
                fit_result = get_fit_data(data['detsum']['xrf_fit_name'].value,
                                          data['detsum']['xrf_fit'].value)
                img_dict.update({fname+'_fit': fit_result})

            if 'scalers' in data:
                det_name = data['scalers/name']
                temp = {}
                for i, n in enumerate(det_name):
                    temp[n] = data['scalers/val'].value[:, :, i]
                img_dict[fname+'_scaler'] = temp

            f.close()

        except ValueError:
            continue
    return img_dict, data_sets


def read_MAPS(working_directory,
              file_names, channel_num=1):
    data_dict = OrderedDict()
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 0

    for fname in file_names:
        try:
            file_path = os.path.join(working_directory, fname)
            with h5py.File(file_path, 'r+') as f:

                data = f['MAPS']

                fname = fname.split('.')[0]

                # for 2D MAP
                #data_dict[fname] = data

                # raw data
                exp_data = data['mca_arr'][:]

                # data from channel summed
                roi_channel = data['channel_names'].value
                roi_val = data['XRF_roi'][:]

                # data from fit
                fit_val = data['XRF_fits'][:]

            exp_shape = exp_data.shape

            exp_data = exp_data.T
            logger.info('File : {} with total counts {}'.format(fname,
                                                                np.sum(exp_data)))
            DS = DataSelection(filename=fname,
                               raw_data=exp_data)
            data_sets.update({fname: DS})

            # save roi and fit into dict
            temp_roi = {}
            temp_fit = {}
            for i, name in enumerate(roi_channel):
                temp_roi[name] = roi_val[i, :, :].T
                temp_fit[name] = fit_val[i, :, :].T
            img_dict[fname+'_roi'] = temp_roi

            img_dict[fname+'_fit'] = temp_fit
            # read fitting results
            # if 'xrf_fit' in data[detID]:
            #     fit_result = get_fit_data(data[detID]['xrf_fit_name'].value,
            #                               data[detID]['xrf_fit'].value)
            #     img_dict.update({fname+'_fit': fit_result})

        except ValueError:
            continue
    return img_dict, data_sets


def read_numpy_data(working_directory,
                    file_names):
    """
    temporary use, bad example.
    """
    #import pickle
    data_sets = OrderedDict()
    img_dict = OrderedDict()

    #pickle_folder = '/Users/Li/Downloads/xrf_data/xspress3/'
    #save_name = 'scan01167_pickle'
    #fpath = os.path.join(pickle_folder, save_name)
    for file_name in file_names:
        fpath = os.path.join(working_directory, file_name)
        exp_data = np.load(fpath)
        DS = DataSelection(filename=file_name,
                           raw_data=exp_data)
        data_sets.update({file_name: DS})

    return img_dict, data_sets


def read_hdf_multi_files_HXN(working_directory,
                             file_prefix, h_dim, v_dim,
                             channel_num=4):
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
    data_sets : dict
        data from each channel and channel summed
    """
    data_dict = OrderedDict()
    data_sets = OrderedDict()

    # cut off bad point on the last position of the spectrum
    bad_point_cut = 1
    start_i = 1
    end_i = h_dim*v_dim
    total_data = np.zeros([v_dim, h_dim, 4096-bad_point_cut])

    for fileID in range(start_i, end_i+1):
        fname = file_prefix + str(fileID)+'.hdf5'
        file_path = os.path.join(working_directory, fname)
        fileID -= start_i
        with h5py.File(file_path, 'r+') as f:
            data = f['entry/instrument']

            #fname = fname.split('.')[0]

            # for 2D MAP???
            #data_dict[fname] = data

            # data from channel summed
            exp_data = np.asarray(data['detector/data'])
            ind_v = fileID//h_dim
            ind_h = fileID - ind_v * h_dim
            print(ind_v, ind_h)
            total_data[ind_v, ind_h, :] = np.sum(exp_data[:, :3, :-bad_point_cut],
                                                 axis=(0, 1))

    DS = DataSelection(filename=file_prefix,
                       raw_data=total_data)
    data_sets.update({file_prefix: DS})

    return data_dict, data_sets


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
                    roi_dict={'Pt_9300_9600': ['Ch1 [9300:9600]', 'Ch2 [9300:9600]', 'Ch3 [9300:9600]']},
                    pos_list=('ssx[um]', 'ssy[um]'),
                    scaler_list=('sclr1_ch2', 'sclr1_ch3', 'sclr1_ch8')):
    """
    Assume data is obained from databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : array
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

    for n in range(len(det_list)):
        detname = 'det'+str(n+1)
        try:
            dataGrp = f.create_group(interpath+'/'+detname)
        except ValueError:
            dataGrp = f[interpath+'/'+detname]

        c_name = det_list[n]
        logger.info('read data from %s' % c_name)
        channel_data = data[c_name]
        new_data = np.zeros([1, len(channel_data), len(channel_data[0])])

        for i in xrange(len(channel_data)):
            channel_data[i][pd.isnull(channel_data[i])] = 0
            new_data[0, i, :] = channel_data[i]
        if sum_data is None:
            sum_data = new_data
        else:
            sum_data += new_data

        new_data = new_data.reshape([datashape[0], datashape[1],
                                     len(channel_data[0])])

        if 'counts' in dataGrp:
            del dataGrp['counts']
        ds_data = dataGrp.create_dataset('counts', data=new_data)
        ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)

    # summed data
    try:
        dataGrp = f.create_group(interpath+'/detsum')
    except ValueError:
        dataGrp = f[interpath+'/detsum']

    sum_data = sum_data.reshape([datashape[0], datashape[1],
                                 len(channel_data[0])])
    print('sum data shape: {}'.format(sum_data.shape))
    if 'counts' in dataGrp:
        del dataGrp['counts']
    ds_data = dataGrp.create_dataset('counts', data=sum_data)
    ds_data.attrs['comments'] = 'Experimental data from channel sum'

    # position data
    # try:
    #     dataGrp = f.create_group(interpath+'/positions')
    # except ValueError:
    #     dataGrp = f[interpath+'/positions']
    #
    # pos_names, pos_data = get_name_value_from_db(pos_list, data,
    #                                              datashape)
    #
    # if 'pos' in dataGrp:
    #     del dataGrp['pos']
    #
    # if 'name' in dataGrp:
    #     del dataGrp['name']
    # dataGrp.create_dataset('name', data=pos_names)
    # dataGrp.create_dataset('pos', data=pos_data)

    # scaler data
    try:
        dataGrp = f.create_group(interpath+'/scalers')
    except ValueError:
        dataGrp = f[interpath+'/scalers']

    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                       datashape)

    if 'val' in dataGrp:
        del dataGrp['val']

    if 'name' in dataGrp:
        del dataGrp['name']
    dataGrp.create_dataset('name', data=scaler_names)
    dataGrp.create_dataset('val', data=scaler_data)

    # roi sum
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
        print(v)
        posv = np.asarray(data[v])
        pos_data[:, :, i] = posv.reshape([datashape[0], datashape[1]])
        pos_names.append(str(v))
    return pos_names, pos_data


def db_to_hdf(fpath, runid,
              datashape,
              det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
              roi_dict={'Pt_9300_9600': ['Ch1 [9300:9600]', 'Ch2 [9300:9600]', 'Ch3 [9300:9600]']},
              pos_list=('ssx[um]', 'ssy[um]'),
              scaler_list=('sclr1_ch2', 'sclr1_ch3', 'sclr1_ch8')):
    """
    Read data from databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    runid : int
        id number for given run
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple or optional
        list of detector channels
    roi_list : list, tuple, optional
        list of roi pv names
    pos_list : list, tuple or optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    """
    data = fetch_data_from_db(runid)
    write_db_to_hdf(fpath, data,
                    datashape, det_list=det_list,
                    roi_dict=roi_dict,
                    pos_list=pos_list,
                    scaler_list=scaler_list)


def get_roi_keys(all_keys, beamline='HXN'):
    """
    Get roi dict in terms of beamlines.
    """
    if beamline == 'HXN':
        return get_roi_keys_hxn(all_keys)


def get_roi_keys_hxn(all_keys):
    """
    Reorganize detector names of roi detector.

    Parameters
    ----------
    all_keys : list
        pv names

    Returns
    -------
    dict:
        format as {'Ch0_sum': ['ch0_1', 'ch0_2', 'ch0_3']}
    """
    Ch1_list = sorted([v for v in all_keys if 'Det1' in v and 'xspress' not in v])
    Ch2_list = sorted([v for v in all_keys if 'Det2' in v and 'xspress' not in v])
    Ch3_list = sorted([v for v in all_keys if 'Det3' in v and 'xspress' not in v])

    Ch_dict = {}
    for i in range(len(Ch1_list)):
        k = Ch1_list[i].replace('_1', '_sum')
        val = [Ch1_list[i], Ch2_list[i], Ch3_list[i]]
        Ch_dict[k] = val
    return Ch_dict


def data_to_hdf_config(fpath, data,
                       datashape, config_file):
    """
    Assume data is ready from databroker, so save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : array
        data from data broker
    datashape : tuple or list
        shape of two D image
    config_file : str
        path to json file which saves all pv names
    """
    with open(config_file, 'r') as json_data:
        config_data = json.load(json_data)

    roi_dict = get_roi_keys(data.keys(), beamline=config_data['beamline'])

    write_db_to_hdf(fpath, data,
                    datashape,
                    det_list=config_data['xrf_detector'],
                    roi_dict=roi_dict,
                    pos_list=config_data['pos_list'],
                    scaler_list=config_data['scaler_list'])


def db_to_hdf_config(fpath, runid,
                     datashape, config_file):
    """
    Assume data is ready from databroker, so save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    runid : int
        id number for given run
    datashape : tuple or list
        shape of two D image
    config_file : str
        path to json file which saves all pv names
    """
    data = fetch_data_from_db(runid)
    data_to_hdf_config(fpath, data, datashape, config_file)
