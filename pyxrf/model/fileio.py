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
import skimage.io as sio

from atom.api import Atom, Str, observe, Typed, Dict, List, Int, Enum, Float

import logging
logger = logging.getLogger(__name__)

try:
    from databroker.databroker import DataBroker as db, get_table
    # registers a filestore handler for the XSPRESS3 detector
    from hxntools import handlers as hxn_handlers
except ImportError as e:
    db = None
    logger.error('databroker and hxntools are not available: %s', e)


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
    #output_directory = Str()
    file_names = List()
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
    file_name = Str()

    def __init__(self, **kwargs):
        self.working_directory = kwargs['working_directory']
        #self.output_directory = kwargs['output_directory']
        #with self.suppress_notifications():
        #    self.working_directory = working_directory

    # @observe('working_directory')
    # def working_directory_changed(self, changed):
    #     # make sure that the output directory stays in sync with working
    #     # directory changes
    #     self.output_directory = self.working_directory

    @observe('file_names')
    def update_more_data(self, change):
        self.file_channel_list = []
        #self.file_names.sort()
        logger.info('Files are loaded: %s' % (self.file_names))

        # focus on single file only for now
        self.file_name = self.file_names[0]

        self.img_dict, self.data_sets = file_handler(self.working_directory,
                                                     self.file_names)

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

        fname = self.fname_from_db
        fpath = os.path.join(self.working_directory, fname)
        config_file = os.path.join(self.working_directory, 'pv_config.json')
        db_to_hdf_config(fpath, self.runid,
                         datashape, config_file)
        self.file_names = [fname]

    @observe('file_opt')
    def choose_file(self, change):
        if self.file_opt == 0:
            return

        # selected file name from all channels
        # controlled at top level gui.py startup
        self.selected_file_name = self.file_channel_list[self.file_opt-1]

        names = self.data_sets.keys()

        # to be passed to fitting part for single pixel fitting
        self.data_all = self.data_sets[names[self.file_opt-1]].raw_data

        # spectrum is averaged in terms of pixel size,
        # fit doesn't work well if spectrum value is too large.
        spectrum = self.data_sets[names[self.file_opt-1]].get_sum()
        #self.data = spectrum/np.max(spectrum)
        #self.data = spectrum/(self.data_all.shape[0]*self.data_all.shape[1])
        self.data = spectrum


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
            return self.data[self.pos1[0], self.pos1[1], :]
        else:
            return np.sum(self.data[self.pos1[0]:self.pos2[0], self.pos1[1]:self.pos2[1], :],
                          axis=(0, 1))


def file_handler(working_directory, file_names):
    # be alter: to be update, temporary use!!!
    if '.h5' in file_names[0] or '.hdf5' in file_names[0]:
        #logger.info('Load APS 13IDE data format.')
        return read_hdf_APS(working_directory, file_names[0])
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


def flip_data(input_data):
    """
    Flip 2D or 3D array. The flip happens on the second index of shape.

    Parameters
    ----------
    input_data : 2D or 3D array.

    Returns
    -------
    flipped data
    """
    new_data = np.array(input_data)
    data_shape = input_data.shape
    if len(data_shape) == 2:
        new_data[1::2, :] = new_data[1::2, ::-1]

    if len(data_shape) == 3:
        new_data[1::2, :, :] = new_data[1::2, ::-1, :]
    return new_data


def read_log_file_srx(fpath, name='ch 2'):
    """
    Read given column value from log file.

    Parameters
    ----------
    fpath : str
        path to log file.
    name : str
        name of a given column

    Returns
    -------
    1d array:
        selected ic value.
    """
    line_num = 7
    base_val = 8.5*1e-10

    with open(fpath, 'r') as f:
        lines = f.readlines()
        col_names = lines[line_num].split('\t')
        col_names = [v for v in col_names if len(v) != 0]

    df = pd.read_csv(fpath,
                     skiprows=line_num+1, skip_footer=1, sep='\s+', header=None)
    df.columns = col_names
    i0 = np.abs(np.array(df[name]) - base_val)
    return i0


def write_xspress3_data_to_hdf(fpath, data_dict):
    """
    Assume data is obained from databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : dict
        data_dict with data from each channel
    """

    interpath = 'xrfmap'
    sum_data = None
    channel_list = [k for k in six.iterkeys(data_dict) if 'channel' in k]
    pos_names = [k for k in six.iterkeys(data_dict) if 'pos' in k]
    scaler_names = [k for k in six.iterkeys(data_dict) if 'scaler' in k]

    with h5py.File(fpath, 'a') as f:

        for n in range(len(channel_list)):
            detname = 'det'+str(n+1)
            try:
                dataGrp = f.create_group(interpath+'/'+detname)
            except ValueError:
                dataGrp = f[interpath+'/'+detname]

            if sum_data is None:
                sum_data = data_dict[channel_list[n]]
            else:
                sum_data += data_dict[channel_list[n]]

            if 'counts' in dataGrp:
                del dataGrp['counts']
            ds_data = dataGrp.create_dataset('counts', data=data_dict[channel_list[n]])
            ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)

        # summed data
        try:
            dataGrp = f.create_group(interpath+'/detsum')
        except ValueError:
            dataGrp = f[interpath+'/detsum']

        if 'counts' in dataGrp:
            del dataGrp['counts']
        ds_data = dataGrp.create_dataset('counts', data=sum_data)
        ds_data.attrs['comments'] = 'Experimental data from channel sum'

        data_shape = sum_data.shape

        # position data
        try:
            dataGrp = f.create_group(interpath+'/positions')
        except ValueError:
            dataGrp = f[interpath+'/positions']

        pos_data = []
        for k in pos_names:
            pos_data.append(data_dict[k])
        pos_data = np.array(pos_data)

        if 'pos' in dataGrp:
            del dataGrp['pos']

        if 'name' in dataGrp:
            del dataGrp['name']
        dataGrp.create_dataset('name', data=pos_names)
        dataGrp.create_dataset('pos', data=pos_data)

        # scaler data
        scaler_data = np.ones([data_shape[0], data_shape[1], len(scaler_names)])
        for i in np.arange(len(scaler_names)):
            scaler_data[:, :, i] = data_dict[scaler_names[i]]
        scaler_data = np.array(scaler_data)
        print('shape for scaler: {}'.format(scaler_data.shape))

        try:
            dataGrp = f.create_group(interpath+'/scalers')
        except ValueError:
            dataGrp = f[interpath+'/scalers']

        if 'val' in dataGrp:
            del dataGrp['val']

        if 'name' in dataGrp:
            del dataGrp['name']
        dataGrp.create_dataset('name', data=scaler_names)
        dataGrp.create_dataset('val', data=scaler_data)


def xspress3_data_to_hdf(fpath_hdf5, fpath_log, fpath_out):
    """
    Assume data is obained from databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath_hdf5: str
        path to raw hdf file
    fpath_log: str
        path to log file
    fpath_out: str
        path to save hdf file
    """
    data_dict = read_xspress3_data(fpath_hdf5)

    data_ic = read_log_file_srx(fpath_log)
    shapev = None

    for k, v in six.iteritems(data_dict):
        data_dict[k] = flip_data(data_dict[k])
        if shapev is None:
            shapev = data_dict[k].shape

    data_ic = data_ic.reshape([shapev[0], shapev[1]])
    print('data ic shape {}'.format(data_ic.shape))
    data_ic = flip_data(data_ic)

    data_dict['scalers_ch2'] = data_ic

    write_xspress3_data_to_hdf(fpath_out, data_dict)


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

    detlist = ['detsum','det1', 'det2', 'det3']
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
                 file_name, channel_num=3,
                 spectrum_cut=3000,
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
    channel_num : int, optional
        detector channel number
    spectrum_cut : int, optional
        only use spectrum from, say 0, 3000
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

        # data from channel summed
        exp_data = np.array(data['detsum/counts'][:, :, 0:spectrum_cut])
        logger.warning('We use spectrum range from 0 to {}'.format(spectrum_cut))
        logger.info('Exp. data from h5 has shape of: {}'.format(exp_data.shape))
        # easy for line plot, the value on first one is so large.
        try:
            exp_data[0, 0, :] = exp_data[1, 0, :]
        except IndexError:
            exp_data[0, 0, :] = exp_data[0, 1, :]

        fname_sum = fname+'_sum'
        DS = DataSelection(filename=fname_sum,
                           raw_data=exp_data)

        data_sets[fname_sum] = DS
        logger.info('Data of detector sum is loaded.')

        if 'scalers' in data:
            det_name = data['scalers/name']
            temp = {}
            for i, n in enumerate(det_name):
                temp[n] = data['scalers/val'].value[:, :, i]
            img_dict[fname+'_scaler'] = temp

        # data from each channel
        if load_each_channel:
            for i in range(1, channel_num+1):
                det_name = 'det'+str(i)
                file_channel = fname+'_ch'+str(i)
                exp_data_new = np.array(data[det_name+'/counts'][:, :, 0:spectrum_cut])
                try:
                    exp_data_new[0, 0, :] = exp_data_new[1, 0, :]
                except IndexError:
                    exp_data_new[0, 0, :] = exp_data_new[0, 1, :]
                DS = DataSelection(filename=file_channel,
                                   raw_data=exp_data_new)
                data_sets[file_channel] = DS
                logger.info('Data from detector channel {} is loaded.'.format(i))

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
                temp[n] = data['positions/pos'].value[i, :]
            img_dict['positions'] = temp

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


def write_db_to_hdf(fpath, data, datashape, get_roi_sum_sign=False,
                    det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
                    roi_dict={'Pt_9300_9600': ['Ch1 [9300:9600]', 'Ch2 [9300:9600]', 'Ch3 [9300:9600]']},
                    pos_list=('ssx[um]', 'ssy[um]'),
                    scaler_list=('sclr1_ch3', 'sclr1_ch4')):
    """
    Assume data is obained from databroker, and save the data to hdf file.

    .. note:: This function should become part of suitcase

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
        new_data = np.zeros([1, len(channel_data), len(channel_data[1])])

        for i in xrange(len(channel_data)):
            channel_data[i+1][pd.isnull(channel_data[i+1])] = 0
            new_data[0, i, :] = channel_data[i+1]
        if sum_data is None:
            sum_data = new_data
        else:
            sum_data += new_data

        new_data = new_data.reshape([datashape[0], datashape[1],
                                     len(channel_data[1])])

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
                                 len(channel_data[1])])

    if 'counts' in dataGrp:
        del dataGrp['counts']
    ds_data = dataGrp.create_dataset('counts', data=sum_data)
    ds_data.attrs['comments'] = 'Experimental data from channel sum'

    # position data
    try:
        dataGrp = f.create_group(interpath+'/positions')
    except ValueError:
        dataGrp = f[interpath+'/positions']

    try:
        pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                     datashape)
    except:
        pos_list = ('ssx', 'ssy')  # name is different for hxn step scan
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
    pos_data = pos_data.transpose()
    print(pos_data.shape)
    data_temp = np.zeros([pos_data.shape[0], pos_data.shape[2], pos_data.shape[1]])
    for i in range(pos_data.shape[0]):
        data_temp[i,:,:] = np.rot90(pos_data[i,:,:], k=3)

    dataGrp.create_dataset('name', data=pos_names)
    dataGrp.create_dataset('pos', data=data_temp)

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

    .. note:: Requires the databroker package from NSLS2
    .. note:: This should probably be moved to suitcase!

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
        k = Ch1_list[i].replace('1', '_sum')
        val = [Ch1_list[i], Ch2_list[i], Ch3_list[i]]
        Ch_dict[k] = val
    return Ch_dict


def get_scaler_list_hxn(all_keys):
    return [v for v in all_keys if 'sclr1_' in v]


def data_to_hdf_config(fpath, data,
                       datashape, config_file=False):
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

    if config_file is not False:
        with open(config_file, 'r') as json_data:
            config_data = json.load(json_data)
        roi_dict = get_roi_keys(data.keys(), beamline=config_data['beamline'])
        scaler_list = get_scaler_list_hxn(data.keys())
        write_db_to_hdf(fpath, data,
                        datashape,
                        det_list=config_data['xrf_detector'],
                        roi_dict=roi_dict,
                        pos_list=config_data['pos_list'],
                        scaler_list=scaler_list)
    else:
        write_db_to_hdf(fpath, data, datashape)


def make_hdf(fpath, runid, datashape, config_file=False):
    """
    Assume data is ready from databroker, so save the data to hdf file.

    .. note:: Requires the databroker package from NSLS2

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
    print('Loading data from database.')
    data = fetch_data_from_db(runid)
    print('Save data to hdf file.')
    data_to_hdf_config(fpath, data, datashape, config_file=config_file)
