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

import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

from atom.api import Atom, Str, observe, Typed, Dict, List, Int

# The following lines need to be updated.
# A better design to hook up with meta data store needs to be done.
folder = '/Users/Li/Research/X-ray/Research_work/all_code/nsls2_gui/nsls2_gui'
file = '2xfm_0304.h5'  # 'NSLS_X27.txt'
data_path = os.path.join(folder, file)
the_data = 0 #np.loadtxt(data_path)


class FileIOModel(Atom):
    """
    This class focuses on file input and output.

    Attributes
    ----------
    folder_name : str
    file_name : str
    data : array
        Experiment data.
    file_path : str
    load_status : str
        Description of file loading status
    """
    folder_name = Str(folder)
    file_name = Str(file)
    file_names = List()
    data = Typed(np.ndarray)
    file_path = Str(data_path)
    path_list = List()
    load_status = Str()
    file_opt = Int()
    data_dict = Dict()

    def __init__(self):
        pass
        #self.load_data()

    #@observe('folder_name', 'file_name')
    #def update(self, changed):
    #    if changed['type'] == 'create':
    #        return
    #    self.file_path = os.path.join(self.folder_name, self.file_name)

    #@observe('file_path')
    #def path_changed(self, changed):
    #    self.load_data()

    @observe('data')
    def data_changed(self, data):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

    def load_data(self):
        """
        This function needs to be updated to handle other data formats.
        """
<<<<<<< HEAD:bubblegum/xrf/model/fileio.py
        if '.txt' in self.file_name:
            try:
                self.data = np.loadtxt(self.file_path)
                self.load_status = 'File {} is loaded successfully.'.format(self.file_name)
            except IOError:
                self.load_status = 'File {} doesn\'t exist.'.format(self.file_name)
            except ValueError:
                self.load_status = 'File {} can\'t be loaded. '.format(self.file_name)

        # consider hdf file now, but this will be replaced by databroker
        # eventually an data object, or dict is returned
        if '.h5' in self.file_name:
            import h5py
            try:
                print('file name is: {}'.format(self.file_path))
                f = h5py.File(self.file_path, 'r')
                g = f['MAPS']
                data = g['mca_arr']
                self.data = np.sum(data, axis=(1, 2))
                self.load_status = 'File {} is loaded successfully.'.format(self.file_name)
                #self.data =
            except IOError:
                self.load_status = 'File {} doesn\'t exist.'.format(self.file_name)
            except ValueError:
                self.load_status = 'File {} can\'t be loaded. '.format(self.file_name)
=======
        #try:
        self.data = np.loadtxt(self.file_path)
        #self.load_status = 'File {} is loaded successfully.'.format(self.file_name)
        #except IOError:
        #    self.load_status = 'File {} doesn\'t exist.'.format(self.file_name)
        #except ValueError:
        #    self.load_status = 'File {} can\'t be loaded. '.format(self.file_name)

    @observe('file_names')
    def update_more_data(self, changed):
        for fname in self.file_names:
            try:
                self.file_path = os.path.join(self.folder_name, fname)
                self.load_data()
                self.file_name = fname
                self.data_dict.update({fname: self.data.copy()})
            except ValueError:
                continue

            #self.pathlist.append(self.file_path)
        print('all keys: {}'.format(self.data_dict.keys()))

    @observe('file_opt')
    def choose_file(self, changed):
        print('option is {}'.format(self.file_opt))
        if self.file_opt == 0:
            return
        self.file_name = self.file_names[self.file_opt-1]
        #self.file_path = os.path.join(self.folder_name, fname)
        self.data = self.data_dict[str(self.file_name)]
>>>>>>> xrf_auto_fit:bubblegum/xrf/model/fileio.py
