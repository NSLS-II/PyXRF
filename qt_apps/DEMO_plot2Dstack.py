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
"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from bubblegum import QtGui, QtCore
import numpy as np

from bubblegum.messenger.mpl.cross_section_2d import CrossSection2DMessenger

from bubblegum.qt_widgets import CrossSectionMainWindow

import logging
logger = logging.getLogger(__name__)


class data_gen(object):
    def __init__(self, length, func=None):
        self._len = length
        self._x, self._y = [_ * 2 * np.pi / 500 for _ in
                            np.ogrid[-500:500, -500:500]]
        self._rep = int(np.sqrt(length))

    def __len__(self):
        return self._len

    def __getitem__(self, k):
        kx = k // self._rep + 1
        ky = k % self._rep
        return np.sin(kx * self._x) * np.cos(ky * self._y) + 1.05

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        len(self._x), len(self._y)

def data_gen(length):
       x, y = [_ * 2 * np.pi / 500 for _ in
                            np.ogrid[-500:500, -500:500]]
       rep = int(np.sqrt(length))
       data = []
       lbls = []
       for idx in range(length):
            lbls.append(str(idx))
            kx = idx // rep + 1
            ky = idx % rep
            data.append(np.sin(kx * x) * np.cos(ky * y) + 1.05)
       return lbls, data


class StackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle('StackExplorer')
        key_list, data_dict = data_gen(25)
        self._main_window = CrossSectionMainWindow(data_list=data_dict,
                                                   key_list=key_list)

        self._main_window.setFocus()
        self.setCentralWidget(self._main_window)

app = QtGui.QApplication(sys.argv)
tt = StackExplorer()
tt.show()
sys.exit(app.exec_())
