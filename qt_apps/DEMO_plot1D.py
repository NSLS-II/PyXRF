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
Example usage of 1-D stack plot widget
"""

# imports to future-proof the code
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import zip
import sys
import numpy as np
from collections import OrderedDict

from bubblegum import QtCore, QtGui
from bubblegum.qt_widgets import Stack1DMainWindow
from bubblegum.messenger.mpl.stack_1d import Stack1DMessenger

import logging
logger = logging.getLogger(__name__)


def data_gen(num_sets=1, phase_shift=0.1, vert_shift=0.1, horz_shift=0.1):
    """
    Generate some data

    Parameters
    ----------
    num_sets: int
        number of 1-D data sets to generate

    Returns
    -------
    x : np.ndarray
        x-coordinates
    y : list of np.ndarray
        y-coordinates
    """
    x_axis = np.arange(0, 25, .01)
    x = []
    y = []
    for idx in range(num_sets):
        x.append(x_axis + horz_shift)
        y.append(np.sin(x_axis + idx * phase_shift) + idx * vert_shift)

    return x, y


class demo_1d(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        # Generate data
        num_sets = 100
        x_data, y_data = data_gen(num_sets=num_sets, phase_shift=0,
                        horz_shift=0, vert_shift=0)
        data_list = []
        key_list = []
        for (lbl, x, y) in zip(range(num_sets), x_data, y_data):
            data_list.append((x, y))
            key_list.append(lbl)

        # init the 1d stack main window
        self.setWindowTitle('OneDimStack Example')
        self._main_window = Stack1DMainWindow(data_list=data_list,
                                       key_list=key_list)

        self._main_window.setFocus()
        self.setCentralWidget(self._main_window)

        # add the demo buttons
        # declare button to generate data for testing/example purposes
        btn_datagen = QtGui.QPushButton("add data set",
                                        parent=self._main_window._ctrl_widget)

        # declare button to append data to existing data set
        btn_append = QtGui.QPushButton("append data",
                                       parent=self._main_window._ctrl_widget)

        btn_datagen.clicked.connect(self.datagen)
        btn_append.clicked.connect(self.append_data)
        ctl_box = self._main_window._ctrl_widget
        demo_box = ctl_box.create_container('demo box')

        demo_box._add_widget('append', btn_append)
        demo_box._add_widget('gen', btn_datagen)

        # connect signals to test harness
        self.sig_append_demo_data.connect(
            self._main_window._messenger.sl_append_data)
        self.sig_add_demo_data.connect(
            self._main_window._messenger.sl_add_data)

    # Qt Signals for Demo
    sig_append_demo_data = QtCore.Signal(list, list, list)
    sig_add_demo_data = QtCore.Signal(list, list, list)

    @QtCore.Slot()
    def append_data(self):
        num_sets = 7
        # get some fake data
        x, y = data_gen(num_sets, phase_shift=0, horz_shift=25, vert_shift=0)
        # emit the signal
        self.sig_append_demo_data.emit(range(num_sets), x, y)

    @QtCore.Slot()
    def datagen(self):
        num_data = 10
        names = np.random.random_integers(10000, size=num_data).tolist()

        self.sig_add_demo_data.emit(names,
                                    *data_gen(num_data, phase_shift=0,
                                              horz_shift=0, vert_shift=0))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    tt = demo_1d()
    tt.show()
    sys.exit(app.exec_())
