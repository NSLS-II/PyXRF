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
from nsls2 import core

import logging
logger = logging.getLogger(__name__)


class FrameSourcerBrownian(QtCore.QObject):
    new_frame = QtCore.Signal(np.ndarray)

    def __init__(self, im_shape, step_scale=1, decay=30,
                 delay=500, parent=None):
        QtCore.QObject.__init__(self, parent)
        self._im_shape = np.asarray(im_shape)
        self._scale = step_scale
        self._decay = decay
        self._delay = delay
        if self._im_shape.ndim != 1 and len(self._im_shape) != 2:
            raise ValueError("image shape must be 2 dimensional "
                             "you passed in {}".format(im_shape))
        self._cur_position = np.array(np.asarray(im_shape) / 2)

        self.timer = QtCore.QTimer(parent=self)
        self.timer.timeout.connect(self.get_next_frame)
        self._count = 0

    @QtCore.Slot()
    def get_next_frame(self):
        print('fired {}'.format(self._count))
        self._count += 1
        im = self.gen_next_frame()
        self.new_frame.emit(im)
        return True

    def gen_next_frame(self):
        # add a random step
        self._cur_position += np.random.randn(2) * self._scale
        # clip it
        self._cur_position = np.array([np.clip(v, 0, mx) for
                                       v, mx in zip(self._cur_position,
                                                    self._im_shape)])

        R = core.pixel_to_radius(self._im_shape,
                                 self._cur_position).reshape(self._im_shape)
        im = np.exp((-R**2 / self._decay))
        return im

    @QtCore.Slot()
    def start(self):
        self.timer.start(self._delay)

    @QtCore.Slot()
    def stop(self):
        self.timer.stop()


class StackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle('Brownian Motion')
        self.thread = QtCore.QThread(parent=self)
        self.worker = FrameSourcerBrownian((1024, 1000), step_scale=5,
                                           decay=100, delay=1000, parent=None)
        self.worker.moveToThread(self.thread)
        self.worker.timer.moveToThread(self.thread)
        init_data = self.worker.gen_next_frame()
        print(init_data.shape)
        self._main_window = CrossSectionMainWindow(data_list=[init_data],
                                                   key_list=['foo'])

        self._main_window.setFocus()
        layout = self._main_window._ctrl_widget._widget.layout()
        self.start_btn = QtGui.QPushButton('start', parent=self)
        self.stop_btn = QtGui.QPushButton('stop', parent=self)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.start_btn.clicked.connect(self.worker.start)
        self.stop_btn.clicked.connect(self.worker.stop)
        self.setCentralWidget(self._main_window)

        self.worker.new_frame.connect(
            self._main_window._messenger._view._xsection.update_image)
        self.thread.start()


app = QtGui.QApplication(sys.argv)
tt = StackExplorer()
tt.show()

sys.exit(app.exec_())
