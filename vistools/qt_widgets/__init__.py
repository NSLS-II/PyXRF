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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .. import QtCore, QtGui
from ..messenger.mpl.stack_1d import Stack1DMessenger
from ..messenger.mpl.cross_section_2d import CrossSection2DMessenger

import logging
logger = logging.getLogger(__name__)

class CrossSectionMainWindow(QtGui.QMainWindow):
    """
    MainWindow
    """

    def __init__(self, title=None, parent=None,
                 data_list=None, key_list=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = "2D Cross Section"
        self.setWindowTitle(title)
        # create view widget, control widget and messenger pass-through
        self._messenger = CrossSection2DMessenger(data_list=data_list,
                                                  key_list=key_list)

        self._ctrl_widget = self._messenger._ctrl_widget
        self._display = self._messenger._display

        # finish the init
        self._display.setFocus()
        self.setCentralWidget(self._display)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)


class Stack1DMainWindow(QtGui.QMainWindow):
    """
    MainWindow
    """

    def __init__(self, title=None, parent=None,
                 data_list=None, key_list=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = "1D Stack"
        self.setWindowTitle(title)
        # create view widget, control widget and messenger pass-through
        self._messenger = Stack1DMessenger(data_list=data_list,
                                           key_list=key_list)

        self._ctrl_widget = self._messenger._ctrl_widget
        self._display = self._messenger._display
        dock_widget = QtGui.QDockWidget()
        dock_widget.setWidget(self._ctrl_widget)
        # finish the init
        self._display.setFocus()
        self.setCentralWidget(self._display)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           dock_widget)
