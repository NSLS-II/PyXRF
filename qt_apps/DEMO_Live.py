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
import six
import os
import sys
import numpy as np
from bubblegum import QtCore, QtGui
from bubblegum.qt_widgets.real_time import LiveWindow

from hashlib import sha1
import datetime

data_keys = ['x_motor', 'intensity']

def gen_id():
    """Generate a sha1 hash of the datetime
    """
    return sha1(str(datetime.datetime.utcnow()))


def gen_ev_description():
    """Return one of a few pre-defined descriptions
    """
    options = ['hkl_scan', 'ascan', 'dscan']
    return options[np.random.randint(len(options))]


def gen_hdr():
    hdr = {
        'id': gen_id(),
        'owner': 'edill',
    }
    return hdr


def gen_ev_desc():
    ev_desc = {
        'id': gen_id(),
        'description': gen_ev_description,
        'data_keys': data_keys,
    }
    return ev_desc


def gen_data():
    num_data = 15
    data = {}
    for idx, key in enumerate(data_keys):
        if 'motor' in key:
            # treat it as a motor
            data[key] = range(num_data)
        elif 'det' in key:
            # treat it as a detector
            if '0' in key:
                # assume it is a point detector
                data[key] = np.random.rand(num_data)
            elif '1' in key:
                # assume it is a strip detector
                raise NotImplementedError('strip detector is not supported '
                                          'in gen_data()')
            elif '2' in key:
                # assume it is an area detector
                raise NotImplementedError('area detector is not supported '
                                          'in gen_data()')
    return data

def gen_event():
    ev = {
        'id': gen_id(),
        'time': datetime.datetime.utcnow(),
        'data': gen_data(),
    }
    return ev


class LiveApp(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle('LiveApp')
        self._main_window = LiveWindow.init_demo()

        self._main_window.setFocus()
        self.setCentralWidget(self._main_window)

        # init the dock widget for the execute button
        btn_panel = QtGui.QDockWidget()
        btn_panel.setFloating(True)
        # create buttons
        btn_hdr = QtGui.QPushButton('new header')
        btn_ev_desc = QtGui.QPushButton('new event descriptor')
        btn_event = QtGui.QPushButton('new event')
        # connect the button click to the relevant slots
        btn_hdr.clicked.connect(self.fire_hdr)
        btn_ev_desc.clicked.connect(self.fire_ev_desc)
        btn_event.clicked.connect(self.fire_event)

        # add the buttons to the layout
        btn_panel_layout = QtGui.QVBoxLayout()
        btn_panel_layout.addWidget(btn_hdr)
        btn_panel_layout.addWidget(btn_ev_desc)
        btn_panel_layout.addWidget(btn_event)

        btn_widget = QtGui.QWidget()
        btn_widget.setLayout(btn_panel_layout)
        btn_panel.setWidget(btn_widget)
        # add the button panel to the window
        self._main_window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, btn_panel)

    @QtCore.Slot()
    def fire_hdr(self):
        # fire a new run header
        self._main_window.update("header", gen_hdr())
    @QtCore.Slot()
    def fire_ev_desc(self):
        # fire a new run header
        self._main_window.update("event_descriptor", gen_ev_desc())
    @QtCore.Slot()
    def fire_event(self):
        # fire a new run header
        self._main_window.update("event", gen_event())


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    tt = LiveApp()
    tt.show()
    sys.exit(app.exec_())
