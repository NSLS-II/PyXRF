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
import numpy as np
from .. import QtCore, QtGui

class LiveWindow(QtGui.QMainWindow):
    """Main window for live display

    """

    def __init__(self, title=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = 'Live Data Widget'
        self.setWindowTitle(title)

    @classmethod
    def init_data_broker(cls, num_prev_runs=5):
        """
        Function that creates a LiveWindow that is connected to the data broker

        Parameters
        ----------
        num_prev_runs : int, optional
            The number of previous runs to obtain from the data broker.
        """
        title = 'Live Data Widget (Connected to the Data Broker)'
        cls = cls(title=title)
        # do some stuff to set up the data broker
        # read a configuration file
        databrokerpath = os.path.expanduser('~/databroker.conf')
        # connect to the data broker
        if os.path.exists(databrokerpath):
            from databroker import connect
            connect(databrokerpath, cls.update)

    @classmethod
    def init_demo(cls):
        """
        Function that creates a LiveWindow for demo-ing purposes
        """
        title = "Live data demo"
        instance = cls(title)

        # create a dock widget and set its widget to the databroker dock
        dock = QtGui.QDockWidget()
        dock.setFloating(True)
        instance.sidebar = DataBrokerSidebar()
        instance.addDockWidget(QtCore.Qt.LeftDockWidgetArea, instance.sidebar)

        # set up the sidebar
        instance.canvas = instance.create_canvas()
        instance.setCentralWidget(instance.canvas)

        return instance

    def create_canvas(self):
        canvas = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        canvas.setLayout(layout)
        layout.addWidget(QtGui.QLabel('label 1'))
        layout.addWidget(QtGui.QLabel('label 2'))
        return canvas

    # define the update options as signals that can be hooked in to
    _update_header = QtCore.Signal(object)
    _update_ev_desc = QtCore.Signal(object)
    _update_event = QtCore.Signal(object)

    @QtCore.Slot(str, object)
    def update(self, msg, msg_obj):
        """ Message passing function

        Update takes a message and an optional object.  The message is used to
        determine behavior of the qt widget. If the object is present, the qt widget
        will respond if it understands the message

        Parameters
        ----------
        msg : {'header', 'event_descriptor', 'event'}
            Message that is used to instruct the widget to perform some action
            header : data broker header object
            event_descriptor : data broker event descriptor
            event : pandas.DataFrame
                Time-aligned data with one or more named columns

        msg_obj : object
            Object that is related to the message
        """
        self._update_dict[msg].emit(msg_obj)


    _update_dict = {'header': _update_header,
                    'event_descriptor': _update_ev_desc,
                    'event': _update_event,
    }


class DataBrokerSidebar(QtGui.QDockWidget):
    """
    This object contains the Data Broker Sidebar
    """

    def __init__(self, parent=None):
        """
        Function that creates a dock widget that has two slots for updating
        the run header and the event descriptor

        Parameters
        ----------
        parent : QtGui.QtWidget, optional
            The parent widget. It's a Qt thing...
        """
        QtGui.QDockWidget.__init__(self)
        # make the control widget float
        self.setFloating(True)

        # add a widget that lives in the floating control widget
        self._widget = QtGui.QWidget(self)
        # give the widget to the dock widget
        self.setWidget(self._widget)
        # create a layout
        layout = QtGui.QVBoxLayout()

        # set the layout to the widget
        self._widget.setLayout(layout)
        self.lbl_hdr = QtGui.QLabel('waiting for run header')
        self.lbl_ev_desc = QtGui.QLabel('waiting for event descriptor')

        self._widget.setLayout(layout)

        layout.addWidget(self.lbl_hdr)
        layout.addWidget(self.lbl_ev_desc)

    @QtCore.Slot(object)
    def update_header(self, hdr):
        self.lbl_hdr.setText('Run header ' + str(np.random.rand()))
    @QtCore.Slot(object)
    def update_ev_desc(self, ev_desc):
        self.lbl_ev_desc.setText('' + str(np.random.rand()))
