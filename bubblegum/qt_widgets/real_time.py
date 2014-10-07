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
from collections import namedtuple
import numpy as np
from . import CrossSectionMainWindow
from .. import QtCore, QtGui

class LiveWindow(QtGui.QMainWindow):
    """Main window for live display

    """

    def __init__(self, title=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = 'Live Data Widget'
        self.setWindowTitle(title)
        self._update_dict = {'header': self._update_header,
                        'event_descriptor': self._update_ev_desc,
                        'event': self._update_event,
        }

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
        instance.canvas = LiveCanvas()
        instance.setCentralWidget(instance.canvas)

        # Connect the signals and slots
        instance._update_header.connect(instance.sidebar.update_header)
        instance._update_ev_desc.connect(instance.sidebar.update_ev_desc)
        instance._update_event.connect(instance.canvas.update_event)

        # return the initialized real time widget
        return instance

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

class LiveCanvas(QtGui.QWidget):
    """
    Class which defines the canvas to display the visualization widgets
    """
    InternalWidget = namedtuple('InternalWidget', ('widget', 'row', 'col',
                                                   'row_span', 'col_span'))

    _widgets = {
        '2Dstack': CrossSectionMainWindow
    }

    def __init__(self, parent=None, *args, **kwargs):
        super(LiveCanvas, self).__init__(*args, **kwargs)
        self.canvas = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self._lbl1 = QtGui.QLabel("Label 1")
        self._lbl2 = QtGui.QLabel("Label 2")
        self.layout.addWidget(self._lbl1)
        self.layout.addWidget(self._lbl2)
        self.setLayout(self.layout)
        self._widget_list = []


    def add_plot(self, plot_type, row, col, row_span=1, col_span=1):
        """Add a plotting widget to the canvas

        Parameters
        ----------
        plot_type: {'3Dstack', '3D', '2Dstack', '2D', '1Dstack', '1D'}
            The type of plotting widget to create. Only 2D stack is currently
            implemented
        row: int
            The row to put the widget in
        col: int
            The column index to place the widget
        row_span: int, optional
            The number of row cells to 'merge' for the widget
        col_span: int, optional
            The number of column cells to 'merge' to create the widget location
        """
        new_widget = self._widgets[plot_type]()
        widg = self.InternalWidget(new_widget, row, col, row_span, col_span)
        self._widget_list.append(widg)
        self.redraw()

    def remove_plot(self, widget):
        """Remove a plotting widget from the canvas.

        Requires a reference to the actual widget

        Parameters
        ----------
        widget : LiveCanvas.InternalWidget
            The reference to the widget you wish to remove

        Returns
        -------
        was_removed : bool
            True: widget was present in the list
            False: widget was not present in the list
            There are actually three cases.
                1. Widget is not present in the internal widget list
                2. Widget is present in the internal widget list but was not
                   removed from the canvas
                3. Widget was present in the internal widget list and was
                   removed from the canvas
        """
        try:
            self._widget_list.pop(self._widget_list.index(widget))
            self.redraw()
            if widget in self._widget_list:
                return 3
            else:
                return 2
        except ValueError:
            return 1

    def redraw(self):
        """Recreate the layout

        This function is automatically called inside of LiveCanvas.add_plot()
        and LiveCanvas.remove_plot()
        """
        # create the new layout
        layout = QtGui.QBoxLayout()
        # add the widgets to the layout
        for widget in self._widget_list:
            layout.addWidget(widget.widget, widget.row, widget.col,
                             widget.row_span, widget.col_span)
        # set the new layout
        self.setLayout(layout)

    @property
    def widget_list(self):
        """
        Retrieve the actual list of widgets.  If you remove or add widgets
        to this list, call
        """
        return self._widget_list

    @QtCore.Slot(object)
    def update_event(self, event):
        rnd = np.random.rand()
        if rnd > 0.5:
            self._lbl1.setText("Label 1: " + str(rnd))
        else:
            self._lbl2.setText("Label 2: " + str(rnd))


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
        self.lbl_ev_desc.setText('Event descriptor ' + str(np.random.rand()))
