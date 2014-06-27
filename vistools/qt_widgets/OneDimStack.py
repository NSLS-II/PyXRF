'''
Created on Jun 16, 2014

@author: edill
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore
from vistools.vistools.messenger import common

from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib import cm, colors
from matplotlib.figure import Figure
from collections import OrderedDict
import numpy as np
from ..backend.mpl import Stack1DView

class Stack1DMessenger(common.AbstractMessenger1D):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data=None):
        super(Stack1DMessenger, self).__init__()
        self._view = Stack1DView(fig=fig, data=data)
        # create a figure to display the mpl axes
        fig = Figure(figsize=(24, 24))
        # create the views
        default_view = 0
        self._views = []
        for view in self.views:
            # create the view
            v = view(fig, data_dict)
            # hide the axes
            v.hide_axes()
            # stash the view in a list
            self._views.append(v)
        # show the default view
        self._views[default_view].show_axes()
        # call the parent class initialization method
        common.AbstractMessenger1D.__init__(self, fig=fig, view=self._views[default_view])

    @QtCore.Slot(float)
    def sl_update_x_offset(self, x_offset):
        """
        Slot to update the x-offset such that additional lines are shifted
        along the x-axis by (idx * x_offset)

        Parameters
        ----------
        x_offset : float
            The amount to offset subsequent plots by (idx * x_offset)
        """
        self._view.set_horz_offset(x_offset)
        self._view.replot()
        self.draw()

    @QtCore.Slot(float)
    def sl_update_y_offset(self, y_offset):
        """
        Slot to update the y-offset such that additional lines are shifted
        along the y-axis by (idx * y_offset)

        Parameters
        ----------
        y_offset : float
            The amount to offset subsequent plots by (idx * y_offset)
        """
        self._view.set_vert_offset(y_offset)
        self._view.replot()
        self.draw()

    @QtCore.Slot(bool)
    def sl_update_autoscaling(self, is_autoscaling):
        """
        Set the widget to autoscale the axes to show all data (true)

        Parameters
        ----------
        is_autoscaling : boolean
            Force the axes to autoscale to show all data
        """
        self._view.set_auto_scale(is_autoscaling)
        self._view.replot()
        self.draw()

    @QtCore.Slot(str)
    def sl_change_views(self, view_name):
        """
        Change the currently active data view

        Parameters
        ----------
        view_name : String
            Name of the view to change to
        """
        self._views[view_name]._data = self._view._data
        self._view.hide_axes()
        self._view = self._views[view_name]
        self._view.show_axes()
        self._view.replot()
        self.draw()


class OneDimStackControlWidget(QtGui.QDockWidget):
    """
    Control widget class

    TODO: Make this more modular...
    @tacaswell seemed to be doing this in /vistools/qt_widgets/common.py

    """
    def __init__(self, name):
        """
        init docstring

        Parameters
        ----------
        name : String
            Name of the control widget
        """
        QtGui.QDockWidget.__init__(self, name)
        # make the control widget float
        self.setFloating(True)
        # add a widget that lives in the floating control widget
        self._widget = QtGui.QWidget(self)

        # create the combobox
        self._view_options = QtGui.QComboBox(parent=self)
        for view in Stack1DMessenger.views:
            self._view_options.addItem(str(view))

        # create the offset spin boxes
        self._x_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)
        self._y_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)

        # create the offset step size spin boxes
        self._x_shift_step_spinbox = QtGui.QDoubleSpinBox(parent=self)
        self._y_shift_step_spinbox = QtGui.QDoubleSpinBox(parent=self)

        # set the min/max limits for the spinboxes to control the offsets
        default_min = -100
        default_max = 100
        default_step = 0.01
        self._x_shift_spinbox.setMinimum(default_min)
        self._y_shift_spinbox.setMinimum(default_min)
        self._x_shift_step_spinbox.setMinimum(default_min)
        self._y_shift_step_spinbox.setMinimum(default_min)
        self._x_shift_spinbox.setMaximum(default_max)
        self._y_shift_spinbox.setMaximum(default_max)
        self._x_shift_step_spinbox.setMaximum(default_max)
        self._y_shift_step_spinbox.setMaximum(default_max)
        self._x_shift_spinbox.setSingleStep(default_step)
        self._y_shift_spinbox.setSingleStep(default_step)

        # set the default step size
        self._x_shift_step_spinbox.setSingleStep(default_step)
        self._y_shift_step_spinbox.setSingleStep(default_step)
        self._x_shift_step_spinbox.setValue(default_step)
        self._y_shift_step_spinbox.setValue(default_step)

        # connect the signals
        self._x_shift_step_spinbox.valueChanged.connect(
            self._x_shift_spinbox.setSingleStep)
        self._y_shift_step_spinbox.valueChanged.connect(
            self._y_shift_spinbox.setSingleStep)

        # declare a checkbox to turn on/off auto-scaling functionality
        self._autoscale_box = QtGui.QCheckBox(parent=self)

        # set up color map combo box
        self._cm_cb = QtGui.QComboBox(parent=self)
        self._cm_cb.setEditable(True)
        self._cm_cb.addItems(common._CMAPS)
        self._cm_cb.setEditText(common._CMAPS[0])

        # declare button to clear data from plot
        self._btn_dataclear = QtGui.QPushButton("clear data", parent=self)

        # set up action-on-click for the buttons
        self._btn_dataclear.clicked.connect(self.clear)

        # declare the layout manager
        layout = QtGui.QFormLayout()

        # add the controls to the layout
        layout.addRow("Data view: ", self._view_options)
        layout.addRow("--- Curve Shift ---", None)
        layout.addRow("x shift", self._x_shift_spinbox)
        layout.addRow("y shift", self._y_shift_spinbox)
        layout.addRow("x step", self._x_shift_step_spinbox)
        layout.addRow("y step", self._y_shift_step_spinbox)
        layout.addRow("--- Misc. ---", None)
        layout.addRow("autoscale data view", self._autoscale_box)
        layout.addRow("color scheme", self._cm_cb)
        layout.addRow(self._btn_dataclear, None)

        # set the widget layout
        self._widget.setLayout(layout)
        self.setWidget(self._widget)
        self._widget.layout()

    # set up the signals
    sig_clear_data = QtCore.Signal()

    @QtCore.Slot()
    def clear(self):
        self.sig_clear_data.emit()


class OneDimStackWidget(common.MPLDisplayWidget):

    def __init__(self, data_dict=None, page_size=10, parent=None):
        # create the viewer widget
        self._canvas = Stack1DMessenger(data_dict)
        # call up the init inheritance chain
        common.MPLDisplayWidget.__init__(self)
        # init the mpl canvas
        self.init_canvas()


class OneDimStackMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, data_dict=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')
        # create view widget and control widget
        self._widget = OneDimStackWidget(data_dict=data_dict)

        self._ctrl_widget = OneDimStackControlWidget("1-D Stack Controls")
        # connect signals/slots between view widget and control widget
        self._ctrl_widget._x_shift_spinbox.valueChanged.connect(
            self._widget._canvas.sl_update_x_offset)
        self._ctrl_widget._y_shift_spinbox.valueChanged.connect(
            self._widget._canvas.sl_update_y_offset)
        self._ctrl_widget._autoscale_box.clicked.connect(
            self._widget._canvas.sl_update_autoscaling)
        self._ctrl_widget._cm_cb.editTextChanged[str].connect(
            self._widget._canvas.sl_update_color_map)
        self._ctrl_widget.sig_clear_data.connect(
            self._widget._canvas.sl_clear_data)
        self._ctrl_widget._view_options.currentIndexChanged.connect(
            self._widget._canvas.sl_change_views)

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)
