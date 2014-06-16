"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib import cm, colors
import numpy as np
from matplotlib.figure import Figure
from vistools.qt_widgets import common
from collections import OrderedDict

import sys


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


class OneDimStackViewer(common.AbstractDataView1D):
    """
    The OneDimStackViewer provides a UI widget for viewing a number of 1-D
    data sets with cumulative offsets in the x- and y- directions.  The
    first data set always has an offset of (0, 0).

    """
    def __init__(self, fig, x_data, y_data, lbls,
                 cmap=None, norm=None):
        """
        __init__ docstring

        Parameters
        ----------
        fig : figure to draw the artists on
        x_data : list
            list of vectors of x-coordinates
        y_data : list
            list of vectors of y-coordinates
        lbls : list
            list of the names of each data set
        cmap : colormap that matplotlib understands
        norm : mpl.colors.Normalize
        """
        od = OrderedDict()
        for (lbl, x, y) in zip(lbls, x_data, y_data):
            od[lbl] = (x, y)

        self._horz_offset = 0
        self._vert_offset = 0
        self._autoscale = False

        # initialize the data view
        common.AbstractDataView1D.__init__(self, data_dict=od, fig=fig,
                                           cmap=cmap, norm=norm)

        idx = 0
        # add the data to the main axes
        for key in self._data.keys():
            # get the (x,y) data from the dictionary
            (x, y) = self._data[key]
            # plot the (x,y) data with default offsets
            self._ax1.plot(x + idx * self._horz_offset,
                           y + idx * self._vert_offset)
            # increment the counter
            idx += 1

    def set_vert_offset(self, vert_offset):
        """
        Set the vertical offset for additional lines that are to be plotted

        Parameters
        ----------
        vert_offset : number
            The amount of vertical shift to add to each line in the data stack
        """
        self._vert_offset = vert_offset

    def set_horz_offset(self, horz_offset):
        """
        Set the horizontal offset for additional lines that are to be plotted

        Parameters
        ----------
        horz_offset : number
            The amount of horizontal shift to add to each line in the data
            stack
        """
        self._horz_offset = horz_offset

    def replot(self):
        """
        @Override
        Replot the data after modifying a display parameter (e.g.,
        offset or autoscaling) or adding new data
        """
        rgba = cm.ScalarMappable(self._norm, self._cmap)
        keys = self._data.keys()
        # number of lines currently on the plot
        num_lines = len(self._ax1.lines)
        # number of datasets in the data dict
        num_datasets = len(keys)
        # set the local counter
        counter = 0
        # loop over the datasets
        for key in keys:
            # get the (x,y) data from the dictionary
            (x, y) = self._data[key]
            # check to see if there is already a line in the axes
            if counter < num_lines:
                self._ax1.lines[counter].set_xdata(
                    x + counter * self._horz_offset)
                self._ax1.lines[counter].set_ydata(
                    y + counter * self._vert_offset)
            else:
                # a new line needs to be added
                # plot the (x,y) data with default offsets
                self._ax1.plot(x + counter * self._horz_offset,
                               y + counter * self._vert_offset)
            # compute the color for the line
            color = rgba.to_rgba(x=(counter / num_datasets))
            # set the color for the line
            self._ax1.lines[counter].set_color(color)
            # increment the counter
            counter += 1
        # check to see if the axes need to be automatically adjusted to show
        # all the data
        if(self._autoscale):
            (min_x, max_x, min_y, max_y) = self.find_range()
            self._ax1.set_xlim(min_x, max_x)
            self._ax1.set_ylim(min_y, max_y)

    def set_auto_scale(self, is_autoscaling):
        """
        Enable/disable autoscaling of the axes to show all data

        Parameters
        ----------
        is_autoscaling: bool
            Automatically rescale the axes to show all the data (true)
            or stop automatically rescaling the axes (false)
        """
        print("autoscaling: {0}".format(is_autoscaling))
        self._autoscale = is_autoscaling

    def find_range(self):
        """
        Find the min/max in x and y

        @tacaswell: I'm sure that this is functionality that matplotlib
            provides but i'm not at all sure how to do it...

        Returns
        -------
        (min_x, max_x, min_y, max_y)
        """
        if len(self._ax1.lines) == 0:
            return 0, 1, 0, 1

        # find min/max in x and y
        min_x = np.zeros(len(self._ax1.lines))
        max_x = np.zeros(len(self._ax1.lines))
        min_y = np.zeros(len(self._ax1.lines))
        max_y = np.zeros(len(self._ax1.lines))

        for idx in range(len(self._ax1.lines)):
            min_x[idx] = np.min(self._ax1.lines[idx].get_xdata())
            max_x[idx] = np.max(self._ax1.lines[idx].get_xdata())
            min_y[idx] = np.min(self._ax1.lines[idx].get_ydata())
            max_y[idx] = np.max(self._ax1.lines[idx].get_ydata())

        return (np.min(min_x), np.max(max_x), np.min(min_y), np.max(max_y))


class OneDimStackCanvas(common.AbstractCanvas1D):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """
    def __init__(self, x, y, labels, parent=None):
        # create a figure to display the mpl axes
        fig = Figure(figsize=(24, 24))
        # call the parent class initialization method
        common.AbstractCanvas1D.__init__(self, fig=fig, parent=parent)
        # create the 1-D Stack viewer
        self._view = OneDimStackViewer(fig, x, y, labels)

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

        # declare button to generate data for testing/example purposes
        self._btn_datagen = QtGui.QPushButton("add data set", parent=self)
        # declare button to clear data from plot
        self._btn_dataclear = QtGui.QPushButton("clear data", parent=self)
        # declare button to append data to existing data set
        self._btn_append = QtGui.QPushButton("append data", parent=self)

        # set up action-on-click for the buttons
        self._btn_datagen.clicked.connect(self.datagen)
        self._btn_dataclear.clicked.connect(self.clear)
        self._btn_append.clicked.connect(self.append_data)

        # declare the layout manager
        layout = QtGui.QFormLayout()

        # add the controls to the layout
        layout.addRow("--- Curve Shift ---", None)
        layout.addRow("x shift", self._x_shift_spinbox)
        layout.addRow("y shift", self._y_shift_spinbox)
        layout.addRow("x step", self._x_shift_step_spinbox)
        layout.addRow("y step", self._y_shift_step_spinbox)
        layout.addRow("--- Misc. ---", None)
        layout.addRow("autoscale data view", self._autoscale_box)
        layout.addRow("color scheme", self._cm_cb)
        layout.addRow(self._btn_dataclear, self._btn_datagen)
        layout.addRow(self._btn_append, None)
        # set the widget layout
        self._widget.setLayout(layout)
        self.setWidget(self._widget)

    # set up the signals
    sig_add_data = QtCore.Signal(list, list, list)
    sig_clear_data = QtCore.Signal()
    sig_remove_last_data = QtCore.Signal()
    sig_remove_first_data = QtCore.Signal()
    sig_append_data = QtCore.Signal(list, list, list)

    @QtCore.Slot()
    def datagen(self):
        num_data = 10
        self.sig_add_data.emit(range(num_data),
                               *data_gen(num_data, phase_shift=0,
                                         horz_shift=0, vert_shift=0))

    @QtCore.Slot()
    def clear(self):
        self.sig_clear_data.emit()

    @QtCore.Slot()
    def remove_first(self):
        self.sig_remove_first_data.emit()

    @QtCore.Slot()
    def remove_last(self):
        self.sig_remove_last_data.emit()

    @QtCore.Slot()
    def append_data(self):
        num_sets = 7
        # get some fake data
        x, y = data_gen(num_sets, phase_shift=0, horz_shift=25, vert_shift=0)
        # emit the signal
        self.sig_append_data.emit(range(num_sets), x, y)


class OneDimStackViewWidget(QtGui.QWidget):

    def __init__(self, x_axis, y_data, lbl, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        # v_box_layout = QtGui.QVBoxLayout()

        self._x_axis = x_axis
        self._y_data = y_data

        self._len = len(y_data)

        # create the viewer widget
        self._canvas = OneDimStackCanvas(x_axis, y_data, lbl)

        # create a layout manager for the widget
        v_box_layout = QtGui.QVBoxLayout()

        self.mpl_toolbar = NavigationToolbar(self._canvas, self)
        # add toolbar
        v_box_layout.addWidget(self.mpl_toolbar)
        # add the 1D widget to the layout
        v_box_layout.addWidget(self._canvas)

        self.setLayout(v_box_layout)


class OneDimStackMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')
        # Generate data
        num_sets = 100
        x, y = data_gen(num_sets=num_sets, phase_shift=0,
                        horz_shift=0, vert_shift=0)
        # create view widget and control widget
        self._widget = OneDimStackViewWidget(x, y, range(num_sets))
        self._ctrl = OneDimStackControlWidget("1-D Stack Controls")

        # connect signals/slots between view widget and control widget
        self._ctrl._x_shift_spinbox.valueChanged.connect(
            self._widget._canvas.sl_update_x_offset)
        self._ctrl._y_shift_spinbox.valueChanged.connect(
            self._widget._canvas.sl_update_y_offset)
        self._ctrl._autoscale_box.clicked.connect(
            self._widget._canvas.sl_update_autoscaling)
        self._ctrl._cm_cb.editTextChanged[str].connect(
            self._widget._canvas.sl_update_color_map)

        self._ctrl.sig_add_data.connect(
            self._widget._canvas.sl_add_data)
        self._ctrl.sig_clear_data.connect(
            self._widget._canvas.sl_clear_data)
        self._ctrl.sig_append_data.connect(
            self._widget._canvas.sl_append_data)

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl)

app = QtGui.QApplication(sys.argv)
tt = OneDimStackMainWindow()
tt.show()
sys.exit(app.exec_())
