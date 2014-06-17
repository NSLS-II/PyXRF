'''
Created on Jun 16, 2014

@author: edill
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore
from vistools.qt_widgets import common

from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib import cm, colors
from matplotlib.figure import Figure
from collections import OrderedDict
import numpy as np


class OneDimStackViewer(common.AbstractDataView1D):
    """
    The OneDimStackViewer provides a UI widget for viewing a number of 1-D
    data sets with cumulative offsets in the x- and y- directions.  The
    first data set always has an offset of (0, 0).

    """
    _default_horz_offset = 0
    _default_vert_offset = 0
    _default_autoscale = False

    def __init__(self, fig, data_dict=None, cmap=None, norm=None):
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
        # set some defaults
        self._horz_offset = OneDimStackViewer._default_horz_offset
        self._vert_offset = OneDimStackViewer._default_vert_offset
        self._autoscale = OneDimStackViewer._default_autoscale

        # save the data_dictionary
        if data_dict is not None:
            self._data = data_dict
        else:
            # create a new data dictionary
            self._data = OrderedDict()
        if fig is None:
            raise Exception("There must be a figure in which to render Artists")
        # stash the figure
        self._fig = fig
        # create the matplotlib axes
        self._ax1 = self._fig.add_subplot(1, 1, 1)
        self._ax1.set_aspect('equal')

        # call up the inheritance chain
        common.AbstractDataView1D.__init__(self, cmap=cmap, norm=norm)
        # create a local counter
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
    def __init__(self, data_dict=None, parent=None):
        # create a figure to display the mpl axes
        fig = Figure(figsize=(24, 24))
        # create the 1-D Stack viewer
        view = OneDimStackViewer(fig, data_dict)
        # call the parent class initialization method
        common.AbstractCanvas1D.__init__(self, fig=fig, view=view)

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

        # declare button to clear data from plot
        self._btn_dataclear = QtGui.QPushButton("clear data", parent=self)

        # set up action-on-click for the buttons
        self._btn_dataclear.clicked.connect(self.clear)

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


class OneDimStackWidget(common.AbstractMPLWidget):

    def __init__(self, data_dict=None, page_size=10, parent=None):
        # create the viewer widget
        self._canvas = OneDimStackCanvas(data_dict)
        # call up the init inheritance chain
        common.AbstractMPLWidget.__init__(self)
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

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)
