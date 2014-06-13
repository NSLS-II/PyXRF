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
        x.append(x_axis + idx * horz_shift)
        y.append(np.sin(x_axis + idx * phase_shift) + idx * vert_shift)

    return x, y


class OneDimStackViewer(object):

    def __init__(self, fig, x, y,
                 cmap=None,
                 norm=None):
        """
        __init__ docstring

        Parameters
        ----------
        fig : figure to draw the artists on
        x : list
            list of x-coordinates
        y : list
            list of y-coordinates
        cmap : colormap that matplotlib understands
        norm : mpl.colors.Normalize
        """

        # stash the figure
        self._fig = fig
        # save the x-axis
        self._x = x
        # save the y-data
        self._y = y
        # save the colormap
        if cmap is None:
            self._cmap = common._CMAPS[0]
        else:
            self._cmap = cmap
        # save the normalization
        if norm is None:
            self._norm = colors.Normalize(0, 1, clip=True)
        else:
            self._norm = norm

        self._rgba = cm.ScalarMappable(norm=self._norm, cmap=self._cmap)

        self._horz_offset = 0
        self._vert_offset = 0
        self._autoscale = False
        self._cmap = "jet"

        # create the matplotlib axes
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_aspect('equal')
        # add the data to the axes
        for idx in range(len(self._y)):
            self._ax.plot(self._x[idx] + idx * self._horz_offset,
                             self._y[idx] + idx * self._vert_offset)

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

    def add_line(self, x_data, y_data):
        """
        Add a line to the matplotlib axes object

        Parameters
        ----------
        x_data : array-like
            M x_data 1 array, x_data-axis
        y_data : array-like
            M x_data 1 array, y_data-axis
        """
        prev_len = len(self._x)
        for idx in range(len(x_data)):
            self._x.append(x_data[idx])
            self._y.append(y_data[idx])
            self._ax.plot(self._x[prev_len + idx] + (prev_len + idx) * self._horz_offset,
                          self._y[prev_len + idx] + (prev_len + idx) * self._vert_offset)

    def replot(self):
        """
        Replot the data after modifying a display parameter (e.g.,
        offset or autoscaling) or adding new data
        """
        for idx in range(len(self._ax.lines)):
            self._ax.lines[idx].set_xdata(self._x[idx] +
                                             idx * self._horz_offset)
            self._ax.lines[idx].set_ydata(self._y[idx] +
                                             idx * self._vert_offset)
            norm = idx / len(self._ax.lines)
            self._ax.lines[idx].set_color(self._rgba.to_rgba(x=norm))

        if(self._autoscale):
            (min_x, max_x, min_y, max_y) = self.find_range()
            self._ax.set_xlim(min_x, max_x)
            self._ax.set_ylim(min_y, max_y)

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

        Returns
        -------
        (min_x, max_x, min_y, max_y)
        """
        if len(self._ax.lines) == 0:
            return 0, 1, 0, 1

        # find min/max in x and y
        min_x = np.zeros(len(self._ax.lines))
        max_x = np.zeros(len(self._ax.lines))
        min_y = np.zeros(len(self._ax.lines))
        max_y = np.zeros(len(self._ax.lines))

        for idx in range(len(self._ax.lines)):
            min_x[idx] = np.min(self._ax.lines[idx].get_xdata())
            max_x[idx] = np.max(self._ax.lines[idx].get_xdata())
            min_y[idx] = np.min(self._ax.lines[idx].get_ydata())
            max_y[idx] = np.max(self._ax.lines[idx].get_ydata())

        return (np.min(min_x), np.max(max_x), np.min(min_y), np.max(max_y))

    def update_colormap(self, new_cmap):
        """
        Update the color map used to display the image
        """
        # TODO: Fix color map so that it sets the instance variable color
        # map and then in calls to "replot", each line gets mapped to one
        # of the colors
        self._rgba = cm.ScalarMappable(norm=self._norm, cmap=new_cmap)

    def remove_data(self, idx):
        """
        Remove the data at the position specified by idx

        Parameters
        ----------
        idx : int

        Returns
        -------
        x : np.ndarray
            x-coordinate array
        y : np.ndarray
            y-coordinate array
        line : mpl 2D line artist
        """
        if idx < len(self._ax.lines):
            x = self._x.pop(idx)
            y = self._y.pop(idx)
            line = self._ax.lines.pop(idx)
            return x, y, line


class OneDimStackCanvas(FigureCanvas):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """
    def __init__(self, x, y, parent=None):
        width = height = 24
        self.fig = Figure(figsize=(width, height))
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self._view = OneDimStackViewer(self.fig, x, y)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    @QtCore.Slot(float)
    def sl_update_x_offset(self, x_offset):
        self._view.set_horz_offset(x_offset)
        self._view.replot()
        self.draw()

    @QtCore.Slot(float)
    def sl_update_y_offset(self, y_offset):
        self._view.set_vert_offset(y_offset)
        self._view.replot()
        self.draw()

    @QtCore.Slot(bool)
    def sl_update_autoscaling(self, is_autoscaling):
        self._view.set_auto_scale(is_autoscaling)
        self._view.replot()
        self.draw()

    @QtCore.Slot(str)
    def sl_update_color_map(self, cmap):
        """
        Updates the color map.  Currently takes a string, should probably be
        redone to take a cmap object and push the look up function up a layer
        so that we do not need the try..except block.
        """
        try:
            self._view.update_colormap(str(cmap))
        except ValueError:
            pass
        self._view.replot()
        self.draw()

    @QtCore.Slot(np.ndarray, np.ndarray)
    def sl_set_data(self, x, y):
        """
        Overwrites the data

        Parameters
        ----------
        x : np.ndarray
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y : np.ndarray
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """

        self._view._x = x
        self._view._y = y
        self._view.replot()
        self.draw()

    @QtCore.Slot(list, list)
    def sl_add_data(self, x, y):
        """
        Overwrites the data

        Parameters
        ----------
        x : np.ndarray
            1 or more columns of x-coordinates.  Must be the same shape as y.
        y : np.ndarray
            1 or more columns of y-coordinates.  Must be the same shape as x.
        """
        self._view.add_line(x, y)
        self._view.replot()
        self.draw()

    @QtCore.Slot()
    def sl_clear_lines(self):
        """
        docstring
        """
        self._view._y = []
        self._view._x = []
        self._view._ax.clear()
        self._view.replot()
        self.draw()

    @QtCore.Slot()
    def sl_remove_last_line(self):
        """
        docstring
        """
        if self._view._x is not None and self._view._y is not None:
            self._view._x.pop()
            self._view._y.pop()
        self._view.replot()
        self.draw()

    @QtCore.Slot()
    def sl_remove_first_line(self):
        """
        docstring
        """
        if self._view._x is not None and self._view._y is not None:
            self._view.remove_data(0)
        self._view.replot()
        self.draw()


class OneDimStackControlWidget(QtGui.QDockWidget):
    """
    Control widget class
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

        # set the min/max limits
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
        self._x_shift_step_spinbox.setSingleStep(default_step)
        self._y_shift_step_spinbox.setSingleStep(default_step)

        self._x_shift_step_spinbox.setValue(default_step)
        self._y_shift_step_spinbox.setValue(default_step)

        # connect the signals
        self._x_shift_step_spinbox.valueChanged.connect(self._x_shift_spinbox.setSingleStep)
        self._y_shift_step_spinbox.valueChanged.connect(self._y_shift_spinbox.setSingleStep)

        self._autoscale_box = QtGui.QCheckBox(parent=self)

        # set up color map combo box
        self._cm_cb = QtGui.QComboBox(parent=self)
        self._cm_cb.setEditable(True)
        self._cm_cb.addItems(common._CMAPS)
        self._cm_cb.setEditText(common._CMAPS[0])

        # add data_gen button
        self._btn_datagen = QtGui.QPushButton("add data set", parent=self)
        self._btn_dataclear = QtGui.QPushButton("clear data", parent=self)
        self._btn_dataremovefirst = QtGui.QPushButton("remove first data set", parent=self)
        self._btn_dataremovelast = QtGui.QPushButton("remove last data set", parent=self)

        self._btn_datagen.clicked.connect(self.btn_datagen)
        self._btn_dataclear.clicked.connect(self.btn_clear)
        self._btn_dataremovefirst.clicked.connect(self.btn_remove_first)
        self._btn_dataremovelast.clicked.connect(self.btn_remove_last)

        # create the layout
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
        layout.addRow("", self._btn_datagen)
        layout.addRow("", self._btn_dataclear)
        layout.addRow("", self._btn_dataremovefirst)
        layout.addRow("", self._btn_dataremovelast)
        # set the widget layout
        self._widget.setLayout(layout)
        self.setWidget(self._widget)

    # set up the signals
    sig_add_data = QtCore.Signal(list, list)
    sig_clear_data = QtCore.Signal()
    sig_remove_last_data = QtCore.Signal()
    sig_remove_first_data = QtCore.Signal()

    @QtCore.Slot()
    def btn_datagen(self):
        self.sig_add_data.emit(*data_gen(100, phase_shift=0,
                                        horz_shift=0, vert_shift=0))

    @QtCore.Slot()
    def btn_clear(self):
        self.sig_clear_data.emit()

    @QtCore.Slot()
    def btn_remove_first(self):
        self.sig_remove_last_data.emit()

    @QtCore.Slot()
    def btn_remove_last(self):
        self.sig_remove_first_data.emit()


class OneDimStackWidget(QtGui.QWidget):

    def __init__(self, x_axis, y_data, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        # v_box_layout = QtGui.QVBoxLayout()

        self._x_axis = x_axis
        self._y_data = y_data

        self._len = len(y_data)

        # create the viewer widget
        self._canvas = OneDimStackCanvas(x_axis, y_data)

        # create a layout manager for the widget
        v_box_layout = QtGui.QVBoxLayout()

        self.mpl_toolbar = NavigationToolbar(self._canvas, self)
        # add toolbar
        v_box_layout.addWidget(self.mpl_toolbar)
        # add the 1D widget to the layout
        v_box_layout.addWidget(self._canvas)

        self.setLayout(v_box_layout)


class OneDimStackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')
        # Generate data
        x, y = data_gen(num_sets=100, phase_shift=0, horz_shift=0, vert_shift=0)
        # create view widget and control widget
        self._widget = OneDimStackWidget(x, y)
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
            self._widget._canvas.sl_clear_lines)
        self._ctrl.sig_remove_last_data.connect(
            self._widget._canvas.sl_remove_last_line)
        self._ctrl.sig_remove_first_data.connect(
            self._widget._canvas.sl_remove_first_line)

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl)

app = QtGui.QApplication(sys.argv)
tt = OneDimStackExplorer()
tt.show()
sys.exit(app.exec_())
