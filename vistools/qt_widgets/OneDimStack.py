'''
Created on Jun 16, 2014

@author: edill
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore

from matplotlib.cm import datad
from ..messenger.mpl import Stack1DMessenger
from .mpl import MPLDisplayWidget


_CMAPS = datad.keys()
_CMAPS.sort()


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
        self._cm_cb.addItems(_CMAPS)
        self._cm_cb.setEditText(_CMAPS[0])

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


class OneDimStackMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, data_dict=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')
        # create view widget, control widget and messenger pass-through
        self._disp = MPLDisplayWidget()
        self._messenger = Stack1DMessenger(fig=self._disp._fig,
                                           data_dict=data_dict)
        self._ctrl_widget = OneDimStackControlWidget("1-D Stack Controls")

        # connect signals/slots between view widget and control widget
        self._ctrl_widget._x_shift_spinbox.valueChanged.connect(
            self._messenger.sl_update_horz_offset)
        self._ctrl_widget._y_shift_spinbox.valueChanged.connect(
            self._messenger.sl_update_vert_offset)
        self._ctrl_widget._autoscale_box.clicked.connect(
            self._messenger.sl_update_autoscaling)
        self._ctrl_widget._cm_cb.editTextChanged[str].connect(
            self._messenger.sl_update_cmap)
        self._ctrl_widget.sig_clear_data.connect(
            self._messenger.sl_clear_data)

        self._disp.setFocus()
        self.setCentralWidget(self._disp)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)
