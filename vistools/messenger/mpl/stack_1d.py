from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore, QtGui

from . import AbstractMPLMessenger
from .. import AbstractMessenger1D
from ...backend.mpl.stack_1d import Stack1DView
from ...backend.mpl import AbstractMPLDataView
from matplotlib.cm import datad


class Stack1DMessenger(AbstractMessenger1D, AbstractMPLMessenger):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, data_dict=None, key_list=None, *args, **kwargs):
        # call up the inheritance toolchain
        super(Stack1DMessenger, self).__init__(data_dict=data_dict,
                                               key_list=key_list,
                                               *args, **kwargs)
        # init the view
        self._view = Stack1DView(fig=self._fig, data_list=data_dict)

        self._ctrl_widget= Stack1DControlWidget(name="1-D Stack Controls")
        # connect signals to slots
        self.connect_sigs_to_slots()

    def connect_sigs_to_slots(self):
        # connect signals/slots between view widget and control widget
        self._ctrl_widget._x_shift_spinbox.valueChanged.connect(
            self.sl_update_horz_offset)
        self._ctrl_widget._y_shift_spinbox.valueChanged.connect(
            self.sl_update_vert_offset)
        self._ctrl_widget._autoscale_box.toggled.connect(
            self.sl_update_autoscaling)
        self._ctrl_widget._cm_cb.editTextChanged[str].connect(
            self.sl_update_cmap)
        self._ctrl_widget.sig_clear_data.connect(
            self.sl_clear_data)

    @QtCore.Slot(float)
    def sl_update_horz_offset(self, horz_offset):
        """
        Slot to update the x-offset such that additional lines are shifted
        along the x-axis by (idx * horz_offset)

        Parameters
        ----------
        horz_offset : float
            The amount to offset subsequent plots by (idx * horz_offset)
        """
        self._view.set_horz_offset(horz_offset)
        self.sl_update_view()


    @QtCore.Slot(float)
    def sl_update_vert_offset(self, vert_offset):
        """
        Slot to update the y-offset such that additional lines are shifted
        along the y-axis by (idx * vert_offset)

        Parameters
        ----------
        vert_offset : float
            The amount to offset subsequent plots by (idx * vert_offset)
        """
        self._view.set_vert_offset(vert_offset)
        self.sl_update_view()

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
        self.sl_update_view()


class Stack1DControlWidget(QtGui.QDockWidget):
    """
    Control widget class

    TODO: Make this more modular...
    @tacaswell seemed to be doing this in /vistools/qt_widgets/common.py

    """

    _CMAPS = datad.keys()
    _CMAPS.sort()
    default_cmap = AbstractMPLDataView._default_cmap

    def __init__(self, name):
        """
        init docstring

        Parameters
        ----------
        name : str
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
        self._cm_cb.addItems(self._CMAPS)
        self._cm_cb.setEditText(self.default_cmap)

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
