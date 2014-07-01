from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore, QtGui

from . import AbstractMPLMessenger
from .. import AbstractMessenger1D
from ...backend.mpl.stack_1d import Stack1DView
from ...qt_widgets.control_widgets import ControlContainer


from matplotlib.cm import datad


class Stack1DMessenger(AbstractMessenger1D, AbstractMPLMessenger):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, data_list=None, key_list=None, *args, **kwargs):
        # call up the inheritance toolchain
        super(Stack1DMessenger, self).__init__(*args, **kwargs)
        # init the view
        self._view = Stack1DView(fig=self._fig, data_list=data_list,
                                 key_list=key_list)

        self._ctrl_widget = make_1D_control_box("Stack 1D")
        # connect signals to slots
        self.connect_sigs_to_slots()

    def connect_sigs_to_slots(self):
        # connect signals/slots between view widget and control widget
        self._ctrl_widget['x_shift'].valueChanged.connect(
            self.sl_update_horz_offset)
        self._ctrl_widget['y_shift'].valueChanged.connect(
            self.sl_update_vert_offset)
        self._ctrl_widget['auto_scale'].toggled.connect(
            self.sl_update_autoscaling)
        self._ctrl_widget['cmap_combo'].editTextChanged[str].connect(
            self.sl_update_cmap)
        self._ctrl_widget['clear'].clicked.connect(self.sl_clear_data)

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


def make_1D_control_box(title):
    """
    init docstring

    Parameters
    ----------
    name : str
        Name of the control widget
    """
    self = None
    ctl_box = ControlContainer(title)
    # make the control widget float

    ctl_box.create_pairspinner('x_shift', init_min=0,
                                           init_max=100, init_step=.1)
    ctl_box.create_pairspinner('y_shift', init_min=0,
                                           init_max=100, init_step=.1)

    # declare a checkbox to turn on/off auto-scaling functionality
    autoscale_box = QtGui.QCheckBox(parent=self)
    ctl_box._add_widget('auto_scale', autoscale_box)

    ctl_box.create_combobox('cmap_combo', key_list=datad.keys())

    # declare button to clear data from plot
    _btn_dataclear = QtGui.QPushButton("clear data", parent=self)
    ctl_box._add_widget('clear', _btn_dataclear)
    # padding to make it look nice
    ctl_box.addStretch()

    return ctl_box
