from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore

from . import AbstractMPLMessenger
from .. import AbstractMessenger1D
from ...backend.mpl.Stack1DView import Stack1DView

__author__ = 'Eric-hafxb'


class Stack1DMessenger(AbstractMessenger1D, AbstractMPLMessenger):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data_dict=None, *args, **kwargs):
        # call up the inheritance toolchain
        super(Stack1DMessenger, self).__init__(data_dict=data_dict, fig=fig,
                                               *args, **kwargs)
        # init the view
        self._view = Stack1DView(fig=fig, data_dict=data_dict)

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