from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import zip

from matplotlib.backends.qt4_compat import QtGui, QtCore

from matplotlib import colors
import numpy as np

from ..backend.mpl import Stack1DView, AbstractMPLDataView, Stack2DView
from . import AbstractMessenger, AbstractMessenger1D, AbstractMessenger2D

__author__ = 'Eric-hafxb'


class AbstractMPLMessenger(AbstractMessenger):
    """
    docstring
    """

    def __init__(self, fig, *args, **kwargs):
        # call up the inheritance toolchain
        super(AbstractMPLMessenger, self).__init__(*args, **kwargs)
        # set a default view
        self._view = AbstractMPLDataView(fig=fig, *args, **kwargs)

    @QtCore.Slot(colors.Normalize)
    def sl_update_norm(self, new_norm):
        """
        Updates the normalization function used for the color mapping
        """
        self._view.update_norm(new_norm)

    @QtCore.Slot(colors.Colormap)
    def sl_update_color_map(self, cmap):
        """
        Updates the color map.  Currently takes a string, should probably be
        redone to take a cmap object and push the look up function up a layer
        so that we do not need the try..except block.
        """
        try:
            self._view.update_cmap(str(cmap))
        except ValueError:
            # do nothing and return
            return
        self.replot()

    @QtCore.Slot()
    def sl_replot(self):
        """
        Pass-through to directly call the replot function in the mpl backend
        """
        self._view.replot()
        self._view._fig.canvas.draw()

class Stack1DMessenger(AbstractMessenger1D):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data_dict=None, *args, **kwargs):
        # call up the inheritance toolchain
        super(Stack1DMessenger, self).__init__(*args, **kwargs)
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
        self.replot()

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
        self.replot()

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
        self.replot()


class CrossSection2DMessenger(AbstractMessenger2D):
    """
    This is a thin wrapper around mpl.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data_dict, parent=None, *args, **kwargs):
        # call up the inheritance chain
        super(CrossSection2DMessenger, self).__init( *args, **kwargs)
        # init the appropriate view
        self._view = Stack2DView(fig=fig, data_dict=data_dict)

    @QtCore.Slot(np.ndarray)
    def sl_update_image(self, img):
        """
        updates the image shown in the widget, assumed to be the same size
        """
        self._view.update_image(img)

    @QtCore.Slot(np.ndarray)
    def sl_replace_image(self, img):
        """
        Replaces the image shown in the widget, rebulids everything
        (so swap axis will work)
        """
        raise NotImplementedError()

#    @QtCore.Slot(object, tuple)
    def sl_update_limit_func(self, limit_func, new_limits):
        """
        Updates the type of limit computation function used
        """
        self._view.set_limit_func(limit_func, new_limits)

    @QtCore.Slot(tuple)
    def sl_update_color_limits(self, new_limits):
        """
        Update the values passed to the limit computation function
        """
        self._view.update_color_limits(new_limits)
