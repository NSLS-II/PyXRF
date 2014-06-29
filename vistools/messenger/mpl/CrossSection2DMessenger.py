from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore
import numpy as np

from . import AbstractMPLMessenger
from .. import AbstractMessenger2D
from ...backend.mpl.CrossSection2DView import CrossSection2DView


__author__ = 'Eric'


class CrossSection2DMessenger(AbstractMessenger2D, AbstractMPLMessenger):
    """
    This is a thin wrapper around mpl.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data_dict, parent=None, *args, **kwargs):
        # call up the inheritance chain
        super(CrossSection2DMessenger, self).__init(fig=fig, data_dict=data_dict, *args, **kwargs)
        # init the appropriate view
        self._view = CrossSection2DView(fig=fig, data_dict=data_dict)

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
