from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore
from matplotlib import colors

from ...backend.mpl import AbstractMPLDataView
from .. import AbstractMessenger


__author__ = 'Eric-i5'


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
        self.sl_update_view()

    @QtCore.Slot(colors.Colormap)
    def sl_update_cmap(self, cmap):
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
        self.sl_update_view()

    @QtCore.Slot()
    def sl_update_view(self):
        self._view.replot()
        self._view._fig.canvas.draw()