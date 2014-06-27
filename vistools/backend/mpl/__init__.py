__author__ = 'Eric-hafxb'

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import zip

from matplotlib import cm

from ...backend import AbstractDataView


class AbstractMPLDataView(AbstractDataView):
    """
    Class docstring
    """

    def __init__(self, fig, *args, **kwargs):
        super(AbstractMPLDataView, self).__init__(*args, **kwargs)
        self._fig=fig

    def replot(self):
        raise NotImplementedError("This method must be implemented by daughter classes")

    def update_cmap(self, new_cmap):
        """
        Update the color map used to display the image
        """
        self._cmap = new_cmap

    def update_norm(self, new_norm):
        """
        Updates the normalization function used for the color mapping
        """
        self._norm = new_norm