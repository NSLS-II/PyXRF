from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import zip

from matplotlib import cm

from ...backend import AbstractDataView

__author__ = 'Eric-hafxb'


class AbstractMPLDataView(AbstractDataView):
    """
    Class docstring
    """
    _default_cmap = 'jet'
    _default_norm = cm.colors.Normalize(vmin=0, vmax=1)

    def __init__(self, fig, cmap=None, norm=None, *args, **kwargs):
        """
        Docstring

        Parameters
        ----------
        fig : mpl.Figure
        """

        # call up the inheritance chain
        super(AbstractMPLDataView, self).__init__(*args, **kwargs)
        # stash the figure
        self._fig=fig
        # set some defaults
        self._cmap = cmap
        self._norm = norm
        if cmap is None:
            self._cmap = self._default_cmap
        if norm is None:
            self._norm = self._default_norm

    def replot(self):
        raise NotImplementedError("This method must be implemented by daughter classes")

    def update_cmap(self, cmap):
        """
        Update the color map used to display the image
        Parameters
        ----------
        cmap : mpl.cm.colors.Colormap
        """
        self._cmap = cmap

    def update_norm(self, norm):
        """
        Updates the normalization function used for the color mapping

        Parameters
        ----------
        norm : mpl.cm.colors.Normalize
        """
        self._norm = norm

    def draw(self):
        self._fig.canvas.draw()