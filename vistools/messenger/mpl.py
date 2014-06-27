__author__ = 'Eric-hafxb'

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import zip

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa

from matplotlib.figure import Figure
from matplotlib.cm import datad
from matplotlib import colors

from ..backend.mpl import Stack1DView, AbstractMPLDataView
from . import AbstractMessenger, AbstractMessenger1D, AbstractMessenger2D


class AbstractMPLMessenger(AbstractMessenger):
    """
    docstring
    """

    def __init__(self, view):
        """

        Parameters
        ----------
        viewer : AbstractMPLDataView
        """
        super(AbstractMPLMessenger, self, view)
        if self._view is None:
            self._view = AbstractMPLDataView()

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
            self._view.update_colormap(str(cmap))
        except ValueError:
            pass
        self._viewerer.replot()
        self.draw()


class Stack1DMessenger(AbstractMessenger1D):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, fig, data=None):
        super(Stack1DMessenger, self).__init__()
        self._view = Stack1DView(fig=fig, data=data)

    @QtCore.Slot(float)
    def sl_update_x_offset(self, x_offset):
        """
        Slot to update the x-offset such that additional lines are shifted
        along the x-axis by (idx * x_offset)

        Parameters
        ----------
        x_offset : float
            The amount to offset subsequent plots by (idx * x_offset)
        """
        self._view.set_horz_offset(x_offset)
        self._view.replot()
        self.draw()

    @QtCore.Slot(float)
    def sl_update_y_offset(self, y_offset):
        """
        Slot to update the y-offset such that additional lines are shifted
        along the y-axis by (idx * y_offset)

        Parameters
        ----------
        y_offset : float
            The amount to offset subsequent plots by (idx * y_offset)
        """
        self._view.set_vert_offset(y_offset)
        self._view.replot()
        self.draw()

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
        self._view.replot()
        self.draw()


class CrossSection2DMessenger(AbstractMessenger2D):
    """
    This is a thin wrapper around mpl.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, data_dict, parent=None):
        super(CrossSection2DMessenger, self).__init()
        self._view = Stack2DView(fig=fig, data=data)
        # create a figure to display the mpl axes
        fig = Figure(figsize=(24, 24))
        # create the views
        view = CrossSection2DMessenger(self._fig, data_dict)
        # call the parent class initialization method
        AbstractMessenger1D.__init__(self, fig=fig, view=view)

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
