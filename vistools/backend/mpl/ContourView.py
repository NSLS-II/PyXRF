__author__ = 'Eric-hafxb'

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import zip

import numpy as np
from . import AbstractMPLDataView
from .. import AbstractDataView2D

class ContourView(AbstractDataView2D, AbstractMPLDataView):
    """
    The OneDimContourViewer provides a UI widget for viewing a number of 1-D
    data sets as a contour plot, starting from dataset 0 at y = 0
    """

    def __init__(self, fig, data_dict=None, cmap=None, norm=None, *args,
                 **kwargs):
        """
        __init__ docstring

        Parameters
        ----------
        fig : figure to draw the artists on
        x_data : list
            list of vectors of x-coordinates
        y_data : list
            list of vectors of y-coordinates
        lbls : list
            list of the names of each data set
        cmap : colormap that matplotlib understands
        norm : mpl.colors.Normalize
        """
        # set some defaults
        # no defaults yet

        # call the parent constructors
        super(ContourView, self).__init__(data_dict=data_dict, fig=fig, *args,
                                          **kwargs)

        # create the matplotlib axes
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_aspect('equal')

        # plot the data
        self.replot()

    def replot(self):
        """
        @Override
        Replot the data after modifying a display parameter (e.g.,
        offset or autoscaling) or adding new data
        """
        # TODO: This class was originally written to convert a 1-D stack into a
        # 2-D contour.  Rewrite this replot method

        # get the keys from the dict
        keys = self._data.keys()
        # number of datasets in the data dict
        num_keys = len(keys)
        # cannot plot data if there are no keys
        if num_keys < 1:
            return
        # set the local counter
        counter = num_keys - 1
        # @tacaswell Should it be required that all datasets are the same
        # length?
        num_coords = len(self._data[keys[0]][0])
        # declare the array
        self._data_arr = np.zeros((num_keys, num_coords))
        # add the data to the main axes
        for key in self._data.keys():
            # get the (x,y) data from the dictionary
            (x, y) = self._data[key]
            # add the data to the array

            self._data_arr[counter] = y
            # decrement the counter
            counter -= 1
        # get the first dataset to get the x axis and number of y datasets
        x, y = self._data[keys[0]]
        y = np.arange(len(keys))
        # TODO: Colormap initialization is not working properly.
        self._ax.contourf(x, y, self._data_arr)  # , cmap=colors.Colormap(self._cmap))
