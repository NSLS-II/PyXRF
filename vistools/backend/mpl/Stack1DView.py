from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import zip

from matplotlib import cm

import numpy as np
from . import AbstractMPLDataView
from .. import AbstractDataView1D

__author__ = 'Eric-hafxb'

class Stack1DView(AbstractDataView1D, AbstractMPLDataView):
    """
    The OneDimStackViewer provides a UI widget for viewing a number of 1-D
    data sets with cumulative offsets in the x- and y- directions.  The
    first data set always has an offset of (0, 0).
    """

    _default_horz_offset = 0
    _default_vert_offset = 0
    _default_autoscale = False

    def __init__(self, fig, data_dict, cmap=None, norm=None, *args, **kwargs):
        """
        __init__ docstring

        Parameters
        ----------
        fig : figure to draw the artists on
        data_dict : OrderedDict
            dictionary of k:v as name : (x,y)
        cmap : colormap that matplotlib understands
        norm : mpl.colors.Normalize
        """
        # call the parent constructors
        super(Stack1DView, self).__init__(fig=fig, data_dict=data_dict, *args,
                                          **kwargs)

        # set some defaults
        self._horz_offset = self._default_horz_offset
        self._vert_offset = self._default_vert_offset
        self._autoscale = self._default_autoscale

        # create the matplotlib axes
        self._ax = []
        self._ax.append(self._fig.add_subplot(1, 1, 1))
        self._ax[0].set_aspect('equal')

        # create a local counter
        counter = 0
        # add the data to the main axes
        for key in self._data_dict.keys():
            # get the (x,y) data from the dictionary
            (x, y) = self._data_dict[key]
            # plot the (x,y) data with default offsets
            self._ax[0].plot(x + counter * self._horz_offset,
                           y + counter * self._vert_offset)
            # increment the counter
            counter += 1

    def set_vert_offset(self, vert_offset):
        """
        Set the vertical offset for additional lines that are to be plotted

        Parameters
        ----------
        vert_offset : number
            The amount of vertical shift to add to each line in the data stack
        """
        self._vert_offset = vert_offset

    def set_horz_offset(self, horz_offset):
        """
        Set the horizontal offset for additional lines that are to be plotted

        Parameters
        ----------
        horz_offset : number
            The amount of horizontal shift to add to each line in the data
            stack
        """
        self._horz_offset = horz_offset

    def replot(self):
        """
        @Override
        Replot the data after modifying a display parameter (e.g.,
        offset or autoscaling) or adding new data
        """
        rgba = cm.ScalarMappable(self._norm, self._cmap)
        keys = self._data_dict.keys()
        # number of lines currently on the plot
        num_lines = len(self._ax[0].lines)
        # number of datasets in the data dict
        num_datasets = len(keys)
        # set the local counter
        counter = 0
        # loop over the datasets
        for key in keys:
            # get the (x,y) data from the dictionary
            (x, y) = self._data_dict[key]
            # check to see if there is already a line in the axes
            if counter < num_lines:
                self._ax[0].lines[counter].set_xdata(
                    x + counter * self._horz_offset)
                self._ax[0].lines[counter].set_ydata(
                    y + counter * self._vert_offset)
            else:

                # a new line needs to be added
                # plot the (x,y) data with default offsets
                self._ax[0].plot(x + counter * self._horz_offset,
                               y + counter * self._vert_offset)
            # compute the color for the line
            color = rgba.to_rgba(x=(counter / num_datasets))
            # set the color for the line
            self._ax[0].lines[counter].set_color(color)
            # increment the counter
            counter += 1
        # check to see if the axes need to be automatically adjusted to show
        # all the data
        if(self._autoscale):
            (min_x, max_x, min_y, max_y) = self.find_range()
            self._ax[0].set_xlim(min_x, max_x)
            self._ax[0].set_ylim(min_y, max_y)

    def set_auto_scale(self, is_autoscaling):
        """
        Enable/disable autoscaling of the axes to show all data

        Parameters
        ----------
        is_autoscaling: bool
            Automatically rescale the axes to show all the data (true)
            or stop automatically rescaling the axes (false)
        """
        print("autoscaling: {0}".format(is_autoscaling))
        self._autoscale = is_autoscaling

    def find_range(self):
        """
        Find the min/max in x and y

        @tacaswell: I'm sure that this is functionality that matplotlib
            provides but i'm not at all sure how to do it...

        Returns
        -------
        (min_x, max_x, min_y, max_y)
        """
        if len(self._ax[0].lines) == 0:
            return 0, 1, 0, 1

        # find min/max in x and y
        min_x = np.zeros(len(self._ax[0].lines))
        max_x = np.zeros(len(self._ax[0].lines))
        min_y = np.zeros(len(self._ax[0].lines))
        max_y = np.zeros(len(self._ax[0].lines))

        for idx in range(len(self._ax[0].lines)):
            min_x[idx] = np.min(self._ax[0].lines[idx].get_xdata())
            max_x[idx] = np.max(self._ax[0].lines[idx].get_xdata())
            min_y[idx] = np.min(self._ax[0].lines[idx].get_ydata())
            max_y[idx] = np.max(self._ax[0].lines[idx].get_ydata())

        return (np.min(min_x), np.max(max_x), np.min(min_y), np.max(max_y))
