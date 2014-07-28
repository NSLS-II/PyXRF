from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves import zip
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullLocator
import numpy as np

from . import AbstractMPLDataView
from .. import AbstractDataView2D


def fullrange_limit_factory(limit_args=None):
    """
    Factory for returning full-range limit functions

    limit_args is ignored.
    """
    def _full_range(im):
        """
        Plot the entire range of the image

        Parameters
        ----------
        im : ndarray
           image data, nominally 2D

        limit_args : object
           Ignored, here to match signature with other
           limit functions

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.
        """
        return (np.min(im), np.max(im))

    return _full_range


def absolute_limit_factory(limit_args):
    """
    Factory for making absolute limit functions
    """
    def _absolute_limit(im):
        """
        Plot the image based on the min/max values in limit_args

        This function is a no-op and just return the input limit_args.

        Parameters
        ----------
        im : ndarray
            image data.  Ignored in this method

        limit_args : array
           (min_value, max_value)  Values are in absolute units
           of the image.

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.

        """
        return limit_args
    return _absolute_limit


def percentile_limit_factory(limit_args):
    """
    Factory to return a percentile limit function
    """
    def _percentile_limit(im):
        """
        Sets limits based on percentile.

        Parameters
        ----------
        im : ndarray
            image data

        limit_args : tuple of floats in [0, 100]
            upper and lower percetile values

        Returns
        -------
        climits : tuple
           length 2 tuple to be passed to `im.clim(...)` to
           set the color limits of a ColorMappable object.

        """
        return np.percentile(im, limit_args)

    return _percentile_limit


class CrossSection2DView(AbstractDataView2D, AbstractMPLDataView):
    """
    CrossSection2DView docstring
    """

    def __init__(self, fig, data_list, key_list, cmap=None, norm=None,
                 limit_func=None, **kwargs):
        """
        Sets up figure with cross section viewer

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to build the class on, will clear
            current contents

        init_image : 2d ndarray
            The initial image

        cmap : str,  colormap, or None
           color map to use.  Defaults to gray

        clim_percentile : float or None
           percentile away from 0, 100 to put the max/min limits at
           ie, clim_percentile=5 -> vmin=5th percentile vmax=95th percentile

        norm : Normalize or None
           Normalization function to us
        """
        if 'limit_args' in kwargs:
            raise Exception("changed API, don't use limit_args anymore, use closures")

        # call up the inheritance chain
        super(CrossSection2DView, self).__init__(fig=fig, data_list=data_list,
                                                 key_list=key_list, norm=norm,
                                                 cmap=cmap)
        self._xsection = CrossSection(fig,
                                      self._data_dict[self._key_list[0]],
                                      cmap=cmap, norm=norm,
                                      limit_func=limit_func)

    def update_cmap(self, cmap):
        self._xsection.update_cmap(cmap)

    def update_image(self, img_idx):
        self._xsection.update_image(self._data_dict[self._key_list[img_idx]])

    def replot(self):
        """
        Update the image displayed by the main axes

        Parameters
        ----------
        new_image : 2D ndarray
           The new image to use
        """
        self._xsection.update_artists()

    def update_norm(self, new_norm):
        """
        Update the way that matplotlib normalizes the image. Default is linear
        """
        self._xsection.update_norm(new_norm)

    def set_limit_func(self, limit_func):
        """
        Set the function to use to determine the color scale

        """
        self._xsection.set_limit_func(limit_func)


def auto_redraw(func):
    def inner(self, *args, **kwargs):
        force_redraw = kwargs.pop('force_redraw', None)
        if force_redraw is None:
            force_redraw = self._auto_redraw

        ret = func(self, *args, **kwargs)

        if force_redraw:
            self.update_artists()
            self._draw()

        return ret

    inner.__name__ = func.__name__
    inner.__doct__ = func.__doc__

    return inner


class CrossSection(object):
    """
    Class to manage the axes, artists and properties associated with
    showing a 2D image, a cross-hair cursor and two parasite axes which
    provide horizontal and vertical cross sections of image.

    Parameters
    ----------

    fig : matplotlib.figure.Figure
        The figure object to build the class on, will clear
        current contents

    init_image : 2d ndarray
        The initial image

    cmap : str,  colormap, or None
        color map to use.  Defaults to gray

    norm : Normalize or None
        Normalization function to us

    limit_func : callable
        function that takes in the image and returns clim values
    """
    def __init__(self, fig, init_image, cmap=None, norm=None,
                 limit_func=None, auto_redraw=True):

        # used to determine if setting properties should force a re-draw
        self._auto_redraw = auto_redraw
        # clean defaults
        if limit_func is None:
            limit_func = fullrange_limit_factory()
        if cmap is None:
            cmap = 'gray'
        # let norm pass through as None, mpl defaults to linear which is fine

        # save a copy of the limit function, we will need it later
        self._limit_func = limit_func

        # this is used by the widget logic
        self._active = True
        self._dirty = True
        self._cb_dirty = True

        # work on setting up the mpl axes

        # this needs to respect percentile
        vlim = self._limit_func(init_image)

        self._fig = fig
        # blow away what ever is currently on the figure
        fig.clf()
        # Configure the figure in our own image
        #
        #     	  +----------------------+
        #	      |   H cross section    |
        #     	  +----------------------+
        #   +---+ +----------------------+
        #   | V | |                      |
        #   |   | |                      |
        #   | x | |                      |
        #   | s | |      Main Axes       |
        #   | e | |                      |
        #   | c | |                      |
        #   | t | |                      |
        #   | i | |                      |
        #   | o | |                      |
        #   | n | |                      |
        #   +---+ +----------------------+

        # make the main axes
        self._im_ax = fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect('equal')
        self._im_ax.xaxis.set_major_locator(NullLocator())
        self._im_ax.yaxis.set_major_locator(NullLocator())
        self._imdata = init_image
        self._im = self._im_ax.imshow(init_image, cmap=cmap, norm=norm,
                                      interpolation='none', aspect='equal')

        # make it dividable
        divider = make_axes_locatable(self._im_ax)

        # set up all the other axes
        # (set up the horizontal and vertical cuts)
        self._ax_h = divider.append_axes('top', .5, pad=0.1,
                                         sharex=self._im_ax)
        self._ax_h.yaxis.set_major_locator(NullLocator())
        self._ax_v = divider.append_axes('left', .5, pad=0.1,
                                         sharey=self._im_ax)
        self._ax_v.xaxis.set_major_locator(NullLocator())
        self._ax_cb = divider.append_axes('right', .2, pad=.5)
        # add the color bar
        self._cb = fig.colorbar(self._im, cax=self._ax_cb)

        # print out the pixel value
        def format_coord(x, y):
            numrows, numcols = self._imdata.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = self._imdata[row, col]
                return "X: {x:d} Y: {y:d} I: {i:.2f}".format(x=col, y=row, i=z)
            else:
                return "X: {x:d} Y: {y:d}".format(x=col, y=row)

        self._im_ax.format_coord = format_coord

        # add the cursor
        self.cur = Cursor(self._im_ax, useblit=True, color='red', linewidth=2)

        # set the y-axis scale for the horizontal cut
        self._ax_h.set_ylim(*vlim)
        self._ax_h.autoscale(enable=False)

        # set the y-axis scale for the vertical cut
        self._ax_v.set_xlim(*vlim)
        self._ax_v.autoscale(enable=False)

        # add lines
        self._ln_v, = self._ax_v.plot(np.zeros(self._imdata.shape[0]),
                                np.arange(self._imdata.shape[0]), 'k-',
                                animated=True,
                                visible=False)

        self._ln_h, = self._ax_h.plot(np.arange(self._imdata.shape[1]),
                                np.zeros(self._imdata.shape[1]), 'k-',
                                animated=True,
                                visible=False)

        # backgrounds for blitting
        self._ax_v_bk = None
        self._ax_h_bk = None

        # stash last-drawn row/col to skip if possible
        self._row = None
        self._col = None

        # set up the call back for the updating the side axes
        def move_cb(event):
            if not self._active:
                return

            # short circuit on other axes
            if event.inaxes is not self._im_ax:
                return
            numrows, numcols = self._imdata.shape
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self._ln_h.set_visible(True)
                self._ln_v.set_visible(True)
                col = int(x + 0.5)
                row = int(y + 0.5)
                if row != self._row or col != self._col:
                    if (col >= 0 and col < numcols and
                          row >= 0 and row < numrows):
                        self._col = col
                        self._row = row
                        for data, ax, bkg, art, set_fun in zip(
                                (self._imdata[row, :], self._imdata[:, col]),
                                (self._ax_h, self._ax_v),
                                (self._ax_h_bk, self._ax_v_bk),
                                (self._ln_h, self._ln_v),
                                (self._ln_h.set_ydata, self._ln_v.set_xdata)):
                            self._fig.canvas.restore_region(bkg)
                            set_fun(data)
                            ax.draw_artist(art)
                            self._fig.canvas.blit(ax.bbox)

        def click_cb(event):
            if event.inaxes is not self._im_ax:
                return
            self.active = not self.active
            if self.active:
                self.cur.onmove(event)
                move_cb(event)

        self.move_cid = self._fig.canvas.mpl_connect('motion_notify_event',
                                        move_cb)

        self.click_cid = self._fig.canvas.mpl_connect('button_press_event',
                                        click_cb)

        self.clear_cid = self._fig.canvas.mpl_connect('draw_event', self.clear)
        self._fig.tight_layout()
        self._fig.canvas.draw()

    def clear(self, event):
        self._ax_v_bk = self._fig.canvas.copy_from_bbox(self._ax_v.bbox)
        self._ax_h_bk = self._fig.canvas.copy_from_bbox(self._ax_h.bbox)
        self._ln_h.set_visible(False)
        self._ln_v.set_visible(False)

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val
        self.cur.active = val

    @auto_redraw
    def update_cmap(self, cmap):
        """
        Set the color map used
        """
        # TODO: this should stash new value, not apply it
        self._im.set_cmap(cmap)
        self._dirty = True

    @auto_redraw
    def update_image(self, new_image):
        """
        Set the image data

        The input data must be the same shape as the current image data
        """
        self._imdata = new_image
        self._dirty = True

    @auto_redraw
    def update_norm(self, new_norm):
        """
        Update the way that matplotlib normalizes the image
        """
        self._im.set_norm(new_norm)
        self._dirty = True
        self._cb_dirty = True

    @auto_redraw
    def set_limit_func(self, limit_func):
        """
        Set the function to use to determine the color scale
        """
        # set the new function to use for computing the color limits
        self._limit_func = limit_func
        self._dirty = True

    def update_artists(self):
        """
        Updates the figure by re-drawing
        """
        # if the figure is not dirty, short-circuit
        if not (self._dirty or self._cb_dirty):
            return

        vlim = self._limit_func(self._imdata)
        # set the color bar limits
        self._im.set_clim(vlim)
        # set the cross section axes limits
        self._ax_v.set_xlim(*vlim[::-1])
        self._ax_h.set_ylim(*vlim)
        # set the imshow data
        self._im.set_data(self._imdata)

        # TODO if cb_dirty, remake the colorbar, I think this is
        # why changing the norm does not play well
        self._dirty = False
        self._cb_dirty = False

    def _draw(self):
        self._fig.canvas.draw()
