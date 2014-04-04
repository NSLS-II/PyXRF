"""
Helpful classes for exploring series of images
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def _compute_limit(im, percentile):
    """
    Compute 'nice' min/max values
    """
    if percentile is None:
        return (np.min(im), np.max(im))

    raise NotImplementedError("does not exist yet _compute_limit")

class xsection_viewer(object):
    def __init__(self, fig, init_image,
                 cmap=None,
                 norm=None,
                 clim_percentile=None):
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
        self._active = True
        self._clim_pct = clim_percentile
        if cmap is None:
            cmap = 'gray'
        # this needs to respect percentile
        self.vmin, self.vmax = _compute_limit(init_image, self._clim_pct)
        # stash the figure
        self.fig = fig
        # clean it
        self.fig.clf()
        # make the main axes
        self._im_ax = fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect('equal')
        self._im_ax.xaxis.set_ticklabels(())
        self._im_ax.yaxis.set_ticklabels(())
        self._imdata = init_image
        self._im = self._im_ax.imshow(init_image, cmap=cmap, norm=norm,
                                      interpolation='none', aspect='equal')

        # make it dividable
        divider = make_axes_locatable(self._im_ax)

        # set up all the other axes
        self._ax_h = divider.append_axes('bottom', .5, pad=0.1,
                                         sharex=self._im_ax)
        self._ax_h.yaxis.set_ticklabels(())
        self._ax_v = divider.append_axes('left', .5, pad=0.1,
                                         sharey=self._im_ax)
        self._ax_v.xaxis.set_ticklabels(())
        self._ax_cb = divider.append_axes('right', .2, pad=.5)
        # add the color bar
        self._cb = fig.colorbar(self._im, cax=self._ax_cb)

        # print out the pixel value
        def format_coord(x, y):
            numrows, numcols = self._imdata.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = self._imdata[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        self._im_ax.format_coord = format_coord

        # add the cursor
        self.cur = Cursor(self._im_ax, useblit=True, color='red', linewidth=2)

        self._ax_h.set_ylim(self.vmin, self.vmax)
        self._ax_h.autoscale(enable=False)
        self._ln_h, = self._ax_h.plot(np.arange(self._imdata.shape[0]),
                                np.zeros(self._imdata.shape[0]), 'k-')

        self._ax_v.set_xlim(self.vmin, self.vmax)
        self._ax_v.autoscale(enable=False)
        self._ln_v, = self._ax_v.plot(np.zeros(self._imdata.shape[1]),
                                      np.arange(self._imdata.shape[1]), 'k-')

        def move_cb(event):
            if not self._active:
                return

            # short circuit on other axes
            if event.inaxes is not self._im_ax:
                return
            numrows, numcols = self._imdata.shape
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                col = int(x+0.5)
                row = int(y+0.5)
                if col >= 0 and col < numcols and row >= 0 and row < numrows:
                    self._ln_h.set_ydata(self._imdata[row, :])
                    self._ln_v.set_xdata(self._imdata[col, :])
            # TODO add blitting
            fig.canvas.draw()

        def click_cb(event):
            if event.inaxes is not self._im_ax:
                return
            self.active = not self.active
            if self.active:
                self.cur.onmove(event)
                move_cb(event)

        self.move_cid = self.fig.canvas.mpl_connect('motion_notify_event',
                                        move_cb)

        self.click_cid = self.fig.canvas.mpl_connect('button_press_event',
                                        click_cb)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val
        self.cur.active = val

    def update_image(self, new_image):
        """
        Update the image displayed by the main axes

        Parameters
        ----------
        new_image : 2D ndarray
           The new image to use
        """
        self.vmin, self.vmax = _compute_limit(new_image, self._clim_pct)
        self._ax_v.set_xlim(self.vmin, self.vmax)
        self._ax_h.set_ylim(self.vmin, self.vmax)
        self._imdata = new_image
        self._im.set_data(new_image)
        self._im.set_clim((self.vmin, self.vmax))
        self.fig.canvas.draw()
        pass

    def update_colormap(self, new_cmap):
        """
        Update the color map used to display the image


        """
        self._im.set_cmap(new_cmap)
        self.fig.canvas.draw()

    def update_norm(self, new_norm):
        """
        Update the norm used

        """
        self._im.set_norm(new_norm)
        self.fig.canvas.draw()
