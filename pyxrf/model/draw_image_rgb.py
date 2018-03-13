# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
from __future__ import (absolute_import, division,
                        print_function)

__author__ = 'Li Li'

import six
import numpy as np
from collections import OrderedDict
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import matplotlib.cm as cm
#from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Bool, Float
import logging
logger = logging.getLogger()

np.seterr(divide='ignore', invalid='ignore') # turn off warning on invalid division

#
# class plot_limit(Atom):
#     low = Float(0)
#     high = Float(100)
#     # g_low = Float(0)
#     # g_high = Float(100)
#     # b_low = Float(0)
#     # b_high = Float(100)


class DrawImageRGB(Atom):
    """
    This class draws RGB image.

    Attributes
    ----------
    fig : object
        matplotlib Figure
    ax : Axes
        The `Axes` object of matplotlib
    ax_r : Axes
        The `Axes` object to add the artist too
    ax_g : Axes
        The `Axes` object to add the artist too
    ax_b : Axes
        The `Axes` object to add the artist too
    file_name : str
    stat_dict : dict
        determine which image to show
    data_dict : dict
        multiple data sets to plot, such as fit data, or roi data
    data_dict_keys : list
    data_opt : int
        index to show which data is chosen to plot
    dict_to_plot : dict
        selected data dict to plot, i.e., fitting data or roi is selected
    items_in_selected_group : list
        keys of dict_to_plot
    scale_opt : str
        linear or log plot
    color_opt : str
        orange or gray plot
    scaler_norm_dict : dict
        scaler normalization data, from data_dict
    scaler_items : list
        keys of scaler_norm_dict
    scaler_name_index : int
        index to select on GUI level
    scaler_data : None or numpy
        selected scaler data
    plot_all : Bool
        to control plot all of the data or not
    """

    fig = Typed(Figure)
    ax =  Typed(Axes)
    ax_r = Typed(Axes)
    ax_g = Typed(Axes)
    ax_b = Typed(Axes)
    stat_dict = Dict()
    data_dict = Dict()
    data_dict_keys = List()
    data_opt = Int(0)
    img_title = Str()
    #plot_opt = Int(0)
    #plot_item = Str()
    dict_to_plot = Dict()
    items_in_selected_group = List()
    scale_opt = Str('Linear')
    scaler_norm_dict = Dict()
    scaler_items = List()
    scaler_name_index = Int()
    scaler_data = Typed(object)
    plot_all = Bool(False)

    rgb_name_list = List()
    index_red = Int(0)
    index_green = Int(1)
    index_blue = Int(2)
    #ic_norm = Float()
    rgb_limit = Dict()
    r_low = Int(0)
    r_high = Int(100)
    g_low = Int(0)
    g_high = Int(100)
    b_low = Int(0)
    b_high = Int(100)
    #r_bound = List()
    #rgb_limit = plot_limit()
    name_not_scalable= List()

    def __init__(self):
        self.fig = plt.figure(figsize=(3,2))
        self.ax = self.fig.add_subplot(111)
        self.ax_r, self.ax_g, self.ax_b = make_rgb_axes(self.ax, pad=0.02)
        self.rgb_name_list = ['R', 'G', 'B']
        self.name_not_scalable = ['r2_adjust'] # do not apply scaler norm on those data

    def data_dict_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        change : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_dict = change['value']

    @observe('data_dict')
    def init_plot_status(self, change):
        # initiate the plotting status once new data is coming
        self.data_opt = 0
        self.rgb_name_list = ['R', 'G', 'B']
        self.index_red = 0
        self.index_green = 1
        self.index_blue = 2

        # init of scaler for normalization
        self.scaler_name_index = 0
        self.data_dict_keys = []
        self.data_dict_keys = list(self.data_dict.keys())
        #logger.info('The following groups are included for 2D image display: {}'.format(self.data_dict_keys))

        scaler_groups = [v for v in list(self.data_dict.keys()) if 'scaler' in v]
        if len(scaler_groups) > 0:
            #self.scaler_group_name = scaler_groups[0]
            self.scaler_norm_dict = self.data_dict[scaler_groups[0]]
            # for GUI purpose only
            self.scaler_items = []
            self.scaler_items = list(self.scaler_norm_dict.keys())
            self.scaler_data = None

        self.show_image()

    @observe('data_opt')
    def _update_file(self, change):
        try:
            if self.data_opt == 0:
                self.dict_to_plot = {}
                self.items_in_selected_group = []
                self.set_stat_for_all(bool_val=False)
                self.img_title = ''
            elif self.data_opt > 0:
                self.set_stat_for_all(bool_val=False)
                plot_item = sorted(self.data_dict_keys)[self.data_opt-1]
                self.img_title = str(plot_item)
                self.dict_to_plot = self.data_dict[plot_item]
                # for GUI purpose only
                self.items_in_selected_group = []
                self.items_in_selected_group = list(self.dict_to_plot.keys())
                self.set_stat_for_all(bool_val=False)
                # set rgb value to 0 and 100
                self.init_rgb()
        except IndexError:
            pass

    def init_rgb(self):
        self.r_low = 0
        self.r_high = 100
        self.g_low = 0
        self.g_high = 100
        self.b_low = 0
        self.b_high = 100

    @observe('scaler_name_index')
    def _get_scaler_data(self, change):
        if self.scaler_name_index == 0:
            self.scaler_data = None
        else:
            scaler_name = self.scaler_items[self.scaler_name_index-1]
            #self.scaler_data = self.data_dict[self.scaler_group_name][scaler_name]
            self.scaler_data = self.scaler_norm_dict[scaler_name]
            logger.info('Use scaler data to normalize,'
                        'and the shape of scaler data is {}'.format(self.scaler_data.shape))
        self.show_image()

    @observe('scale_opt')
    def _update_scale(self, change):
        if change['type'] != 'create':
            self.show_image()

    def set_stat_for_all(self, bool_val=False):
        """
        Set plotting status for all the 2D images.
        """
        self.stat_dict.clear()
        self.stat_dict = {k: bool_val for k in self.items_in_selected_group}

    def preprocess_data(self):
        """
        Normalize data or prepare for linear/log plot.
        """

        selected_data = []
        selected_name = []

        stat_temp = self.get_activated_num()
        stat_temp = OrderedDict(sorted(six.iteritems(stat_temp), key=lambda x: x[0]))

        plot_interp = 'Nearest'

        if self.scaler_data is not None:
            if len(self.scaler_data[self.scaler_data == 0]) > 0:
                logger.warning('scaler data has zero values at {}'.format(np.where(self.scaler_data == 0)))
                self.scaler_data[self.scaler_data == 0] = np.mean(self.scaler_data[self.scaler_data != 0])
                logger.warning('Use mean value {} instead for those points'.format(np.mean(self.scaler_data)))

        if self.scale_opt == 'Linear':
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):

                if self.scaler_data is not None:
                    if k in self.name_not_scalable:
                        data_dict = self.dict_to_plot[k]
                    else:
                        data_dict = self.dict_to_plot[k]/self.scaler_data

                else:
                    data_dict = self.dict_to_plot[k]

                selected_data.append(data_dict)
                selected_name.append(k) #self.file_name+'_'+str(k)

        else:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):

                if self.scaler_data is not None:
                    if k in self.name_not_scalable:
                        data_dict = np.log(self.dict_to_plot[k])
                    else:
                        data_dict = np.log(self.dict_to_plot[k]/self.scaler_data*self.ic_norm)

                else:
                    data_dict = np.log(self.dict_to_plot[k])

                selected_data.append(data_dict)
                selected_name.append(k)

        return selected_data, selected_name

    @observe('r_low, r_high, g_low, g_high, b_low, b_high')
    def _update_scale(self, change):
        if change['type'] != 'create':
            self.show_image()

    def show_image(self):

        selected_data, selected_name = self.preprocess_data()
        selected_data = np.asarray(selected_data)

        if len(selected_name) >0 and len(selected_name) < 3:
            logger.error('Please select three elements for RGB plot.')
        elif len(selected_name) >= 3:
            if len(selected_name) > 3:
                logger.warning('Please select only three elements for RGB plot.')
            self.rgb_name_list = selected_name[:3]

        try:
            data_r = selected_data[0,:,:]
        except IndexError:
            selected_data = np.ones([3,10,10])

        name_r = self.rgb_name_list[self.index_red]
        data_r = selected_data[self.index_red,:,:]
        name_g = self.rgb_name_list[self.index_green]
        data_g = selected_data[self.index_green,:,:]
        name_b = self.rgb_name_list[self.index_blue]
        data_b = selected_data[self.index_blue,:,:]

        rgb_dict = {'data0':{'low':self.r_low, 'high':self.r_high},
                    'data1':{'low':self.g_low, 'high':self.g_high},
                    'data2':{'low':self.b_low, 'high':self.b_high}}
        # norm data
        data_r = norm_data(data_r)
        data_g = norm_data(data_g)
        data_b = norm_data(data_b)

        # set limit, there should be a better way to do this
        rl = rgb_dict['data'+str(self.index_red)]['low']
        rh = rgb_dict['data'+str(self.index_red)]['high']
        data_r[data_r<rl/100.0] = 0.0
        data_r[data_r>rh/100.0] = rh/100.0
        gl = rgb_dict['data'+str(self.index_green)]['low']
        gh = rgb_dict['data'+str(self.index_green)]['high']
        data_g[data_g<gl/100.0] = 0.0
        data_g[data_g>gh/100.0] = gh/100.0
        bl = rgb_dict['data'+str(self.index_blue)]['low']
        bh = rgb_dict['data'+str(self.index_blue)]['high']
        data_b[data_b<bl/100.0] = 0.0
        data_b[data_b>bh/100.0] = bh/100.0

        # data_r[data_r<self.rgb_limit['R'][0]/100.0] = 0.0
        # data_r[data_r>self.rgb_limit['R'][1]/100.0] = self.rgb_limit['R'][1]/100.0
        # data_g[data_g<self.rgb_limit['G'][0]/100.0] = 0.0
        # data_g[data_g>self.rgb_limit['G'][1]/100.0] = self.rgb_limit['G'][1]/100.0
        # data_b[data_b<self.rgb_limit['B'][0]/100.0] = 0.0
        # data_b[data_b>self.rgb_limit['B'][1]/100.0] = self.rgb_limit['B'][1]/100.0

        R, G, B, RGB = make_cube(data_r,
                                 data_g,
                                 data_b)

        #self.fig.clf()
        red_patch = mpatches.Patch(color='red', label=name_r)
        green_patch = mpatches.Patch(color='green', label=name_g)
        blue_patch = mpatches.Patch(color='blue', label=name_b)

        kwargs = dict(origin="upper", interpolation="nearest")
        self.ax.imshow(RGB, **kwargs)
        self.ax_r.imshow(R, **kwargs)
        self.ax_g.imshow(G, **kwargs)
        self.ax_b.imshow(B, **kwargs)

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # sb_x = 38
        # sb_y = 46
        # sb_length = 10
        # sb_height = 1
        #ax.add_patch(mpatches.Rectangle(( sb_x, sb_y), sb_length, sb_height, color='white'))
        #ax.text(sb_x + sb_length /2, sb_y - 1*sb_height,  '100 nm', color='w', ha='center', va='bottom', backgroundcolor='black', fontsize=18)

        self.ax.legend(bbox_to_anchor=(0., 1.0, 1., .10), ncol=3,
                       handles=[red_patch, green_patch, blue_patch], mode="expand", loc=3)

        #self.fig.tight_layout(pad=4.0, w_pad=0.8, h_pad=0.8)
        #self.fig.tight_layout()
        #self.fig.canvas.draw_idle()
        self.fig.suptitle(self.img_title, fontsize=20)
        self.fig.canvas.draw_idle()

    def get_activated_num(self):
        return {k: v for (k, v) in six.iteritems(self.stat_dict) if v is True}


def norm_data(data):
    """
    Normalize data between (0,1).
    Parameters
    ----------
    data : 2D array
    """
    return (data-np.min(data))/(np.max(data)-np.min(data))

def make_cube(r, g, b):
    """
    Create 3D array for rgb image.
    Parameters
    ----------
    r : 2D array
    g : 2D array
    b : 2D array
    """
    ny, nx = r.shape
    R = np.zeros([ny, nx, 3])
    R[:,:,0] = r
    G = np.zeros_like(R)
    G[:,:,1] = g
    B = np.zeros_like(R)
    B[:,:,2] = b

    RGB = R + G + B

    return R, G, B, RGB
