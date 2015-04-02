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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from .. import QtCore, QtGui
from matplotlib import colors
from matplotlib.cm import datad


import numpy as np

from . import AbstractMPLMessenger
from .. import AbstractMessenger2D
from ...backend.mpl.cross_section_2d import CrossSection2DView
from ...backend.mpl import cross_section_2d as View
from ...backend.mpl import AbstractMPLDataView

import logging
logger = logging.getLogger(__name__)


class CrossSection2DMessenger(AbstractMessenger2D, AbstractMPLMessenger):
    """
    This is a thin wrapper around mpl.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """

    def __init__(self, data_list, key_list, parent=None, *args, **kwargs):
        # call up the inheritance chain
        super(CrossSection2DMessenger, self).__init__(*args, **kwargs)
        # init the appropriate view
        self._view = CrossSection2DView(fig=self._fig, data_list=data_list,
                                        key_list=key_list)

        # TODO: Address issue of data storage in the cross section widget
        self._ctrl_widget = CrossSection2DControlWidget(name="2-D CrossSection"
                                                             " Controls",
                                                        init_img=data_list[0],
                                                        num_images=len(
                                                            key_list))
        # connect signals to slots
        self.connect_sigs_to_slots()

    def connect_sigs_to_slots(self):
        """
        Connect the signals of the control box to the slots of the messenger
        """
        # standard data manipulation signal/slot pairs
        # TODO Fix this connection. It throws an exception b/c the connection fails
        self._ctrl_widget.sig_update_norm.connect(self.sl_update_norm)

        # standard mpl signal/slot pairs
        self._ctrl_widget._cm_cb.editTextChanged[str].connect(
            self.sl_update_cmap)
        self._ctrl_widget._cm_cb.setEditText(self._ctrl_widget.default_cmap)

        # signal/slot pairs specific to the CrossSection2DView
        self._ctrl_widget.sig_update_limit_function.connect(
            self.sl_update_limit_func)

        self._ctrl_widget._slider_img.valueChanged.connect(self.sl_update_image)
        self._ctrl_widget.sig_update_interpolation.connect(
            self._view.update_interpolation)

    @QtCore.Slot(int)
    def sl_update_image(self, img_idx):
        """
        updates the image shown in the widget, assumed to be the same size
        """
        self._view.update_image(img_idx)
        self.sl_update_view()
        im = self._view._data_dict[self._view._key_list[img_idx]]
        self._ctrl_widget.set_im_lim(lo=np.min(im), hi=np.max(im))

    @QtCore.Slot(np.ndarray)
    def sl_replace_image(self, img):
        """
        Replaces the image shown in the widget, rebulids everything
        (so swap axis will work)
        """
        raise NotImplementedError()

    @QtCore.Slot(object)
    def sl_update_limit_func(self, limit_func):
        """
        Updates the type of limit computation function used
        """
        self._view.set_limit_func(limit_func)
        self.sl_update_view()


class CrossSection2DControlWidget(QtGui.QDockWidget):
    """
    This object contains the CrossSectionViewer (2D Image Display) and
    finish the doc string...
    """
    # set up the signals
    sig_update_image = QtCore.Signal(int)
    sig_update_norm = QtCore.Signal(colors.Normalize)
    sig_update_limit_function = QtCore.Signal(object)
    sig_update_interpolation = QtCore.Signal(str)

    # some defaults
    default_cmap = AbstractMPLDataView._default_cmap

    _CMAPS = datad.keys()
    _CMAPS.sort()

    def __init__(self, name, init_img, num_images):
        QtGui.QDockWidget.__init__(self, name)
        # make the control widget float
        self.setFloating(True)

        # add a widget that lives in the floating control widget
        self._widget = QtGui.QWidget(self)
        # give the widget to the dock widget
        self.setWidget(self._widget)
        # create a layout
        ctrl_layout = QtGui.QVBoxLayout()
        # set the layout to the widget
        self._widget.setLayout(ctrl_layout)

        self._axis_order = np.arange(init_img.ndim+1)
        self._lo = np.min(init_img)
        self._hi = np.max(init_img)

        # set up axis swap buttons
        self._cb_ax1 = QtGui.QComboBox(parent=self)
        self._cb_ax2 = QtGui.QComboBox(parent=self)
        self._btn_swap = QtGui.QPushButton('Swap Axes', parent=self)
        self.init_swap_btns(self._cb_ax1, self._cb_ax2, self._btn_swap)

        # set up slider and spinbox
        self._slider_img = QtGui.QSlider(parent=self)
        self._spin_img = QtGui.QSpinBox(parent=self)
        # init the slider and spinbox
        self.init_img_changer(self._slider_img, self._spin_img, num_images)

        widget_box1 = QtGui.QVBoxLayout()
        slider_label = QtGui.QLabel("&Frame")
        slider_label.setBuddy(self._slider_img)

        widget_box1_hbox = QtGui.QHBoxLayout()
        widget_box1_hbox.addWidget(self._slider_img)
        widget_box1_hbox.addWidget(self._spin_img)
        widget_box1.addWidget(slider_label)
        widget_box1.addLayout(widget_box1_hbox)

        # set up color map combo box
        self._cm_cb = QtGui.QComboBox(parent=self)
        self.init_cmap_box(self._cm_cb)

        # set up the interpolation combo box
        self._cmb_interp = QtGui.QComboBox(parent=self)
        self._cmb_interp.addItems(CrossSection2DView.interpolation)

        # set up intensity manipulation combo box
        intensity_behavior_data = [(View.fullrange_limit_factory,
                                    self._no_limit_config),
                                   (View.percentile_limit_factory,
                                    self._percentile_config),
                                   (View.absolute_limit_factory,
                                    self._absolute_limit_config)]
        intensity_behavior_types = ['full range',
                                    'percentile',
                                    'absolute']
        self._intensity_behav_dict = {k: v for k, v in zip(
                                      intensity_behavior_types,
                                      intensity_behavior_data)}
        # TODO should not have to hard-code this, but it is getting
        # called before it is fully updated, figure out why
        self._limit_factory = View.fullrange_limit_factory
        self._cmbbox_intensity_behavior = QtGui.QComboBox(parent=self)
        self._cmbbox_intensity_behavior.addItems(intensity_behavior_types)

        # can add PowerNorm, BoundaryNorm, but those require extra inputs
        norm_names = ['linear', 'log']
        norm_funcs = [colors.Normalize, colors.LogNorm]
        self._norm_dict = {k: v for k, v in zip(norm_names, norm_funcs)}
        self._cmbbox_norm = QtGui.QComboBox(parent=self)
        self._cmbbox_norm.addItems(norm_names)

        # set up intensity manipulation spin boxes
        # create the intensity manipulation spin boxes
        self._spin_min = QtGui.QDoubleSpinBox(parent=self)
        self._spin_max = QtGui.QDoubleSpinBox(parent=self)
        self._spin_step = QtGui.QDoubleSpinBox(parent=self)
        self.init_spinners(self._spin_min, self._spin_max, self._spin_step,
                           min_intensity=np.min(init_img),
                           max_intensity=np.max(init_img))

        ctrl_form = QtGui.QFormLayout()
        ctrl_form.addRow("Color &map", self._cm_cb)
        ctrl_form.addRow("&Interpolation", self._cmb_interp)
        ctrl_form.addRow("&Normalization", self._cmbbox_norm)
        ctrl_form.addRow("limit &strategy", self._cmbbox_intensity_behavior)
        ctrl_layout.addLayout(ctrl_form)

        clim_spinners = QtGui.QGroupBox("clim parameters")
        ispiner_form = QtGui.QFormLayout()
        ispiner_form.addRow("mi&n", self._spin_min)
        ispiner_form.addRow("ma&x", self._spin_max)
        ispiner_form.addRow("s&tep", self._spin_step)
        clim_spinners.setLayout(ispiner_form)
        ctrl_layout.addWidget(clim_spinners)

        # construct widget box 1
        widget_box1_sub1 = QtGui.QVBoxLayout()
        axes_swap_form = QtGui.QFormLayout()
        axes_swap_form.addRow("axes A", self._cb_ax1)
        axes_swap_form.addRow("axes B", self._cb_ax2)
        widget_box1_sub1.addLayout(axes_swap_form)
        widget_box1_sub1.addWidget(self._btn_swap)
        swap_axes_box = QtGui.QGroupBox("Swap!")
        swap_axes_box.setLayout(widget_box1_sub1)
        swap_axes_box.setEnabled(False)
        ctrl_layout.addWidget(swap_axes_box)
        ctrl_layout.addLayout(widget_box1)
        ctrl_layout.addStretch()

        # set this down here to make sure the function will run
        self._cmbbox_intensity_behavior.currentIndexChanged[str].connect(
            self.set_image_intensity_behavior)
        # set to full range, do this last so all the call-back propagate
        self._cmbbox_intensity_behavior.setCurrentIndex(0)
        # force the issue about emitting
        self._cmbbox_intensity_behavior.currentIndexChanged[str].emit(
            intensity_behavior_types[0])

        # set this down here to make sure the function will run
        self._cmbbox_norm.currentIndexChanged[str].connect(
            self.set_normalization)
        # set to full range, do this last so all the call-back propagate
        self._cmbbox_norm.setCurrentIndex(0)
        # force the issue about emitting
        self._cmbbox_norm.currentIndexChanged[str].emit(
            norm_names[0])
        self._cmb_interp.currentIndexChanged[str].connect(
            self.sig_update_interpolation)

    def set_im_lim(self, lo, hi):
        self._lo = lo
        self._hi = hi

    def init_img_changer(self, slider_img, spin_img, num_images):
        slider_img.setRange(0, num_images - 1)
        slider_img.setTracking(True)
        slider_img.setSingleStep(1)
        slider_img.setPageStep(10)
        slider_img.setOrientation(QtCore.Qt.Horizontal)
        spin_img.setRange(slider_img.minimum(), slider_img.maximum())
        spin_img.valueChanged.connect(slider_img.setValue)
        slider_img.valueChanged.connect(spin_img.setValue)
        slider_img.rangeChanged.connect(spin_img.setRange)

    def init_spinners(self, spin_min, spin_max, spin_step, min_intensity,
                      max_intensity):
        # allow the spin boxes to be any value
        spin_min.setMinimum(float("-inf"))
        spin_min.setMaximum(float("inf"))
        spin_max.setMinimum(float("-inf"))
        spin_max.setMaximum(float("inf"))
        spin_step.setMinimum(0)
        spin_step.setMaximum(float("inf"))

        # connect the intensity spinboxes to their updating method
        spin_min.valueChanged.connect(
            self.set_min_intensity_limit)
        spin_max.valueChanged.connect(
            self.set_max_intensity_limit)
        spin_step.valueChanged.connect(
            self.set_intensity_step)

        # set the initial values for the spin boxes
        spin_step.setValue((max_intensity-min_intensity)/100)
        spin_max.setValue(max_intensity)
        spin_min.setValue(min_intensity)

    def init_swap_btns(self, cb_ax1, cb_ax2, btn_swap):
        cb_ax1.setEditable(False)
        cb_ax2.setEditable(False)
        # TODO need to deal with changing the items in this combobox when the data is changed
        cb_ax1.addItems(np.arange(len(self._axis_order)).astype(str))
        cb_ax2.addItems(np.arange(len(self._axis_order)).astype(str))
        btn_swap.resize(btn_swap.sizeHint())
        btn_swap.clicked.connect(self.swap_stack_axes)
        btn_swap.setEnabled(False)

    def init_cmap_box(self, cm_cb):
        cm_cb.setEditable(True)
        cm_cb.addItems(self._CMAPS)
        cm_cb.setEditText(self.default_cmap)

    def swap_stack_axes(self):
        """
        Swap the axes of the image stack based on the indices of the combo
        boxes. The tooltip of the axis swap button maintains the current
        position of the axes based on where they began.

        e.g., the tooltip will start as [0 1 2] if a 3d array is passed.
            If axes 0 and 2 are swapped, the tooltip will now read [2 1 0].
        """
        axis1 = self._cb_ax1.currentIndex()
        axis2 = self._cb_ax2.currentIndex()
        cur_axis1 = self._axis_order[axis1]
        cur_axis2 = self._axis_order[axis2]
        self._axis_order[axis1] = cur_axis2
        self._axis_order[axis2] = cur_axis1
        self._btn_swap.setToolTip(np.array_str(self._axis_order))
        self._stack = np.swapaxes(self._stack, axis1, axis2)
        self.set_img_stack(self._stack)
        print("stack.shape: {0}".format(self._stack.shape))
        self._len = self._stack.shape[0]
        self._slider_img.setRange(0, self._len - 1)
        self._spin_img.setRange(self._slider_img.minimum(),
                                self._slider_img.maximum())

    @QtCore.Slot(str)
    def set_normalization(self, norm_name):
        norm = self._norm_dict[str(norm_name)]
        self.sig_update_norm.emit(norm())

    @QtCore.Slot(str)
    def set_image_intensity_behavior(self, im_behavior):
        # get the limit factory to use
        (limit_fac, get_params) = self._intensity_behav_dict[str(im_behavior)]
        # stash the limit function factory for later use
        self._limit_factory = limit_fac
        # fixes the gui state, grabs default spinner values + spinner state
        limits, state = get_params()
        # make the limit function
        limit_func = limit_fac(limits)
        # emit the function to be passed on to the underlying object
        self.sig_update_limit_function.emit(limit_func)
        # set the new limits
        self._set_spinbox_limits(*limits)
        self._spinbox_enabler(state)

    def _spinbox_enabler(self, state):
        self._spin_max.setEnabled(state)
        self._spin_min.setEnabled(state)
        self._spin_step.setEnabled(state)

    def _no_limit_config(self):
        """
        Helper function to set up the gui for the 'no limit'
        (max/min) color bounds
        """
        # turn off the spin boxes
        # just echo back what it is and don't change it
        return (self._spin_min.value(),
                self._spin_max.value()), False

    def _percentile_config(self):
        """
        helper function to set up the gui for use with the percentile
        color bounds
        """
        # return full range
        return (0, 100), True

    def _absolute_limit_config(self):
        """
        Helper function to set up the gui for use with absolute limits
        """
        return (self._lo, self._hi), True

    def _set_spinbox_limits(self, bottom_val, top_val):
        # turn off signals on the spin boxes
        reset_state = [(sb, sb.blockSignals(True)) for sb in
                       (self._spin_max,
                        self._spin_min)]
        try:
            # set the top and bottom limits on the spinboxs to be in bounds
            self._spin_max.setMinimum(bottom_val)
            self._spin_min.setMinimum(bottom_val)

            self._spin_max.setMaximum(top_val)
            self._spin_min.setMaximum(top_val)
            # don't let the step be bigger than the total allowed range
            self._spin_step.setMaximum(top_val - bottom_val)

            if not np.isinf(bottom_val) or not np.isinf(top_val):
                # set the current values
                self._spin_min.setValue(bottom_val)
                self._spin_max.setValue(top_val)

                # this will trigger via the call-back updating everything else
                self._spin_step.setValue(
                    (top_val - bottom_val) / 100)
        finally:
            # un-wrap the signal blocking
            [sb.blockSignals(state) for sb, state in reset_state]

    @QtCore.Slot(float)
    def set_intensity_step(self, intensity_step):
        """
        Slot method for the intensity step spinbox valueChanged() method.
        The intensity_step is passed as a string which needs to be parsed into
        """
        # set the intensity steps for each of the combo boxes
        self._spin_step.setSingleStep(intensity_step)
        self._spin_max.setSingleStep(intensity_step)
        self._spin_min.setSingleStep(intensity_step)

        # parse the currently displayed string to determine if the last digit
        # is non-zero.  If it is, increase the number of displayed decimal
        # places by 1
        str_intensity_step = str(intensity_step)
        num_decimals = len(str_intensity_step.split('.')[-1])
        last_decimal = str_intensity_step[-1]
        if last_decimal != 0:
            self._spin_step.setDecimals(num_decimals + 1)
            self._spin_min.setDecimals(num_decimals + 1)
            self._spin_max.setDecimals(num_decimals + 1)

    @QtCore.Slot(float)
    def set_min_intensity_limit(self, min_intensity):
        # grab the max value
        max_intensity = self._spin_max.value()
        # grab the step value
        intensity_step = self._spin_step.value()
        # covert max/min to number of steps
        _max = int(round(max_intensity / intensity_step))
        _min = int(round(min_intensity / intensity_step))
        # if max is not atleast a step greater than min, adjust
        if not _max > _min:
            max_intensity = min_intensity + intensity_step
            # this should take care of the call back to the viewer
            self._spin_max.setValue(max_intensity)
        else:
            limit_func = self._limit_factory((min_intensity, max_intensity))
            self.sig_update_limit_function.emit(limit_func)

    @QtCore.Slot(float)
    def set_max_intensity_limit(self, max_intensity):
        # grab the max value
        min_intensity = self._spin_min.value()
        # grab the step value
        intensity_step = self._spin_step.value()

        _max = int(round(max_intensity / intensity_step))
        _min = int(round(min_intensity / intensity_step))
        if not _max > _min:
            min_intensity = max_intensity - intensity_step
            self._spin_min.setValue(min_intensity)
        else:
            limit_func = self._limit_factory((min_intensity, max_intensity))
            self.sig_update_limit_function.emit(limit_func)

    @QtCore.Slot(float, float)
    def set_limits(self, bottom, top):
        # TODO update the spinners + validate
        limit_func = self._limit_factory((bottom, top))
        self.sig_update_limit_function.emit(limit_func)

    def set_img_stack(self, img_stack):
        """
        Give the widget a new image stack without remaking the widget.
        Only call this after the widget has been constructed.  In
        other words, don't call this from the __init__ method

        Parameters
        ----------
        img_stack: anything that returns a 2D array when __getitem__ is called
        """
        if img_stack is not None:
            self.stack = img_stack
            self._view.sl_update_image(0)

    @QtCore.Slot(int)
    def update_frame(self, frame_idx):
        self.sig_update_image.emit(frame_idx)
