from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from . import images

# grab the version from mpl which has done the work of smoothing over
# the differences
from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure
from matplotlib.cm import datad
import matplotlib.colors
import numpy as np


class Xsection_widget(FigureCanvas):
    """
    This is a thin wrapper around images.xsection_viewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """
    def __init__(self, init_image, parent=None):
        width = height = 24
        self.fig = Figure(figsize=(width, height))
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self._xsection = images.xsection_viewer(self.fig, init_image)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    @QtCore.Slot(str)
    def sl_update_color_map(self, cmap):
        """
        Updates the color map.  Currently takes a string, should probably be
        redone to take a cmap object and push the look up function up a layer
        so that we do not need the try..except block.
        """
        try:
            self._xsection.update_colormap(str(cmap))
        except ValueError:
            pass

    @QtCore.Slot(np.ndarray)
    def sl_update_image(self, img):
        """
        updates the image shown in the widget, assumed to be the same size
        """
        self._xsection.update_image(img)

    @QtCore.Slot(np.ndarray)
    def sl_replace_image(self, img):
        """
        Replaces the image shown in the widget, rebulids everything
        (so swap axis will work)
        """
        raise NotImplementedError()

    @QtCore.Slot(matplotlib.colors.Normalize)
    def sl_update_norm(self, new_norm):
        """
        Updates the normalization function used for the color mapping
        """
        self._xsection.update_norm(new_norm)

#    @QtCore.Slot(object, tuple)
    def sl_update_limit_func(self, limit_func, new_limits):
        """
        Updates the type of limit computation function used
        """
        self._xsection.set_limit_func(limit_func, new_limits)

    @QtCore.Slot(tuple)
    def sl_update_color_limits(self, new_limits):
        """
        Update the values passed to the limit computation function
        """
        self._xsection.update_color_limits(new_limits)


_CMAPS = datad.keys()
_CMAPS.sort()


class StackScanner(QtGui.QWidget):
    # set up the signals
    sig_update_cmap = QtCore.Signal(str)
    sig_update_image = QtCore.Signal(np.ndarray)
    sig_update_norm = QtCore.Signal(matplotlib.colors.Normalize)
    sig_update_limit_function = QtCore.Signal(object, tuple)
    sig_update_color_limits = QtCore.Signal(tuple)

    def __init__(self, stack, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        v_box_layout = QtGui.QVBoxLayout()

        self._stack = stack

        self._len = len(stack)

        self._axis_order = np.arange(self._stack.ndim)

        # get the shape of the stack so that the stack direction can be varied
        self._dims = stack.shape
        # create the viewer widget
        self.xsection_widget = Xsection_widget(stack[0])

        # connect up the signals/slots to boss the viewer around
        self.sig_update_cmap.connect(self.xsection_widget.sl_update_color_map)
        self.sig_update_image.connect(self.xsection_widget.sl_update_image)
        self.sig_update_norm.connect(self.xsection_widget.sl_update_norm)
        self.sig_update_limit_function.connect(
            self.xsection_widget.sl_update_limit_func)
        self.sig_update_color_limits.connect(
            self.xsection_widget.sl_update_color_limits)

        # ---- set up widget box 1---------------------------------------------
        # --------- it has: ---------------------------------------------------
        # -------------- axis swap buttons ------------------------------------
        # -------------- slider to change images ------------------------------
        # -------------- spinbox to change images -----------------------------

        # set up axis swap buttons
        self._cb_ax1 = QtGui.QComboBox(parent=self)
        self._cb_ax2 = QtGui.QComboBox(parent=self)
        self._cb_ax1.setEditable(False)
        self._cb_ax2.setEditable(False)
        self._cb_ax1.addItems(np.arange(self._stack.ndim).astype(str))
        self._cb_ax2.addItems(np.arange(self._stack.ndim).astype(str))

        self._btn_swap_ax = QtGui.QPushButton('Swap Axes', parent=self)
        self._btn_swap_ax.resize(self._btn_swap_ax.sizeHint())
        self._btn_swap_ax.clicked.connect(self.swap_stack_axes)
        self._btn_swap_ax.setEnabled(False)

        # set up slider
        self._slider = QtGui.QSlider(parent=self)
        self._slider.setRange(0, self._len - 1)
        self._slider.setTracking(True)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(page_size)
        self._slider.valueChanged.connect(self.update_frame)
        self._slider.setOrientation(QtCore.Qt.Horizontal)

        # and its spin box
        self._spinbox = QtGui.QSpinBox(parent=self)
        self._spinbox.setRange(self._slider.minimum(), self._slider.maximum())
        self._spinbox.valueChanged.connect(self._slider.setValue)
        self._slider.valueChanged.connect(self._spinbox.setValue)
        self._slider.rangeChanged.connect(self._spinbox.setRange)

        # construct widget box 1
        widget_box1_sub1 = QtGui.QHBoxLayout()
        widget_box1_sub1.addWidget(self._btn_swap_ax)
        widget_box1_sub1.addWidget(self._cb_ax1)
        widget_box1_sub1.addWidget(self._cb_ax2)

        widget_box1 = QtGui.QHBoxLayout()
        widget_box1.addLayout(widget_box1_sub1)
        widget_box1.addWidget(self._slider)
        widget_box1.addWidget(self._spinbox)

        # ---- set up widget box 2---------------------------------------------
        # --------- it has: ---------------------------------------------------
        # -------------- color map combo box ----------------------------------
        # -------------- intensity manipulation combo box ---------------------
        # -------------- spinboxes for intensity min/max/step values-----------

        # set up color map combo box
        self._cm_cb = QtGui.QComboBox(parent=self)
        self._cm_cb.setEditable(True)
        self._cm_cb.addItems(_CMAPS)

        self._cm_cb.setEditText('gray')
        self._cm_cb.editTextChanged[str].connect(
            self.xsection_widget.sl_update_color_map)

        # set up intensity manipulation combo box
        intensity_behavior_data = [(images._full_range,
                                     (0, 2**16),
                                     self._no_limit_config),
                                    (images._percentile_limit,
                                     (0, 100),
                                     self._percentile_config),
                                    (images._absolute_limit,
                                     (0, 2**16),
                                     self._absolute_limit_config)
                                     ]
        intensity_behavior_types = ['full range',
                                    'percentile',
                                    'absolute']
        self._intensity_behav_dict = {k: v for k, v in zip(
                                    intensity_behavior_types,
                                    intensity_behavior_data)}
        self._dflt_limits_dict = {k: v for k, v in zip(

)}
        self._cmbbox_intensity_behavior = QtGui.QComboBox(parent=self)
        self._cmbbox_intensity_behavior.addItems(
                list(self._intensity_behav_dict.keys()))
        self._cmbbox_intensity_behavior.activated[str].connect(
                self.set_image_intensity_behavior)

        # set up intensity manipulation spin boxes
        # determine the initial values for the spin boxes
        self._min_intensity = np.min(stack[0])
        self._max_intensity = np.max(stack[0])
        self._intensity_step = (self._max_intensity -
                                self._min_intensity) / 100

        # create the intensity manipulation spin boxes
        self._spinbox_min_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_max_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_intensity_step = QtGui.QDoubleSpinBox(parent=self)

        # allow the spin boxes to be any value
        self._spinbox_min_intensity.setMinimum(float("-inf"))
        self._spinbox_min_intensity.setMaximum(float("inf"))
        self._spinbox_max_intensity.setMinimum(float("-inf"))
        self._spinbox_max_intensity.setMaximum(float("inf"))
        self._spinbox_intensity_step.setMinimum(0)
        self._spinbox_intensity_step.setMaximum(float("inf"))

        # connect the intensity spinboxes to their updating method
        self._spinbox_min_intensity.valueChanged.connect(
                self.set_min_intensity_limit)
        self._spinbox_max_intensity.valueChanged.connect(
                self.set_max_intensity_limit)
        self._spinbox_intensity_step.valueChanged.connect(
                self.set_intensity_step)

        # set the initial values for the spin boxes
        self._spinbox_min_intensity.setValue(self._min_intensity)
        self._spinbox_max_intensity.setValue(self._max_intensity)
        self._spinbox_intensity_step.setValue(self._intensity_step)

        # construct widget box 2
        widget_box2 = QtGui.QHBoxLayout()
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(self._cm_cb)
        hbox2.addWidget(self._cmbbox_intensity_behavior)
        widget_box2.addLayout(hbox2)
        widget_box2.addWidget(self._spinbox_min_intensity)
        widget_box2.addWidget(self._spinbox_max_intensity)
        widget_box2.addWidget(self._spinbox_intensity_step)

        self.mpl_toolbar = NavigationToolbar(self.xsection_widget, self)
        # add toolbar
        v_box_layout.addWidget(self.mpl_toolbar)
        # add main widget
        v_box_layout.addWidget(self.xsection_widget)
        # add slider v_box_layout
        v_box_layout.addLayout(widget_box1)
        # add colormap selector and autonorm box
        v_box_layout.addLayout(widget_box2)
        self.setLayout(v_box_layout)

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
        self._btn_swap_ax.setToolTip(np.array_str(self._axis_order))
        self._stack = np.swapaxes(self._stack, axis1, axis2)
        self.set_img_stack(self._stack)
        print("stack.shape: {0}".format(self._stack.shape))
        self._len = self._stack.shape[0]
        self._slider.setRange(0, self._len - 1)
        self._spinbox.setRange(self._slider.minimum(), self._slider.maximum())

    @QtCore.Slot(str)
    def set_image_intensity_behavior(self, im_behavior):
        # get parameters from spin boxes for min and max
        print(im_behavior)

        (limit_func, limits,
              gui_fix_up) = self._intensity_behav_dict[str(im_behavior)]

        # updates the underlying object
        self.sig_update_limit_function.emit(limit_func, limits)
        # fixes the gui state
        gui_fix_up()
        # set the new limits
        self._set_spinbox_limits(*limits)

    def _spinbox_enabler(self, state):
        self._spinbox_max_intensity.setEnabled(state)
        self._spinbox_min_intensity.setEnabled(state)
        self._spinbox_intensity_step.setEnabled(state)

    def _no_limit_config(self):
        """
        Helper function to set up the gui for the 'no limit'
        (max/min) color bounds
        """
        # turn off the spin boxes
        self._spinbox_enabler(False)

    def _percentile_config(self):
        """
        helper function to set up the gui for use with the percentile
        color bounds
        """
        self._spinbox_enabler(True)

    def _absolute_limit_config(self):
        """
        Helper function to set up the gui for use with absolute limits
        """
        self._spinbox_enabler(True)

    def _set_spinbox_limits(self, bottom_val, top_val):
        # set the top and bottom limits on the spinboxs to be in bounds
        self._spinbox_max_intensity.setMinimum(bottom_val)
        self._spinbox_min_intensity.setMinimum(bottom_val)

        self._spinbox_max_intensity.setMaximum(top_val)
        self._spinbox_min_intensity.setMaximum(top_val)
        # don't let the step be bigger than the total allowed range
        self._spinbox_intensity_step.setMaximum(top_val - bottom_val)

        if not np.isinf(bottom_val) or not np.isinf(top_val):
            # set the current values
            self._spinbox_min_intensity.setValue(bottom_val)
            self._spinbox_max_intensity.setValue(top_val)

            # this will trigger via the call-back updating everything else
            self._spinbox_intensity_step.setValue((top_val - bottom_val)/100)

    @QtCore.Slot(float)
    def set_intensity_step(self, intensity_step):
        """
        Slot method for the intensity step spinbox valueChanged() method.
        The intensity_step is passed as a string which needs to be parsed into
        """
        # set the intensity steps for each of the combo boxes
        self._intensity_step = intensity_step
        self._spinbox_intensity_step.setSingleStep(self._intensity_step)
        self._spinbox_max_intensity.setSingleStep(self._intensity_step)
        self._spinbox_min_intensity.setSingleStep(self._intensity_step)

        # parse the currently displayed string to determine if the last digit
        # is non-zero.  If it is, increase the number of displayed decimal
        # places by 1
        str_intensity_step = str(self._intensity_step)
        num_decimals = len(str_intensity_step.split('.')[-1])
        last_decimal = str_intensity_step[-1]
        if last_decimal != 0:
            self._spinbox_intensity_step.setDecimals(num_decimals + 1)
            self._spinbox_min_intensity.setDecimals(num_decimals + 1)
            self._spinbox_max_intensity.setDecimals(num_decimals + 1)

    @QtCore.Slot(float)
    def set_min_intensity_limit(self, min_intensity):
        # grab the max value
        max_intensity = self._spinbox_max_intensity.value()
        # grab the step value
        intensity_step = self._spinbox_intensity_step.value()
        # covert max/min to number of steps
        _max = int(round(max_intensity / intensity_step))
        _min = int(round(min_intensity / intensity_step))
        # if max is not atleast a step greater than min, adjust
        if not _max > _min:
            max_intensity = min_intensity + intensity_step
            # this should take care of the call back to the viewer
            self._spinbox_max_intensity.setValue(max_intensity)
        else:
            self.sig_update_color_limits.emit((min_intensity, max_intensity))

    @QtCore.Slot(float)
    def set_max_intensity_limit(self, max_intensity):
        # grab the max value
        min_intensity = self._spinbox_min_intensity.value()
        # grab the step value
        intensity_step = self._spinbox_intensity_step.value()

        _max = int(round(self._max_intensity / self._intensity_step))
        _min = int(round(self._min_intensity / self._intensity_step))
        if not _max > _min:
            min_intensity = max_intensity - intensity_step
            self._spinbox_min_intensity.setValue(min_intensity)
        else:
            self.sig_update_color_limits.emit((min_intensity, max_intensity))

    @QtCore.Slot(float, float)
    def set_limits(self, bottom, top):
        self.sig_update_color_limits.emit((bottom, top))

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
            self.update_frame(0)

    @QtCore.Slot(int)
    def update_frame(self, n):
        self.sig_update_image.emit(self._stack[n])
