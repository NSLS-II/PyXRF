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
import numpy as np


class Xsection_widget(FigureCanvas):
    def __init__(self, init_image, parent=None):
        width = height = 24
        self.fig = Figure(figsize=(width, height))
        # self.fig = Figure((24, 24), tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.xsection = images.xsection_viewer(self.fig, init_image)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

_CMAPS = datad.keys()
_CMAPS.sort()


class StackScanner(QtGui.QWidget):
    def __init__(self, stack, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        v_box_layout = QtGui.QVBoxLayout()

        self._stack = stack

        self._len = len(stack)

        self._axis_order = np.arange(self._stack.ndim)

        # get the shape of the stack so that the stack direction can be varied
        self._dims = stack.shape
        self.xsection_widget = Xsection_widget(stack[0])

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
        #        self._cm_cb.currentIndexChanged.connect(self.update_cmap)
        self._cm_cb.setEditText('gray')
        self._cm_cb.editTextChanged.connect(self.update_cmap)

        # set up intensity manipulation combo box
        intensity_behavior_types = ['none', 'percentile', 'absolute']
        intensity_behavior_funcs = [images._no_limit,
                                    images._percentile_limit,
                                    images._absolute_limit]
        self._intensity_behav_dict = {k: v for k, v in zip(
                                    intensity_behavior_types,
                                    intensity_behavior_funcs)}

        self._cmbbox_intensity_behavior = QtGui.QComboBox(parent=self)
        self._cmbbox_intensity_behavior.addItems(
                list(self._intensity_behav_dict.keys()))
        self._cmbbox_intensity_behavior.activated['QString'].connect(
                self.set_image_intensity_behavior)

        # set up intensity manipulation spin boxes
        # determine the initial values for the spin boxes
        self._min_intensity = min(stack[0].flatten())
        self._max_intensity = max(stack[0].flatten())
        self._intensity_step = (self._max_intensity -
                                self._min_intensity) / 100
        if self._intensity_step != 0:
            self._num_decimals = int(round(1.0 / self._intensity_step))
        else:
            self._intensity_step = 0.01
            self._num_decimals = 2

        # create the intensity manipulation spin boxes
        self._spinbox_min_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_max_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_intensity_step = QtGui.QDoubleSpinBox(parent=self)

        self._spinbox_min_intensity.setPrefix("min:  ")
        self._spinbox_max_intensity.setPrefix("max:  ")
        self._spinbox_intensity_step.setPrefix("step:  ")

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

    @QtCore.pyqtSlot(str)
    def set_image_intensity_behavior(self, im_behavior):
        # get parameters from spin boxes for min and max
        print(im_behavior)

        limit_func = self._intensity_behav_dict[str(im_behavior)]
        self.xsection_widget.xsection.set_limit_func(limit_func)

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
        chars = list(str_intensity_step)
        num_chars = len(chars)
        decimal_pos = str_intensity_step.find(".")
        num_decimals = num_chars - decimal_pos - 1
        last_decimal = int(chars[len(chars) - 1])
        if last_decimal != 0:
            self._num_decimals = num_decimals + 1
            self._spinbox_intensity_step.setDecimals(self._num_decimals)
            self._spinbox_min_intensity.setDecimals(self._num_decimals)
            self._spinbox_max_intensity.setDecimals(self._num_decimals)

    @QtCore.Slot(float)
    def set_min_intensity_limit(self, min_intensity):
        self._min_intensity = min_intensity
        _max = int(round(self._max_intensity / self._intensity_step))
        _min = int(round(self._min_intensity / self._intensity_step))
        if not _max > _min:
            self._max_intensity = self._min_intensity + self._intensity_step
            self._spinbox_max_intensity.setValue(self._max_intensity)
            self.set_max_intensity_limit(self._max_intensity)

        self.xsection_widget.xsection.set_min_limit(self._min_intensity)

    @QtCore.Slot(float)
    def set_max_intensity_limit(self, max_intensity):
        self._max_intensity = max_intensity
        _max = int(round(self._max_intensity / self._intensity_step))
        _min = int(round(self._min_intensity / self._intensity_step))
        if not _max > _min:
            self._min_intensity = self._max_intensity - self._intensity_step
            self._spinbox_min_intensity.setValue(self._min_intensity)
            self.set_min_intensity_limit(self._spinbox_min_intensity.value())

        self.xsection_widget.xsection.set_max_limit(self._max_intensity)

    def set_img_stack(self, img_stack):
        """
        Give the widget a new image stack without remaking the widget.
        Only call this after the widget has been constructed.  In
        other words, don't call this from the __init__ method
        Parameters
        ----------
        img_stack: stack of 2D ndarray
        """
        if img_stack is not None:
            self.stack = img_stack
            self.update_frame(0)

    @QtCore.Slot(int)
    def update_frame(self, n):
        self.xsection_widget.xsection.update_image(self._stack[n])

    @QtCore.Slot(str)
    def update_cmap(self, cmap_name):
        try:
            self.xsection_widget.xsection.update_colormap(cmap_name)
        except ValueError:
            pass
