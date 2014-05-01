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
        layout = QtGui.QVBoxLayout()

        self._stack = stack

        self._len = len(stack)
        self.xsection_widget = Xsection_widget(stack[0])

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

        # make slider layout
        slider_layout = QtGui.QHBoxLayout()
        slider_layout.addWidget(self._slider)
        slider_layout.addWidget(self._spinbox)

        # ---- color map combo box --------------------------------------------
        self._cm_cb = QtGui.QComboBox()
        self._cm_cb.setEditable(True)
        self._cm_cb.addItems(_CMAPS)
        #        self._cm_cb.currentIndexChanged.connect(self.update_cmap)
        self._cm_cb.setEditText('gray')
        self._cm_cb.editTextChanged.connect(self.update_cmap)

        # ---- intensity manipulation combo box--------------------------------
        intensity_behavior_types = ['none', 'percentile', 'absolute']
        intensity_behavior_funcs = [images._no_limit, \
                                    images._percentile_limit, \
                                    images._absolute_limit]
        self._intensity_behav_dict = \
            {k: v for k, v in zip(intensity_behavior_types, \
                                 intensity_behavior_funcs)}

        self._cmbbox_intensity_behavior = QtGui.QComboBox()
        self._cmbbox_intensity_behavior.addItems(
                list(self._intensity_behav_dict.keys()))
        self._cmbbox_intensity_behavior.activated['QString'].connect(
                self.set_image_intensity_behavior)

        # ---- intensity manipulation spin boxes ------------------------------
        # determine the initial values for the spin boxes
        self._min_intensity = min(stack[0].flatten())
        self._max_intensity = max(stack[0].flatten())
        self._intensity_step = (self._max_intensity - self._min_intensity) / 100
        self._num_decimals = int(round(1.0 / self._intensity_step))

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

        # combine color map selector and auto-norm button
        hbox = QtGui.QHBoxLayout()
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(self._cm_cb)
        hbox2.addWidget(self._cmbbox_intensity_behavior)
        hbox.addLayout(hbox2)
        hbox.addWidget(self._spinbox_min_intensity)
        hbox.addWidget(self._spinbox_max_intensity)
        hbox.addWidget(self._spinbox_intensity_step)

        self.mpl_toolbar = NavigationToolbar(self.xsection_widget, self)
        # add toolbar
        layout.addWidget(self.mpl_toolbar)
        # add main widget
        layout.addWidget(self.xsection_widget)
        # add slider layout
        layout.addLayout(slider_layout)
        # add colormap selector and autonorm box
        layout.addLayout(hbox)
        self.setLayout(layout)

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
        self.intensity_step = intensity_step
        self._spinbox_intensity_step.setSingleStep(self.intensity_step)
        self._spinbox_max_intensity.setSingleStep(self.intensity_step)
        self._spinbox_min_intensity.setSingleStep(self.intensity_step)

        # parse the currently displayed string to determine if the last digit
        # is non-zero.  If it is, increase the number of displayed decimal
        # places by 1
        str_intensity_step = str(intensity_step)
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
        Give the widget a new image stack without remaking the widget
        Parameters
        ----------
        img_stack: stack of 2D ndarray
        """
        if img_stack is not None:
            self.stack = img_stack
            self.init_widgets(self.img_stack[0])
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

    def compute_decimals_to_show(self, value):
        """
        Compute the number of decimals to show in the intensity spin boxes.
        The step spin box will have two more decimals shown than the
        min/max spin boxes
        ----------
        Parameters
        ----------
        value: The current value of the intensity_step
        """
        if value > 0:
            num_decimals = int(np.log10(round(1.0 / value))) + 2
        else:
            num_decimals = 2
        self._spinbox_intensity_step.setDecimals(num_decimals)
        self._spinbox_min_intensity.setDecimals(num_decimals)
        self._spinbox_max_intensity.setDecimals(num_decimals)
