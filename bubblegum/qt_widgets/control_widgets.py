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
from collections import defaultdict
from .util import mapping_mixin
from .. import QtCore, QtGui

import logging
logger = logging.getLogger(__name__)

_defaults = {
    "check_box_hover_text": "Enable this widget",
    "check_box_state": True,
}


class UtilsWidget(QtGui.QWidget):
    """
    `UtilsWidget` has a `QHBoxLayout`, an en/disable checkbox and a label
    with hover text.  Daughter classes of `UtilsWidget` just need to
    add their respective input box to self._layout and set the layout with
    :code:`to self.setLayout(self._layout)`

    """
    def __init__(self, label_text, hover_text=None, has_check_box=False):
        super(UtilsWidget, self).__init__()
        # set the defaults
        if hover_text is None:
            hover_text = label_text
        # make the label
        self._lab = QtGui.QLabel(label_text)
        # set the text to display on mouse cursor hover
        self._lab.setToolTip(hover_text)

        # make layout
        self._layout = QtGui.QHBoxLayout()

        # make the check box if it is needed
        self._has_check_box = has_check_box
        if self._has_check_box:
            self._check_box = QtGui.QCheckBox()
            self._check_box.setToolTip(_defaults["check_box_hover_text"])
            self._check_box.setChecked(_defaults["check_box_state"])
            # todo disable input when the check_box is not checked
            self._layout.addWidget(self._check_box)

        self._layout.addWidget(self._lab)


class Slider(UtilsWidget):
    """
    Fancier version of a slider which includes a label and
    a spinbox
    """
    # export sub-set of slider signals
    # this should be exhaustive eventually
    valueChanged = QtCore.Signal(int)
    rangeChanged = QtCore.Signal(int, int)

    # todo make more things configurable
    def __init__(self, label_text, min_v, max_v, tracking=True,
                 hover_text=None, has_check_box=False):
        super(Slider, self).__init__(label_text=label_text,
                                     hover_text=hover_text,
                                     has_check_box=has_check_box)

        # set up slider
        self._slider = QtGui.QSlider(parent=self)
        self._slider.setRange(min_v, max_v)
        self._slider.setTracking(tracking)
        self._slider.setSingleStep(1)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        # internal connections
        self._slider.valueChanged.connect(self.valueChanged)
        self._slider.rangeChanged.connect(self.rangeChanged)
        # make buddy with label
        self._label.setBuddy(self._slider)

        # and its spin box
        self._spinbox = QtGui.QSpinBox(parent=self)
        self._spinbox.setRange(self._slider.minimum(), self._slider.maximum())
        self._spinbox.valueChanged.connect(self._slider.setValue)
        self._slider.valueChanged.connect(self._spinbox.setValue)
        self._slider.rangeChanged.connect(self._spinbox.setRange)

        # add widegts
        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._spinbox)

        self.setLayout(self._layout)

    # TODO make sure all the slots are included
    @QtCore.Slot(int)
    def setValue(self, val):
        # internal call backs will take care of the spinbox
        self._slider.setValue(val)


class DateTimeBox(UtilsWidget):
    dateChanged = QtCore.Signal(QtCore.QDate)
    dateTimeChanged = QtCore.Signal(QtCore.QDateTime)
    timeChanged = QtCore.Signal(QtCore.QTime)

    # todo make more things configurable
    def __init__(self, label_text, hover_text=None, has_check_box=False):
        # pass up the stack
        super(DateTimeBox, self).__init__(label_text=label_text,
                                          hover_text=hover_text,
                                          has_check_box=has_check_box)
        # make the date time box
        self._datetime = QtGui.QDateTimeEdit(QtCore.QDate.currentDate())
        self._datetime.setCalendarPopup(True)
        self._datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        # buddy them up
        self._lab.setBuddy(self._datetime)
        # add the datetime widget to the layout
        self._layout.addWidget(self._datetime)

        # set the widget's layout
        self.setLayout(self._layout)

        # connect the signals
        self._datetime.dateChanged.connect(self.dateChanged)
        self._datetime.dateTimeChanged.connect(self.dateTimeChanged)
        self._datetime.timeChanged.connect(self.timeChanged)

    # forward the slots
    @QtCore.Slot(QtCore.QDate)
    def setDate(self, date):
        self._datetime.setDate(date)

    @QtCore.Slot(QtCore.QDateTime)
    def setDateTime(self, dateTime):
        self._datetime.setDateTime(dateTime)

    @QtCore.Slot(QtCore.QTime)
    def setTime(self, time):
        self._datetime.setTime(time)

    def getValue(self):
        if self._has_check_box and self._check_box.isChecked():
            try:
                return self._datetime.dateTime().toPython()
            except AttributeError :
                return self._datetime.dateTime().toPyDateTime()

        return None


class ComboBox(UtilsWidget):
    activated = QtCore.Signal(str)
    currentIndexChanged = QtCore.Signal(str)
    editTextChanged = QtCore.Signal(str)
    highlighted = QtCore.Signal(str)

    # todo make more things configurable
    def __init__(self, label_text, list_of_strings, hover_text=None,
                 default_entry=0, editable=True, has_check_box=False):
        # pass up the stack
        super(ComboBox, self).__init__(label_text=label_text,
                                       hover_text=hover_text,
                                       has_check_box=has_check_box)
        # make the cb
        self._cb = QtGui.QComboBox()
        self._cb.setEditable(editable)
        # stash the text
        self._list_of_strings = list_of_strings
        # shove in the text
        self._cb.addItems(list_of_strings)
        # buddy them up
        self._lab.setBuddy(self._cb)
        # make and set the layout
        # add the combo box to the layout defined in UtilsWidget
        self._layout.addWidget(self._cb)
        self.setLayout(self._layout)

        # connect on the signals
        self._cb.activated[str].connect(self.activated)
        self._cb.currentIndexChanged[str].connect(self.currentIndexChanged)
        self._cb.editTextChanged[str].connect(self.editTextChanged)
        self._cb.highlighted[str].connect(self.highlighted)

    # forward the slots
    @QtCore.Slot()
    def clear(self):
        self._cb.clear()

    @QtCore.Slot(int)
    def setCurrentIndex(self, in_val):
        self._cb.setCurrentIndex(in_val)

    @QtCore.Slot(str)
    def setEditText(self, in_str):
        self._cb.setEditText(in_str)

    def getValue(self):
        if self._has_check_box and self._check_box.isChecked():
            return self._list_of_strings[self._cb.currentIndex()]

        return None


class LineEdit(UtilsWidget):
    cursorPositionChanged = QtCore.Signal(int, int)
    editingFinished = QtCore.Signal()
    returnPressed = QtCore.Signal()
    selectionChanged = QtCore.Signal()
    textChanged = QtCore.Signal(str)
    textEdited = QtCore.Signal(str)

    def __init__(self, label_text, hover_text=None, editable=True,
                 has_check_box=False):
        # pass up the stack
        super(LineEdit, self).__init__(label_text=label_text,
                                       hover_text=hover_text,
                                       has_check_box=has_check_box)
        # make the line edit box
        self._line_editor = QtGui.QLineEdit()
        # buddy them up
        self._lab.setBuddy(self._line_editor)
        # add the line edit widget to the layout
        self._layout.addWidget(self._line_editor)
        self.setLayout(self._layout)

        # connect the signals
        self._line_editor.cursorPositionChanged.connect(self.cursorPositionChanged)
        self._line_editor.editingFinished.connect(self.editingFinished)
        self._line_editor.returnPressed.connect(self.returnPressed)
        self._line_editor.selectionChanged.connect(self.selectionChanged)
        self._line_editor.textChanged.connect(self.textChanged)
        self._line_editor.textEdited.connect(self.textEdited)

    # forward the slots
    @QtCore.Slot()
    def clear(self):
        self._line_editor.clear()

    @QtCore.Slot()
    def copy(self):
        self._line_editor.copy()

    @QtCore.Slot()
    def cut(self):
        self._line_editor.cut()

    @QtCore.Slot()
    def paste(self):
        self._line_editor.paste()

    @QtCore.Slot()
    def redo(self):
        self._line_editor.redo()

    @QtCore.Slot()
    def selectAll(self):
        self._line_editor.selectAll()

    @QtCore.Slot()
    def setText(self, str):
        self._line_editor.setText(str)

    @QtCore.Slot()
    def undo(self):
        self._line_editor.undo()

    def getValue(self):
        if self._has_check_box and self._check_box.isChecked():
            # returned as a QString
            text = str(self._line_editor.text())
            # check to see if it empty
            if text == '':
                return None
            return text
        return None


class CheckBox(UtilsWidget):
    stateChanged = QtCore.Signal()

    # todo make more things configurable
    def __init__(self, label_text, hover_text=None, editable=True,
                 has_check_box=False):
        # pass up the stack
        super(CheckBox, self).__init__(label_text=label_text,
                                       hover_text=hover_text,
                                       has_check_box=has_check_box)
        # make the check box
        self._check = QtGui.QCheckBox()
        # buddy them up
        self._lab.setBuddy(self._check)

        self._layout.addWidget(self._check)
        self.setLayout(self._layout)

        # connect the signal
        self._check.stateChanged.connect(self.stateChanged)

    # forward the slots
    # no slots to forward

    def getValue(self):
        if self._has_check_box and self._check_box.isChecked():
            return self._check.isChecked()
        return None


class TripleSpinner(QtGui.QGroupBox):
    """
    A class to wrap up the logic for dealing with a min/max/step
    triple spin box.
    """
    # signal to be emitted when the spin boxes are changed
    # and settled
    valueChanged = QtCore.Signal(float, float)

    def __init__(self, title='', parent=None):
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        self._spinbox_min_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_max_intensity = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_intensity_step = QtGui.QDoubleSpinBox(parent=self)

        ispiner_form = QtGui.QFormLayout()
        ispiner_form.addRow("min", self._spinbox_min_intensity)
        ispiner_form.addRow("max", self._spinbox_max_intensity)
        ispiner_form.addRow("step", self._spinbox_intensity_step)
        self.setLayout(ispiner_form)

    # TODO

    @QtCore.Slot(float, float)
    def setValues(self, bottom, top):
        """
        """
        pass

    @QtCore.Slot(float, float)
    def setLimits(self, bottom, top):
        """
        """
        pass

    @QtCore.Slot(float)
    def setStep(self, step):
        """
        """
        pass

    @property
    def values(self):
        return (self._spinbox_min_intensity.value,
                self._spinbox_max_intensity.value)


class PairSpinner(QtGui.QGroupBox):
    valueChanged = QtCore.Signal(float)
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self, init_min, init_max,
                 init_step, parent=None, title='',
                 value_str=None, step_str=None):
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        if value_str is None:
            value_str = 'value'
        if step_str is None:
            step_str = 'step'

        self._spinbox_value = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_step = QtGui.QDoubleSpinBox(parent=self)
        self._spinbox_step.valueChanged.connect(
            self._spinbox_value.setSingleStep)

        self._spinbox_value.valueChanged.connect(
            self.valueChanged)

        ispiner_form = QtGui.QFormLayout()
        ispiner_form.addRow(value_str, self._spinbox_value)
        ispiner_form.addRow(step_str, self._spinbox_step)
        self.setLayout(ispiner_form)
        self.setStep(init_step)
        self.setRange(init_min, init_max)

    @QtCore.Slot(float)
    def setStep(self, new_step):
        self._spinbox_step.setValue(new_step)

    @QtCore.Slot(float, float)
    def setRange(self, new_min, new_max):
        self._spinbox_value.setMinimum(new_min)
        self._spinbox_value.setMaximum(new_max)
        self.rangeChanged.emit(new_min, new_max)


class ControlContainer(QtGui.QGroupBox, mapping_mixin):
    _delim = '.'
    _dispatch_map = {'slider': 'create_slider'}

    def create_widget(self, key, type_str, param_dict):
        create_fun_name = self._dispatch_map[type_str]
        create_fun = getattr(self, create_fun_name)
        return create_fun(key, **param_dict)

    def __len__(self):
        print('len')
        return len(list(iter(self)))

    def __contains__(self, key):
        print('contains')
        return key in iter(self)

    def __init__(self, title, parent=None):
        # call parent constructor
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        # nested containers
        self._containers = dict()
        # all non-container contents of this container
        self._contents = dict()

        # specialized listings
        # this is a dict keyed on type of dicts
        # The inner dicts are keyed on name
        self._by_type = defaultdict(dict)

        # make the layout
        self._layout = QtGui.QVBoxLayout()
        # add it to self
        self.setLayout(self._layout)

    def __getitem__(self, key):
        print(key)
        # TODO make this sensible un-wrap KeyException errors
        try:
            # split the key
            split_key = key.strip(self._delim).split(self._delim, 1)
        except TypeError:
            raise KeyError("key is not a string")
        # if one element back -> no splitting needed
        if len(split_key) == 1:
            return self._contents[split_key[0]]
        # else, at least one layer of testing
        else:
            # unpack the key parts
            outer, inner = split_key
            # get the container and pass through the remaining key
            return self._containers[outer][inner]

    def create_container(self, key, container_title=None):
        """
        Create a nested container with in this container

        TODO : add rest of GroupBox parameters

        Parameters
        ----------
        key : str
            The key used to identify this container

        container_title : str or None
            The title of the container.
            If None, defaults to the key.
            If you want to title, use ''

        Returns
        -------
        control_container : ControlContainer
           The container created.
        """
        if container_title is None:
            container_title = key
        control_container = ControlContainer(container_title, parent=self)
        self._layout.addWidget(control_container)
        self._containers[key] = control_container
        return control_container

    def create_button(self, key):
        pass

    def create_checkbox(self, key):
        pass

    def create_combobox(self, key, key_list, editable=True, title=None):
        if title is None:
            title = key
        cb = ComboBox(title, key_list, editable=editable)
        self._add_widget(key, cb)
        return cb

    def create_dict_display(self, key, input_dict):
        pass

    def create_pairspinner(self, key, *args, **kwargs):
        ds = PairSpinner(*args, **kwargs)
        self._add_widget(key, ds)
        return ds

    def create_text(self, key, text):
        """
        Create and add a text label to the control panel
        """
        # create text
        tmp_label = QtGui.QLabel(text)
        self._add_widget(key, tmp_label)

    def create_radiobuttons(self, key):
        pass

    def create_slider(self, key, min_val, max_val, label=None):
        """

        Parameters
        ----------
        """
        if label is None:
            label = key
        # set up slider
        slider = Slider(label, min_val, max_val)
        self._add_widget(key, slider)
        return slider

    def create_triplespinbox(self, key):
        pass

    def _add_widget(self, key, in_widget):
        split_key = key.strip(self._delim).rsplit(self._delim, 1)
        # key is not nested, add to this object
        if len(split_key) == 1:
            key = split_key[0]
            # add to the type dict
            self._by_type[type(in_widget)][key] = in_widget
            # add to the contents list
            self._contents[key] = in_widget
            # add to layout
            self._layout.addWidget(in_widget)
        # else, grab the nested container and add it to that
        else:
            container, key = split_key
            self[container]._add_widget(key, in_widget)

    def iter_containers(self):
        return self._iter_helper_container([])

    def get_container(self, key):
        """
        Get a (possibly nested) container (the normal
        iterator skips these).  We may end up with two
        parallel sets of mapping functions.
        """
        split_key = key.strip(self._delim).rsplit(self._delim, 1)
        if len(split_key) == 1:
            return self._containers[split_key[0]]
        return self._containers[split_key[0]].get_containers(split_key[1])

    def _iter_helper(self, cur_path_list):
        """
        Recursively (depth-first) walk the tree and return the names
        of the leaves

        Parameters
        ----------
        cur_path_list : list of str
            A list of the current path
        """
        for k, v in six.iteritems(self._containers):
            for inner_v in v._iter_helper(cur_path_list + [k]):
                yield inner_v
        for k in six.iterkeys(self._contents):
            yield self._delim.join(cur_path_list + [k])

    def _iter_helper_container(self, cur_path_list):
        """
        Recursively (depth-first) walk the tree and return the names
        of the containers

        Parameters
        ----------
        cur_path_list : list of str
            A list of the current path
        """
        for k, v in six.iteritems(self._containers):
            for inner_v in v._iter_helper_container(cur_path_list + [k]):
                yield inner_v
        if len(cur_path_list):
            yield self._delim.join(cur_path_list)

    def __iter__(self):
        return self._iter_helper([])

    def addStretch(self):
        self._layout.addStretch()


class DictDisplay(QtGui.QGroupBox):
    """
    A generic widget for displaying dictionaries

    Parameters
    ----------
    title : string
       Widget title

    ignore_list : iterable or None
        keys to ignore

    parent : QWidget or None
        Parent widget, passed up stack
    """
    def __init__(self, title, ignore_list=None, parent=None):
        # pass up the stack, GroupBox takes care of the title
        QtGui.QGroupBox.__init__(self, title, parent=parent)

        if ignore_list is None:
            ignore_list = ()
        # make layout
        self.full_layout = QtGui.QVBoxLayout()
        # set layout
        self.setLayout(self.full_layout)
        # make a set of the ignore list
        self._ignore = set(ignore_list)
        self._disp_table = []

    @QtCore.Slot(dict)
    def update(self, in_dict):
        """
        updates the table

        Parameters
        ----------
        in_dict : dict
            The dictionary to display
        """
        # remove everything that is there
        for c in self._disp_table:
            c.deleteLater()

        # make a new list
        self._disp_table = []

        # add the keys alphabetically
        for k, v in sorted(list(in_dict.iteritems())):
            # if key in the ignore list, continue
            if k in self._ignore:
                continue
            self._add_row(k, v)

    def _add_row(self, k, v):
        """
        Private function

        Adds a row to the table

        Parameters
        ----------
        k : object
           The key

        v : object
           The value
        """
        # make a widget for our row
        tmp_widget = QtGui.QWidget(self)
        tmp_layout = QtGui.QHBoxLayout()
        tmp_widget.setLayout(tmp_layout)

        # add the key and value to the row widget
        tmp_layout.addWidget(QtGui.QLabel(str(k) + ':'))
        tmp_layout.addStretch()
        tmp_layout.addWidget(QtGui.QLabel(str(v)))

        # add the row widget to the full layout
        self.full_layout.addWidget(tmp_widget)
        # add the row widget to
        self._disp_table.append(tmp_widget)
