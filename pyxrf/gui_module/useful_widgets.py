from qtpy.QtWidgets import (
    QLineEdit,
    QWidget,
    QHBoxLayout,
    QComboBox,
    QTextEdit,
    QSizePolicy,
    QLabel,
    QPushButton,
    QGridLayout,
    QSlider,
    QSpinBox,
    QCheckBox,
)
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QPalette, QColor, QFontMetrics, QIntValidator, QDoubleValidator

import logging

logger = logging.getLogger(__name__)

global_gui_parameters = {"vertical_spacing_in_tabs": 5}

global_gui_variables = {
    # Reference to main window
    "ref_main_window": None,
    # The flags that control current GUI state
    # (global state that determines if elements are enabled/visible)
    "gui_state": {
        "databroker_available": False,
        "running_computations": False,
        # The following states are NOT mutually exclusive
        "state_file_loaded": False,
        "state_model_exists": False,
        "state_model_fit_exists": False,
        "state_xrf_map_exists": False,
    },
    # Indicates if tooltips must be shown
    "show_tooltip": True,
    "show_matplotlib_toolbar": True,
}


def clear_gui_state(gui_vars):
    """
    Clear GUI state. Reset the variables that determine GUI state.
    This should be done before the new data is loaded from file or from Databoker.
    The variables are set so that the state of GUI is "No data is loaded"

    Parameters
    ----------
    gui_vars: dict
        reference to the dictionary `global_gui_variables`
    """
    gui_vars["gui_state"]["state_file_loaded"] = False
    gui_vars["gui_state"]["state_model_exists"] = False
    gui_vars["gui_state"]["state_model_fit_exists"] = False
    gui_vars["gui_state"]["state_xrf_map_exists"] = False


def set_tooltip(widget, text):
    """
    Set tooltips for the widget. Use global variable `global_gui_variables["show_tooltips"]`
    to determine if tooltips must be set.

    Parameters
    ----------
    widget: QWidget
        reference to the widget
    text: str
        text to set as a tooltip
    """
    if not global_gui_variables["show_tooltip"]:
        text = ""
    widget.setToolTip(text)


class LineEditExtended(QLineEdit):
    """
    LineEditExtended allows to mark the displayed value as invalid by setting
    its `valid` property to False. By default, the text color is changed to Light Red.
    It also emits `focusOut` signal at `self.focusOutEvent`.
    """

    # Emitted at focusOutEvent
    focusOut = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._valid = True
        self._style_sheet_valid = ""  # By default, clear the style sheet
        self._style_sheet_invalid = "color: rgb(255, 0, 0);"
        self._update_valid_status()

    def _update_valid_status(self):
        if self._valid:
            super().setStyleSheet(self._style_sheet_valid)
        else:
            super().setStyleSheet(self._style_sheet_invalid)

    def setStyleSheet(self, style_sheet, *, valid=True):
        """
        Set style sheet for valid/invalid states. If call with one parameter, the function
        works the same as `setStyleSheet` of QWidget. If `valid` is set to `False`, the
        supplied style sheet will be applied only if 'invalid' state is activated. The
        style sheets for the valid and invalid states are independent and can be set
        separately.

        The default behavior: 'valid' state - clear style sheet, 'invalid' state -
        use the style sheet `"color: rgb(255, 0, 0);"`

        Parameters
        ----------
        style_sheet: str
            style sheet
        valid: bool
            True - activate 'valid' state, False - activate 'invalid' state
        """
        if valid:
            self._style_sheet_valid = style_sheet
        else:
            self._style_sheet_invalid = style_sheet
        self._update_valid_status()

    def getStyleSheet(self, *, valid):
        """
        Return the style sheet used 'valid' or 'invalid' state.

        Parameters
        ----------
        valid: bool
            True/False - return the style sheet that was set for 'valid'/'invalid' state.
        """
        if valid:
            return self._style_sheet_valid
        else:
            return self._style_sheet_invalid

    def setValid(self, state):
        """Set the state of the line edit box.: True - 'valid', False - 'invalid'"""
        self._valid = bool(state)
        self._update_valid_status()

    def isValid(self):
        """
        Returns 'valid' status of the line edit box (bool).
        """
        return self._valid

    def focusOutEvent(self, event):
        """
        Overriddent QWidget method. Sends custom 'focusOut()' signal
        """
        super().focusOutEvent(event)
        self.focusOut.emit()


class LineEditReadOnly(LineEditExtended):
    """
    Read-only version of QLineEdit with background set to the same color
    as the background of the disabled QLineEdit, but font color the same
    as active QLineEdit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        # Set background color the same as for disabled window.
        p = self.palette()
        p.setColor(QPalette.Base, p.color(QPalette.Disabled, QPalette.Base))
        self.setPalette(p)


class TextEditReadOnly(QTextEdit):
    """
    Read-only version of QTextEdit with background set to the same color
    as the background of the disabled QLineEdit, but font color the same
    as active QLineEdit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        # Set background color the same as for disabled window.
        p = self.palette()
        p.setColor(QPalette.Base, p.color(QPalette.Disabled, QPalette.Base))
        self.setPalette(p)


class PushButtonMinimumWidth(QPushButton):
    """
    Push button with text ".." and minimum width
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        text = self.text()
        font = self.font()

        fm = QFontMetrics(font)
        text_width = fm.width(text) + 6
        self.setFixedWidth(text_width)


class PushButtonNamed(QPushButton):
    """
    Push box that returns 'name' as the first parameter with the signals.
    Named widget is useful to distinguish between widgets when they are inserted in table rows.
    """

    clicked = Signal(str)

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        super().clicked.connect(self._clicked)

    def getName(self):
        return self._name

    @Slot()
    def _clicked(self):
        name = self._name if self._name is not None else ""
        self.clicked.emit(name)


class ComboBoxNamed(QComboBox):
    """
    Combo box that returns 'name' as the first parameter with the signals.
    Named widget is useful to distinguish between widgets when they are inserted in table rows.
    """

    currentIndexChanged = Signal(str, int)

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        super().currentIndexChanged.connect(self._current_index_changed)

    def getName(self):
        return self._name

    @Slot(int)
    def _current_index_changed(self, index):
        name = self._name if self._name is not None else ""
        self.currentIndexChanged.emit(name, index)


class ComboBoxNamedNoWheel(ComboBoxNamed):
    """
    Named combobox that ignores wheel events. Wheel events cause combo box to
    scroll through the element, which is not always desirable. For example when
    combo boxes are inserted in large tables, wheel is naturally used to scroll
    through the table. If mouse is accidently pointed to a combo box, then rotating
    the wheel will change combo box current element instead of scrolling through
    the table. Ignoring the wheel events in the combo box passes them to the
    parent for processing, which solves the issue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wheelEvent(self, event):
        event.ignore()


class CheckBoxNamed(QCheckBox):

    stateChanged = Signal(str, int)

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        super().stateChanged.connect(self._state_changed)

    def getName(self):
        return self._name

    @Slot(int)
    def _state_changed(self, state):
        name = self._name if self._name is not None else ""
        self.stateChanged.emit(name, state)


class SpinBoxNamed(QSpinBox):

    valueChanged = Signal(str, int)

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        super().valueChanged.connect(self._value_changed)

    def getName(self):
        return self._name

    @Slot(int)
    def _value_changed(self, value):
        name = self._name if self._name is not None else ""
        self.valueChanged.emit(name, value)


class SecondaryWindow(QWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # The variable indicates if the window was moved using 'position_once' function
        self._never_positioned = True

    def position_once(self, x, y, *, x_shift=30, y_shift=30, force=False):
        """
        Move the window once (typically before it is first shown).
        Used to position the window regarding the parent window before it is show.
        Then the user may move the window anywhere on the screen and it will remain
        there.

        Parameters
        ----------
        x, y: int
            screen coorinates of the left top corner of the parent window
        x_shift, y_shift: int
            shift applied to 'x' and 'y' to position left top corner of this window:
            the window is positioned at (x+x_shift, y+y_shift)
        force: bool
            True - move anyway, False (default) - move only the first time the function is called
        """
        if self._never_positioned or force:
            self._never_positioned = False
            self.move(x + x_shift, y + y_shift)


def adjust_qlistwidget_height(list_widget, *, other_widgets=None, min_height=40):
    """
    Adjust the height of QListWidget so that it fits the items exactly.
    If the computed height is smaller than `min_height`, then the height
    is set to `min_height`. If the list is empty, then the height is set
    to `min_height` (for pleasing look).

    The width of the widget is still adjusted automatically.
    Call this function each time the number of items in the list
    is changed.

    Parameters
    ----------
    list_widget: QListWidget
        reference to the list widget that needs to be adjusted.
    min_height: int
        minimum height of the widget.
    """

    if other_widgets is None:
        other_widgets = []

    # Compute and set the height of the list
    height = 0
    n_list_elements = list_widget.count()
    if n_list_elements:
        # Compute the height necessary to accommodate all the elements
        height = list_widget.sizeHintForRow(0) * n_list_elements + 2 * list_widget.frameWidth() + 3
    # Set some visually pleasing height if the list contains no elements
    height = max(height, min_height)
    list_widget.setMinimumHeight(height)
    list_widget.setMaximumHeight(height)

    # Now update size of the other ('parent') widgets
    for w in other_widgets:
        w.adjustSize()
        w.updateGeometry()  # This is necessary in some cases


def get_background_css(rgb, widget="QWidget", editable=False):
    """Returns the string that contain CSS to set widget background to specified color"""

    rgb = tuple(rgb)
    if len(rgb) != 3:
        raise ValueError(r"RGB must be represented by 3 elements: rgb = {rgb}")
    if any([(_ > 255) or (_ < 0) for _ in rgb]):
        raise ValueError(r"RGB values must be in the range 0..255: rgb={rgb}")

    # Shaded widgets appear brighter, so the brightness needs to be reduced
    shaded_widgets = ("QComboBox", "QPushButton")
    if widget in shaded_widgets:
        rgb = [max(int(255 - (255 - _) * 1.5), 0) for _ in rgb]

    # Increase brightness of editable element
    if editable:
        rgb = [255 - int((255 - _) * 0.5) for _ in rgb]

    color_css = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    return f"{widget} {{ background-color: {color_css}; }}"


class IntValidatorStrict(QIntValidator):
    """
    `IntValidatorStrict` verifies additional condition: int number can not
    contain commas, since it can't be converted to int number directly.
    """

    def validate(self, text, pos):
        result = super().validate(text, pos)
        # Additional condition: the number can not contain ",". For some reason
        #   the standard validators ignore commas at the end of the number.
        if "," in text[pos:]:
            result = (QIntValidator.Invalid, result[1], result[2])
        return result


class IntValidatorRelaxed(IntValidatorStrict):
    """
    IntValidatorRelaxed is an extension of QIntValidator. In the overridden
    `validate` method, the original output is changed so that  the return value
    `Invalid` (input is rejected) is replaced by `Intermediate` (input is expected
    to be acceptable once the typing is finished).

    When the validator is set for the QLineEdit, users are allowed to freely type
    in the line edit box. Validator will still prevent 'editingFinished' signal
    from being emitted.
    """

    def validate(self, text, pos):
        result = super().validate(text, pos)
        # Replace QIntValidator.Invalid with QIntValidator.Intermediate
        if result[0] == QIntValidator.Invalid:
            result = (QIntValidator.Intermediate, result[1], result[2])
        return result


class DoubleValidatorStrict(QDoubleValidator):
    """
    `DoubleValidatorStrict` verifies additional condition: double number can not
    contain commas, since it can't be converted to floating point number directly.
    """

    def validate(self, text, pos):
        result = super().validate(text, pos)
        # Additional condition: the number can not contain ",". For some reason
        #   the standard validators ignore commas at the end of the number.
        if "," in text[pos:]:
            result = (QDoubleValidator.Invalid, result[1], result[2])
        return result


class DoubleValidatorRelaxed(DoubleValidatorStrict):
    """
    DoubleValidatorRelaxed is similar to `IntValidatorRelaxed`, but works with
    values of `double` type.
    """

    def validate(self, *args, **kwargs):
        result = super().validate(*args, **kwargs)
        # Replace QDoubleValidator.Invalid with QIntValidator.Intermediate
        if result[0] == QDoubleValidator.Invalid:
            result = (QDoubleValidator.Intermediate, result[1], result[2])
        return result


class RangeManager(QWidget):
    """Width of the widgets can be set using `setMaximumWidth`. The size policy is set
    so that the widget may shrink if there is not enough space."""

    selection_changed = Signal(float, float, str)

    def __init__(self, *, name="", add_sliders=False, slider_steps=10000, selection_to_range_min=0.001):
        """
        Class constructor for RangeManager

        Parameters
        ----------
        add_sliders: bool
            True - the widget will include sliders for controlling the range,
            False - the widget will have no sliders without sliders
        slider_steps: int
            The number of slider steps. Determines the precision of the slider.
            Default value is sufficient in most cases
        selection_to_range_min: float
            Minimum ratio of the selected range and total range. Must be floating
            point number >=0. Used only when the value type is set to "float":
            `self.set_value_type("float")`. Minimum selected range is always 1
            when "int" value type is set.
        """
        super().__init__()

        self._name = name

        # Set the maximum number of steps for the sliders (resolution)
        self.sld_n_steps = slider_steps
        # Ratio of the minimum range and total range. It is used to compute
        #   the value of 'self._range_min_diff'. The widget will prevent
        #   range to be set to smaller value than 'self._range_min_diff'.
        self._selection_to_range_min = selection_to_range_min

        self._range_low = 0.0
        self._range_high = 100.0
        self._range_min_diff = (self._range_high - self._range_low) * self._selection_to_range_min
        self._value_per_step = (self._range_high - self._range_low) / self.sld_n_steps
        self._value_type = "float"

        # The following values are used to keep the low and high of the range.
        #   Those values are 'accepted' values that reflect current selected range.
        self._value_low = self._range_low
        self._value_high = self._range_high

        max_element_width = 200

        self.le_min_value = LineEditExtended()
        self.le_max_value = LineEditExtended()
        self.validator_low = DoubleValidatorRelaxed()
        self.validator_high = DoubleValidatorRelaxed()

        self.le_min_value.setMaximumWidth(max_element_width)
        self.le_min_value.textEdited.connect(self.le_min_value_text_edited)
        self.le_min_value.textChanged.connect(self.le_min_value_text_changed)
        self.le_min_value.editingFinished.connect(self.le_min_value_editing_finished)
        self.le_max_value.setMaximumWidth(max_element_width)
        self.le_max_value.textEdited.connect(self.le_max_value_text_edited)
        self.le_max_value.textChanged.connect(self.le_max_value_text_changed)
        self.le_max_value.editingFinished.connect(self.le_max_value_editing_finished)

        self.le_min_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.le_max_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # The flag is set true if mouse is pressed on one of the sliders
        #   Both sliders can not be pressed at once, so one variable is sufficient
        self._sld_mouse_pressed = False

        self.sld_min_value = QSlider(Qt.Horizontal)
        self.sld_min_value.valueChanged.connect(self.sld_min_value_value_changed)
        self.sld_min_value.sliderPressed.connect(self.sld_min_value_slider_pressed)
        self.sld_min_value.sliderReleased.connect(self.sld_min_value_slider_released)
        self.sld_max_value = QSlider(Qt.Horizontal)
        self.sld_max_value.valueChanged.connect(self.sld_max_value_value_changed)
        self.sld_max_value.sliderPressed.connect(self.sld_max_value_slider_pressed)
        self.sld_max_value.sliderReleased.connect(self.sld_max_value_slider_released)

        self.sld_min_value.setMaximumWidth(max_element_width)
        self.sld_max_value.setMaximumWidth(max_element_width)

        # The slider for controlling minimum is inverted
        self.sld_min_value.setInvertedAppearance(True)
        self.sld_min_value.setInvertedControls(True)

        self.sld_min_value.setMaximum(self.sld_n_steps)
        self.sld_max_value.setMaximum(self.sld_n_steps)

        self.sld_min_value.setValue(self.sld_min_value.maximum())
        self.sld_max_value.setValue(self.sld_max_value.maximum())

        self.set_value_type(self._value_type)  # Set the validator

        grid = QGridLayout()
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.le_min_value, 0, 0)
        grid.addWidget(QLabel(".."), 0, 1)
        grid.addWidget(self.le_max_value, 0, 2)

        if add_sliders:
            grid.addWidget(self.sld_min_value, 1, 0)
            grid.addWidget(QLabel(""), 1, 1)
            grid.addWidget(self.sld_max_value, 1, 2)

        self.setLayout(grid)

        sp = QSizePolicy()
        sp.setControlType(QSizePolicy.PushButton)
        sp.setHorizontalPolicy(QSizePolicy.Maximum)
        self.setSizePolicy(sp)

    def le_min_value_text_edited(self, text):
        if self._min_value_validate(text):
            v = float(text)  # Works even if the value is expected to be 'int'
            n_steps = self._value_to_slider(v)
            self.sld_min_value.setValue(self.sld_n_steps - n_steps)

    def le_min_value_text_changed(self, text):
        self._min_value_validate(text)

    def le_min_value_editing_finished(self):
        # We don't set validator, so this method is called each time QLineEdit
        #   is losing focus or Enter is pressed
        val = self.le_min_value.text() if self._min_value_validate() else self._value_low
        if self._accept_value_low(val):
            self.emit_selection_changed()

    def le_max_value_text_edited(self, text):
        if self._max_value_validate(text):
            v = float(text)  # Works even if the value is expected to be 'int'
            n_steps = self._value_to_slider(v)
            self.sld_max_value.setValue(n_steps)

    def le_max_value_text_changed(self, text):
        self._max_value_validate(text)

    def le_max_value_editing_finished(self):
        # We don't set validator, so this method is called each time QLineEdit
        #   is losing focus or Enter is pressed
        val = self.le_max_value.text() if self._max_value_validate() else self._value_high
        if self._accept_value_high(val):
            self.emit_selection_changed()

    def sld_min_value_value_changed(self, n_steps):
        # Invert the reading for 'min' slider
        if self._sld_mouse_pressed:
            n_steps = self.sld_n_steps - n_steps
            v = self._slider_to_value(n_steps)
            self.le_min_value.setText(self._format_value(v))

    def sld_min_value_slider_pressed(self):
        self._sld_mouse_pressed = True

    def sld_min_value_slider_released(self):
        self._sld_mouse_pressed = False
        n_steps = self.sld_n_steps - self.sld_min_value.value()
        v = self._slider_to_value(n_steps)
        if self._accept_value_low(v):
            self.emit_selection_changed()

    def sld_max_value_value_changed(self, n_steps):
        if self._sld_mouse_pressed:
            v = self._slider_to_value(n_steps)
            self.le_max_value.setText(self._format_value(v))

    def sld_max_value_slider_pressed(self):
        self._sld_mouse_pressed = True

    def sld_max_value_slider_released(self):
        self._sld_mouse_pressed = False
        n_steps = self.sld_max_value.value()
        v = self._slider_to_value(n_steps)
        if self._accept_value_high(v):
            self.emit_selection_changed()

    def _format_value(self, value):
        return f"{value:.8g}"

    def _round_value(self, value):
        # Compute rounded value based on formatting used in the line edit boxes
        #   This rounding is needed to properly set the validators
        s = self._format_value(value) if not isinstance(value, str) else value
        return self._convert_type(s)

    def _check_value_type(self, value_type):
        if value_type not in ("float", "int"):
            raise ValueError(f"RangeManager.set_value_type(): value type '{value_type}' is not supported")

    def _min_value_validate(self, text=None):
        text = text if text is not None else self.le_min_value.text()
        is_valid = self.validator_low.validate(text, 0)[0] == 2
        self.le_min_value.setValid(is_valid)
        return is_valid

    def _max_value_validate(self, text=None):
        text = text if text is not None else self.le_max_value.text()
        is_valid = self.validator_high.validate(text, 0)[0] == 2
        self.le_max_value.setValid(is_valid)
        return is_valid

    def _convert_type(self, val):
        if self._value_type == "float":
            return float(val)
        else:
            # Convert to int (input may be float or text string).
            # We want to round the value to the nearest int.
            return round(float(val))

    def _slider_to_value(self, sld_n):
        v = self._range_low + (self._range_high - self._range_low) * (sld_n / self.sld_n_steps)
        return self._convert_type(v)

    def _value_to_slider(self, value):
        rng = self._range_high - self._range_low
        if rng > 1e-30:
            return round((value - self._range_low) / rng * self.sld_n_steps)
        else:
            return 0

    def _accept_value_low(self, val):
        val = self._convert_type(val)
        val_max = self._value_high - self._range_min_diff
        val = val if val <= val_max else val_max
        return self.set_selection(value_low=val)

    def _accept_value_high(self, val):
        val = self._convert_type(val)
        val_min = self._value_low + self._range_min_diff
        val = val if val >= val_min else val_min
        return self.set_selection(value_high=val)

    def _adjust_min_diff(self):
        if self._value_type == "float":
            self._range_min_diff = (self._range_high - self._range_low) * self._selection_to_range_min
        else:
            self._range_min_diff = 1

    def _adjust_validators(self):
        """Set the range for validators based on full range and the selected range."""
        if self._value_type == "float":
            # Validator type: QDoubleValidator
            # The range is set a little wider (1% wider) in order to cover the 'true'
            #   boundary value. 'decimals=-1' - it seems that the precision is getting ignored.
            self.validator_low.setRange(
                self._round_value(self._range_low),
                self._round_value(self._value_high - self._range_min_diff * 0.99),
                decimals=-1,
            )
            self.validator_high.setRange(
                self._round_value(self._value_low + self._range_min_diff * 0.99),
                self._round_value(self._range_high),
                decimals=-1,
            )
        else:
            # Validator type: QIntValidator
            # With integer arithmetic we can set the range precisely
            self.validator_low.setRange(round(self._range_low), round(self._value_high - self._range_min_diff))
            self.validator_high.setRange(round(self._value_low + self._range_min_diff), round(self._range_high))

    def setAlignment(self, flags):
        """
        Set text alignment in QLineEdit widgets

        Parameters
        ----------
        flags: Qt.Alignment flags
            flags that set alignment of text in QLineEdit widgets
            For example `Qt.AlignCenter`. The default settings
            for the widget is `Qt.AlignRight | Qt.AlignVCenter`.
        """

        self.le_min_value.setAlignment(flags)
        self.le_max_value.setAlignment(flags)

    def setBackground(self, rgb):
        """
        Set background color of the widget. Similar to QTableWidgetItem.setBackground,
        but accepting a tuple of RGB values instead of QBrush.

        Parameters
        ----------
        rgb: tuple(int)
            RGB color in the form of (R, G, B)
        """
        self.setStyleSheet(get_background_css(rgb, widget="QWidget", editable=False))

        self.le_min_value.setStyleSheet(get_background_css(rgb, widget="QLineEdit", editable=True))
        self.le_max_value.setStyleSheet(get_background_css(rgb, widget="QLineEdit", editable=True))

    def setTextColor(self, rgb):
        """
        Set text color for the widget. This color is used in 'normal' (valid) state.

        Parameters
        ----------
        rgb: tuple(int)
            RGB color in the form of (R, G, B)
        """
        color = QColor(*rgb)
        pal = self.le_min_value.palette()
        pal.setColor(QPalette.Text, color)
        self.le_min_value.setPalette(pal)
        pal = self.le_max_value.palette()
        pal.setColor(QPalette.Text, color)
        self.le_max_value.setPalette(pal)

    def set_value_type(self, value_type="float"):
        """
        Set value type for the range widget. The value type determines
        the type and format of the displayed and returned values and the type of
        the validator used by the line edit widgets. The current choices are:
        "float" used for working with ranges expressed as floating point (double) numbers;
        "int" is intended for ranges expressed as integers.

        Parameters
        ----------
        value_type: str
            Type of values managed by the widget. The choices are "float" and "int".
            `ValueError` is raised if wrong value is supplied.

        Returns
        -------
        True - selected range was changed when full range was changed, False otherwise.
        """
        self._check_value_type(value_type)
        self._value_type = value_type

        # We don't set validators for QLineEdit widgets. Instead we call validators
        #   explicitly when the input changes. Validators are used to indicate invalid
        #   inputs and prevent user from accepting invalid inputs. There is no goal
        #   to prevent users from typing invalid expressions.
        if self._value_type == "float":
            self.validator_low = DoubleValidatorRelaxed()
            self.validator_high = DoubleValidatorRelaxed()
        else:
            self.validator_low = IntValidatorRelaxed()
            self.validator_high = IntValidatorRelaxed()

        # Completely reset the widget
        return self.set_range(self._range_low, self._range_high)

    def set_range(self, low, high):
        """
        Set the full range of the RangeManager widget. The range may not be negative or
        zero: `low` must be strictly smaller than `high`. The `ValueError` is raised if
        `low >= high`. The function will not emit `selection_changed` signal. Call
        `emit_selection_changed()` method to emit the signal.

        Parameters
        ----------
        low, high: float or int
            lower and upper boundaries of the full range

        Returns
        -------
        True - selected range was changed when full range was changed, False otherwise.
        """
        low, high = self._convert_type(low), self._convert_type(high)
        # Check range
        if low >= high:
            raise ValueError(f"RangeManager.set_range(): incorrect range: low > high ({low} > {high})")

        def _compute_new_value(val_old):
            """
            Adjust the current value so that the selected range covers the same
            fraction of the total range.
            """
            range_old = self._range_high - self._range_low
            range_new = high - low
            return (val_old - self._range_low) / range_old * range_new + low

        new_value_low = _compute_new_value(self._value_low)
        new_value_high = _compute_new_value(self._value_high)

        self._range_high = high
        self._range_low = low

        self._value_per_step = (self._range_high - self._range_low) / (self.sld_n_steps - 1)
        self._range_min_diff = (self._range_high - self._range_low) * self._selection_to_range_min

        return self.set_selection(value_low=new_value_low, value_high=new_value_high)

    def set_selection(self, *, value_low=None, value_high=None):
        """
        Set the selected range. The function may be used to set only lower or upper
        boundary. The function will not emit `selection_changed` signal.
        Call `emit_selection_changed()` method to emit the signal.

        Parameters
        ----------
        value_low: float, int or None
            lower boundary of the selected range. If `None`, then the lower boundary
            is not changed. If `value_low` is outside the full range, then it is clipped.
        value_high: float, int or None
            upper boundary of the selected range. If `None`, then the upper boundary
            is not changed. If `value_low` is outside the full range, then it is clipped.

        Returns
        -------
        True - selected range changed, False - selected range stayed the same
        """
        old_low, old_high = self._value_low, self._value_high

        if value_low is not None or value_high is not None:
            self._adjust_min_diff()
            if value_low is not None:
                value_low = self._convert_type(value_low)
                value_low = max(min(value_low, self._range_high), self._range_low)
                self._value_low = value_low
            if value_high is not None:
                value_high = self._convert_type(value_high)
                value_high = max(min(value_high, self._range_high), self._range_low)
                self._value_high = value_high
            # Exceptional case when the selection is smaller than the minimum selected
            #   range (or negative). Adjust the range: start at the specified 'low' value
            #   and cover the minimum selectable range; if 'high' value exceeds the top
            #   of the full range, then shift the selected range downwards to fit within
            #   the full range
            if self._value_high < self._value_low + self._range_min_diff:
                self._value_high = self._value_low + self._range_min_diff
                if self._value_high > self._range_high:
                    self._value_high = self._range_high
                    self._value_low = self._range_high - self._range_min_diff
            self._adjust_validators()
        if value_low is not None:
            self.sld_min_value.setValue(self.sld_n_steps - self._value_to_slider(self._value_low))
            self.le_min_value.setText(self._format_value(self._value_low))
        if value_high is not None:
            self.sld_max_value.setValue(self._value_to_slider(self._value_high))
            self.le_max_value.setText(self._format_value(self._value_high))

        # Return True if selection changed
        return (old_low != self._value_low) or (old_high != self._value_high)

    def reset(self):
        """
        Reset the selected range to full range of the RangeManager widget. The method will
        not emit `selection_changed` signal. Call `emit_selection_changed()` to
        emit the signal.

        Returns
        -------
        True - selected range changed, False - selected range stayed the same
        """
        return self.set_selection(value_low=self._range_low, value_high=self._range_high)

    def get_range(self):
        """
        Get the full range

        Returns
        -------
        tuple `(range_low, range_high)`, the values of `range_low` and `range_high` may be `int`
        or `float` type depending on the type set by `set_value_type()` method.
        """
        return self._range_low, self._range_high

    def get_selection(self):
        """
        Get the selected range

        Returns
        -------
        tuple `(v_low, v_high)`, the values of `v_low` and `v_high` may be `int`
        or `float` type depending on the type set by `set_value_type()` method.
        """
        return self._value_low, self._value_high

    def emit_selection_changed(self):
        """
        Emit `selection_changed` signal that passes the selected range as parameters.
        Note, that the parameters of the signal are ALWAYS `float`.
        """
        v_low = self._convert_type(self._value_low)
        v_high = self._convert_type(self._value_high)
        logger.debug(
            f"RangeManager ({self._name}): Emitting the signal 'selection_changed'. "
            f"Selection: ({v_low}, {v_high})"
        )
        self.selection_changed.emit(v_low, v_high, self._name)


class ElementSelection(QWidget):
    """Width of the widgets can be set using `setMaximumWidth`. The size policy is set
    so that the widget may shrink if there is not enough space."""

    signal_current_item_changed = Signal(int, str)

    def __init__(self):
        super().__init__()

        self._item_list = []
        # The 'first item' is appended to the beginning of the list. The index returned
        #   by the functions of the class is the index of 'self._item_list' array, that
        #   does not include the 'first item'.
        self._first_item = "Select Line:"

        self.cb_element_list = QComboBox()
        self.cb_element_list.currentIndexChanged.connect(self.cb_element_list_current_index_changed)

        self.setMaximumWidth(300)
        self.pb_prev = PushButtonMinimumWidth("<")
        self.pb_prev.pressed.connect(self.pb_prev_pressed)
        self.pb_next = PushButtonMinimumWidth(">")
        self.pb_next.pressed.connect(self.pb_next_pressed)

        self.cb_element_list.addItems([self._first_item])
        self.cb_element_list.setCurrentIndex(0)

        hbox = QHBoxLayout()
        hbox.setSpacing(0)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.pb_prev)
        hbox.addWidget(self.cb_element_list)
        hbox.addWidget(self.pb_next)

        self.setLayout(hbox)

        sp = QSizePolicy()
        sp.setControlType(QSizePolicy.PushButton)
        sp.setHorizontalPolicy(QSizePolicy.Maximum)
        self.setSizePolicy(sp)

    def set_item_list(self, item_list):
        _, current_item = self.get_current_item()

        self._item_list = item_list.copy()

        current_index = -1
        if current_item:
            try:
                current_index = self._item_list.index(current_item)
            except ValueError:
                pass

        self.cb_element_list.clear()
        self.cb_element_list.addItems([self._first_item] + self._item_list)

        self.cb_element_list.setCurrentIndex(current_index + 1)

        self._adjust_button_state()

    def set_current_index(self, index):
        current_index = -1
        if 0 <= index < len(self._item_list):
            current_index = index
        self.cb_element_list.setCurrentIndex(current_index + 1)

    def set_current_item(self, item):
        current_index = -1
        try:
            current_index = self._item_list.index(item)
        except ValueError:
            pass
        self.cb_element_list.setCurrentIndex(current_index + 1)

    def get_current_item(self):
        current_index = self.cb_element_list.currentIndex() - 1
        if 0 <= current_index < len(self._item_list):
            current_item = self._item_list[current_index]
        else:
            current_item = ""

        return current_index, current_item

    def _adjust_button_state(self):

        current_index = self.cb_element_list.currentIndex() - 1
        n_items = len(self._item_list)

        enable_prev, enable_next = True, True
        if n_items == 0:
            enable_prev, enable_next = False, False
        else:
            if current_index < 0:
                enable_prev = False
            if current_index >= n_items - 1:
                enable_next = False

        self.pb_prev.setEnabled(enable_prev)
        self.pb_next.setEnabled(enable_next)

    def cb_element_list_current_index_changed(self, index):
        self._adjust_button_state()
        current_index, current_item = self.get_current_item()
        self.signal_current_item_changed.emit(current_index, current_item)

    def pb_prev_pressed(self):
        current_index = self.cb_element_list.currentIndex() - 1
        if current_index >= 0:
            self.cb_element_list.setCurrentIndex(current_index)

    def pb_next_pressed(self):
        current_index = self.cb_element_list.currentIndex() - 1
        n_items = len(self._item_list)
        if current_index < n_items - 1:
            self.cb_element_list.setCurrentIndex(current_index + 2)
