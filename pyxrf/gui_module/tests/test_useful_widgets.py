import pytest
import numpy.testing as npt
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from pyxrf.gui_module.useful_widgets import (
    IntValidatorStrict,
    IntValidatorRelaxed,
    DoubleValidatorRelaxed,
    RangeManager,
)


def enter_text_via_keyboard(qtbot, widget, text, *, finish=True):
    """Use keyboard to enter values line edit box"""
    qtbot.mouseClick(widget, Qt.LeftButton)  # Put focus on the widget
    widget.selectAll()  # Select all existing text
    qtbot.keyClicks(widget, text)  # Then type
    if finish:
        qtbot.keyClick(widget, Qt.Key_Enter)


# ==============================================================
#   Class IntValidatorStrict
# fmt: off
@pytest.mark.parametrize("text, range, result", [
    ("", None, IntValidatorRelaxed.Intermediate),
    ("123", None, IntValidatorRelaxed.Acceptable),
    ("1&34", None, IntValidatorRelaxed.Invalid),
    ("2", (0, 10), IntValidatorRelaxed.Acceptable),
    ("2", (10, 20), IntValidatorRelaxed.Intermediate),
    ("-2", (0, 10), IntValidatorRelaxed.Invalid),
    ("12", (0, 10), IntValidatorRelaxed.Intermediate),
    ("12,", (0, 10), IntValidatorRelaxed.Invalid),  # Comma after the number
])
@pytest.mark.xfail(reason="Test fails with PyQT 5.9: remove 'xfail' once transition to conda-forge is complete")
# fmt: on
def test_IntValidatorStrict(text, range, result):
    validator = IntValidatorStrict()
    if range is not None:
        validator.setRange(range[0], range[1])
    assert validator.validate(text, 0)[0] == result, "Validation failed"


# ==============================================================
#   Class IntValidatorRelaxed

# fmt: off
@pytest.mark.parametrize("text, range, result", [
    ("", None, IntValidatorRelaxed.Intermediate),
    ("123", None, IntValidatorRelaxed.Acceptable),
    ("1&34", None, IntValidatorRelaxed.Intermediate),
    ("2", (0, 10), IntValidatorRelaxed.Acceptable),
    ("2", (10, 20), IntValidatorRelaxed.Intermediate),
    ("-2", (0, 10), IntValidatorRelaxed.Intermediate),
    ("12", (0, 10), IntValidatorRelaxed.Intermediate),
])
# fmt: on
def test_IntValidatorRelaxed_1(text, range, result):
    validator = IntValidatorRelaxed()
    if range is not None:
        validator.setRange(range[0], range[1])
    assert validator.validate(text, 0)[0] == result, "Validation failed"


# fmt: off
@pytest.mark.parametrize("text, range, valid", [
    ("", None, False),
    ("123", None, True),
    ("12&3", None, False),
    ("2", (0, 10), True),
    ("2", (10, 20), False),
    ("-2", (0, 10), False),
    ("12", (0, 10), False),

])
# fmt: on
def test_IntValidatorRelaxed_2(qtbot, text, range, valid):
    """
    Test how `IntValidatorRelaxed` operates with `QLineEdit`, which is expected
    to emit the signal `editingFinished()` only when the validator returns `Acceptable`.
    The typed string is supposed to be displayed (returned by `text()` method).
    """
    le = QLineEdit()
    validator = IntValidatorRelaxed()
    le.setValidator(validator)
    qtbot.addWidget(le)
    le.show()

    if range is not None:
        validator.setRange(range[0], range[1])

    def fn():
        enter_text_via_keyboard(qtbot, le, text)

    signal = le.editingFinished
    if valid:
        # 'editingFinished()' should be emitted
        with qtbot.waitSignal(signal, timeout=100):
            fn()
    else:
        # 'editingFinished()' should not be emitted
        with qtbot.assertNotEmitted(signal, wait=100):
            fn()

    # We verify that the text can be entered via keyboard is still displayed
    assert le.text() == text, "Entered and displayed text does not match"


# ==============================================================
#   Class DoubleValidatorStrict
# fmt: off
@pytest.mark.parametrize("text, range, result", [
    ("", None, DoubleValidatorRelaxed.Intermediate),
    ("123", None, DoubleValidatorRelaxed.Acceptable),
    ("123.34562", None, DoubleValidatorRelaxed.Acceptable),
    ("1.3454e10", None, DoubleValidatorRelaxed.Acceptable),
    ("1&34.34", None, DoubleValidatorRelaxed.Invalid),
    ("2.0", (0.0, 10.0), DoubleValidatorRelaxed.Acceptable),
    ("2.0", (10.0, 20.0), DoubleValidatorRelaxed.Intermediate),
    ("-2.342", (0, 10.0), DoubleValidatorRelaxed.Invalid),
    ("12.453", (0, 10.0), DoubleValidatorRelaxed.Intermediate),
    ("12.453,", (0, 10.0), DoubleValidatorRelaxed.Invalid),  # Comma after the number
])
# fmt: on
def test_DoubleValidatorStrict(text, range, result):
    validator = QDoubleValidator()
    if range is not None:
        validator.setRange(range[0], range[1], 5)
    assert validator.validate(text, 0)[0] == result, "Validation failed"


# ==============================================================
#   Class DoubleValidatorRelaxed

# fmt: off
@pytest.mark.parametrize("text, range, result", [
    ("", None, DoubleValidatorRelaxed.Intermediate),
    ("123", None, DoubleValidatorRelaxed.Acceptable),
    ("123.34562", None, DoubleValidatorRelaxed.Acceptable),
    ("1.3454e10", None, DoubleValidatorRelaxed.Acceptable),
    ("1&34.34", None, DoubleValidatorRelaxed.Intermediate),
    ("2.0", (0.0, 10.0), DoubleValidatorRelaxed.Acceptable),
    ("2.0", (10.0, 20.0), DoubleValidatorRelaxed.Intermediate),
    ("-2.342", (0, 10.0), DoubleValidatorRelaxed.Intermediate),
    ("12.453", (0, 10.0), DoubleValidatorRelaxed.Intermediate),
])
# fmt: on
def test_DoubleValidatorRelaxed_1(text, range, result):
    validator = DoubleValidatorRelaxed()
    if range is not None:
        validator.setRange(range[0], range[1], decimals=10)
    assert validator.validate(text, 0)[0] == result, "Validation failed"


# fmt: off
@pytest.mark.parametrize("text, range, valid", [
    ("", None, False),
    ("123", None, True),
    ("123.34562", None, True),
    ("1.3454e10", None, True),
    ("1&34.34", None, False),
    ("2.0", (0, 10.0), True),
    ("2.0", (10.0, 20.0), False),
    ("-2.342", (0, 10.0), False),
    ("12.453", (0, 10.0), False),
])
# fmt: on
def test_DoubleValidatorRelaxed_2(qtbot, text, range, valid):
    """
    Test how `DoubleValidatorRelaxed` operates with `QLineEdit`, which is expected
    to emit the signal `editingFinished()` only when the validator returns `Acceptable`.
    The typed string is supposed to be displayed (returned by `text()` method).
    """
    le = QLineEdit()
    validator = DoubleValidatorRelaxed()
    le.setValidator(validator)
    qtbot.addWidget(le)
    le.show()

    if range is not None:
        validator.setRange(range[0], range[1], decimals=10)

    def fn():
        enter_text_via_keyboard(qtbot, le, text)

    signal = le.editingFinished
    if valid:
        # 'editingFinished()' should be emitted
        with qtbot.waitSignal(signal, timeout=100):
            fn()
    else:
        # 'editingFinished()' should not be emitted
        with qtbot.assertNotEmitted(signal, wait=100):
            fn()

    # We verify that the text can be entered via keyboard is still displayed
    assert le.text() == text, "Entered and displayed text does not match"


# ==============================================================
#   Class RangeManager


def test_RangeManager_1(qtbot):
    """
    RangeManager: test the function for setting and changing value type
    """
    rman = RangeManager()
    qtbot.addWidget(rman)
    rman.show()

    def _compare_tuples(*, returned, expected, v_type):
        assert len(returned) == len(expected), "Returned selection has wrong number of elements"
        for v in returned:
            assert isinstance(v, v_type), f"Returned value has wrong type: {type(v)})"
        npt.assert_array_almost_equal(
            returned, expected, err_msg="Returned and original selection are not identical"
        )

    # The value type is 'float' by default
    f_range = (25.123, 89.892)
    f_selection = (32.65, 47.2)
    rman.set_range(f_range[0], f_range[1])
    rman.set_selection(value_low=f_selection[0], value_high=f_selection[1])

    # Read the selection back
    _compare_tuples(returned=rman.get_selection(), expected=f_selection, v_type=float)
    _compare_tuples(returned=rman.get_range(), expected=f_range, v_type=float)

    # Switch value type to "int". Selection is expected to change because due to rouning.
    # The selection is also expected to change because the new selection is supposed to
    #   cover the same fraction of the total range, but the numbers were selected so
    #   this does not affect the result.
    selection_changed = rman.set_value_type("int")
    assert selection_changed, "Selection was not changed (expected to change)"

    i_range = tuple(round(_) for _ in f_range)
    i_selection = tuple(round(_) for _ in f_selection)

    _compare_tuples(returned=rman.get_selection(), expected=i_selection, v_type=int)
    _compare_tuples(returned=rman.get_range(), expected=i_range, v_type=int)

    # Switch value type back to "float"
    selection_changed = rman.set_value_type("float")
    assert not selection_changed, "Selection changed (not expected to change)"

    _compare_tuples(returned=rman.get_selection(), expected=i_selection, v_type=float)
    _compare_tuples(returned=rman.get_range(), expected=i_range, v_type=float)


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-14.86, -5.2), (-11.78, -7.9), "float"),
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((10.4, 37.45), (13.45, 20.45), "float"),
    ((0, 37.45), (13.45, 20.45), "float"),
    ((-49, -5), (-30, -20), "int"),
    ((-49, 90), (-30, -20), "int"),
    ((5, 90), (10, 70), "int"),
    ((0, 90), (10, 70), "int"),
])
# fmt: on
def test_RangeManager_2(qtbot, full_range, selection, value_type):
    """Test the `reset()` method: resetting the selection"""
    rman = RangeManager()
    qtbot.addWidget(rman)
    rman.show()

    rman.set_value_type(value_type)
    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])

    npt.assert_array_almost_equal(rman.get_range(), full_range, err_msg="Range is set incorrectly")

    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{selection[0]:.10g}", "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{selection[1]:.10g}", "Upper boundary is displayed incorrectly"
    npt.assert_array_almost_equal(rman.get_selection(), selection, err_msg="Selection is set incorreclty")

    selection_changed = rman.reset()
    assert selection_changed is True, "Change of selection is incorrectly reported"

    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{full_range[0]:.10g}", "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{full_range[1]:.10g}", "Upper boundary is displayed incorrectly"
    npt.assert_array_almost_equal(rman.get_selection(), full_range, err_msg="Selection is set incorreclty")

    # Attemtp to reset again (selection shouldn't change)
    selection_changed = rman.reset()
    assert selection_changed is False, "Change of selection is incorrectly reported"


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-0.254, 37.45), (-0.5, 90.45), "float"),
    ((-0.254, 37.45), (10.25, 10.25), "float"),
    ((-0.254, 37.45), (10.25, 9.25), "float"),
    ((-0.254, 37.45), (37.45, 37.45), "float"),
    ((-0.254, 37.45), (-0.254, -0.254), "float"),
    ((-0.254, 37.45), (50.34, 60.4), "float"),  # Selection is 'above' the range
    ((-0.254, 37.45), (-50.34, -40.4), "float"),  # Selection is 'below' the range
    ((-49, 90), (-20, 60), "int"),
    ((-49, 90), (-100, 100), "int"),
    ((-49, 90), (60, 60), "int"),
    ((-49, 90), (60, 40), "int"),
    ((-49, 90), (90, 90), "int"),
    ((-49, 90), (-49, -49), "int"),
    ((-49, 90), (100, 110), "int"),  # Selection is 'above' the range
    ((-49, 90), (-100, -90), "int"),  # Selection is 'below' the range
])
# fmt: on
def test_RangeManager_3(qtbot, full_range, selection, value_type):
    """Test `set_selection()` method"""

    slider_steps = 1000
    selection_to_range_min = 0.01

    rman = RangeManager(slider_steps=slider_steps, selection_to_range_min=selection_to_range_min)
    qtbot.addWidget(rman)
    rman.show()

    rman.set_value_type(value_type)
    rman.set_range(full_range[0], full_range[1])

    def _clip_selection(sel, rng, selection_to_range_min):
        """
        Clip the selection values to compute the expected selection.
        Reproduces the clipping performed by the RangeManager.
        """
        sel = [max(min(_, rng[1]), rng[0]) for _ in sel]
        if value_type == "float":
            min_diff = (rng[1] - rng[0]) * selection_to_range_min
        else:
            min_diff = 1
        if sel[1] - sel[0] < min_diff:
            sel[1] = sel[0] + min_diff
            if sel[1] > rng[1]:
                sel = [rng[1] - min_diff, rng[1]]
        return sel

    sel = _clip_selection(selection, full_range, selection_to_range_min)

    result_expected = True
    if tuple(sel) == tuple(full_range):
        result_expected = False

    result = rman.set_selection(value_low=selection[0], value_high=selection[1])
    assert result == result_expected, f"Incorrect return value {result} by `set_selection()` method"

    npt.assert_array_almost_equal(rman.get_selection(), sel, err_msg="Selection is set incorrectly")
    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{sel[0]:.10g}", "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{sel[1]:.10g}", "Upper boundary is displayed incorrectly"

    # Check positions of the sliders
    step = (full_range[1] - full_range[0]) / slider_steps
    assert rman.sld_min_value.value() == slider_steps - round(
        (sel[0] - full_range[0]) / step
    ), "Slider position (min. value) is incorrect"
    assert rman.sld_max_value.value() == round(
        (sel[1] - full_range[0]) / step
    ), "Slider position (max. value) is incorrect"


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-49, 90), (-20, 60), "int"),
])
# fmt: on
def test_RangeManager_4(qtbot, full_range, selection, value_type):
    """
    RangeManager: additional testing of `set_selection()` method:
    change only higher or lower boundaries of the selection.
    """
    rman = RangeManager()
    qtbot.addWidget(rman)
    rman.show()

    def _verify_selection(sel):
        # Verify that selection is set correctly
        npt.assert_array_almost_equal(rman.get_selection(), sel, err_msg="Selection is set incorrectly")
        # Verify that selection is displayed correctly
        assert rman.le_min_value.text() == f"{sel[0]:.10g}", "Lower boundary is displayed incorrectly"
        assert rman.le_max_value.text() == f"{sel[1]:.10g}", "Upper boundary is displayed incorrectly"

    rman.set_value_type(value_type)
    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])
    _verify_selection(selection)

    # Change lower boundary
    sel = (selection[0] + 1, selection[1])
    result = rman.set_selection(value_low=sel[0])
    assert result is True, f"Incorrect return value {result} by `set_selection()` method"
    _verify_selection(sel)

    # Change upper boundary
    sel = (sel[0], sel[1] - 1)
    result = rman.set_selection(value_high=sel[1])
    assert result is True, f"Incorrect return value {result} by `set_selection()` method"
    _verify_selection(sel)


# fmt: off
@pytest.mark.parametrize("full_range, selection, new_range, value_type", [
    # Selection fits in both ranges
    ((-0.254, 37.45), (10.123, 20.45), (6.23, 29.14), "float"),
    ((-49, 90), (-20, 60), (-30, 70), "int"),
    # Upper boundary of the selection doesn't fit the new range
    ((-0.254, 37.45), (10.123, 20.45), (6.23, 15.3), "float"),
    ((-49, 90), (-20, 60), (-30, 55), "int"),
    # Lower boundary of the selection doesn't fit the new range
    ((-0.254, 37.45), (10.123, 20.45), (15.23, 29.14), "float"),
    ((-49, 90), (-20, 60), (-10, 70), "int"),
    # Both selection boundaries are outside the range
    ((-0.254, 37.45), (10.123, 20.45), (12.23, 18.3), "float"),
    ((-49, 90), (-20, 60), (-10, 55), "int"),
    # The range is shifted above the selection region
    ((-0.254, 37.45), (10.123, 20.45), (60.23, 70.3), "float"),
    ((-49, 90), (-20, 60), (100, 120), "int"),
    # The range is shifted below the selection region
    ((-0.254, 37.45), (10.123, 20.45), (-70.23, -60.3), "float"),
    ((-49, 90), (-20, 60), (-120, -100), "int"),
])
# fmt: on
def test_RangeManager_5(qtbot, full_range, selection, new_range, value_type):
    """
    RangeManager: additional testing of `set_selection()` method:
    change only higher or lower boundaries of the selection.
    """
    slider_steps = 1000
    selection_to_range_min = 0.01

    rman = RangeManager(slider_steps=slider_steps, selection_to_range_min=selection_to_range_min)
    qtbot.addWidget(rman)
    rman.show()

    def _verify_selection(sel, rng):
        # Verify that selection is set correctly
        npt.assert_array_almost_equal(rman.get_selection(), sel, err_msg="Selection is set incorrectly")
        # Verify that selection is displayed correctly
        assert rman.le_min_value.text() == f"{sel[0]:.8g}", "Lower boundary is displayed incorrectly"
        assert rman.le_max_value.text() == f"{sel[1]:.8g}", "Upper boundary is displayed incorrectly"
        # Check positions of the sliders
        step = (rng[1] - rng[0]) / slider_steps
        assert rman.sld_min_value.value() == slider_steps - round(
            (sel[0] - rng[0]) / step
        ), "Slider position (min. value) is incorrect"
        assert rman.sld_max_value.value() == round(
            (sel[1] - rng[0]) / step
        ), "Slider position (max. value) is incorrect"

    def _verify_range(rng):
        # Verify that selection is set correctly
        npt.assert_array_almost_equal(rman.get_range(), rng, err_msg="Range is set incorrectly")

    # Set the range and verify that the selection was scaled correctly
    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])
    _verify_selection(selection, full_range)
    _verify_range(full_range)

    def _scale_selection(val, old_rng, new_rng):
        """
        Scale the selection boundary. Reproduces computations performed by
        RangeManager when changing the range
        """
        val -= old_rng[0]
        val *= (new_rng[1] - new_rng[0]) / (old_rng[1] - old_rng[0])
        val += new_rng[0]
        return val

    def _clip_selection(sel, rng, selection_to_range_min):
        """
        Clip the selection values to compute the expected selection.
        Reproduces the clipping performed by the RangeManager.
        """
        sel = [max(min(_, rng[1]), rng[0]) for _ in sel]
        if value_type == "float":
            min_diff = (rng[1] - rng[0]) * selection_to_range_min
        else:
            min_diff = 1
        if sel[1] - sel[0] < min_diff:
            sel[1] = sel[0] + min_diff
            if sel[1] > rng[1]:
                sel = [rng[1] - min_diff, rng[1]]
        return sel

    # Scale the selection
    sel = [_scale_selection(_, full_range, new_range) for _ in selection]
    sel = _clip_selection(sel, new_range, selection_to_range_min)

    # Set the new range
    result = rman.set_range(new_range[0], new_range[1])
    assert result is True, f"Incorrect value returned by RangeManager.set_range(): {result}"
    _verify_selection(sel, new_range)
    _verify_range(new_range)

    # Set the same range (shouldn't change the selection)
    result = rman.set_range(new_range[0], new_range[1])
    assert result is False, f"Incorrect value returned by RangeManager.set_range(): {result}"
    _verify_selection(sel, new_range)
    _verify_range(new_range)


@pytest.mark.parametrize("add_sliders", [None, True, False])
def test_RangeManager_6(qtbot, add_sliders):
    """
    Create RangeManager with/without sliders
    """
    # Sliders don't exist only if 'add_sliders' is False
    sliders_exist = True if add_sliders is True else False

    kwargs = {}
    if add_sliders is not None:  # If None then use default (False)
        kwargs.update({"add_sliders": add_sliders})

    rman = RangeManager(**kwargs)
    qtbot.addWidget(rman)
    rman.show()

    def _check_slider_status(slider, status):
        # Check if slider has parent (it was added to layout)
        assert (slider.parent() is not None) == status, "Slider visibility is set incorrectly"

    _check_slider_status(rman.sld_min_value, sliders_exist)
    _check_slider_status(rman.sld_max_value, sliders_exist)


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-49, 90), (-20, 60), "int"),
])
# fmt: on
def test_RangeManager_7(qtbot, full_range, selection, value_type):
    """Check if the signal `selection_changed` is emitted correctly"""

    rman = RangeManager()
    qtbot.addWidget(rman)
    rman.show()

    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])

    def _check_signal_params(low, high, name):
        return low == low and high == high

    with qtbot.waitSignal(rman.selection_changed, check_params_cb=_check_signal_params):
        rman.emit_selection_changed()


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-49, 90), (-20, 60), "int"),
])
# fmt: on
def test_RangeManager_8(qtbot, full_range, selection, value_type):
    """Entering selection boundaries via keyboard"""

    slider_steps = 1000
    selection_to_range_min = 0.01

    rman = RangeManager(slider_steps=slider_steps, selection_to_range_min=selection_to_range_min)
    qtbot.addWidget(rman)
    rman.show()

    def _verify_sliders(slider_low, slider_high, sel, rng):
        step = (rng[1] - rng[0]) / slider_steps
        # 'low' value
        pos_expected = slider_steps - round((sel[0] - rng[0]) / step)
        assert slider_low.value() == pos_expected, "Lower boundary slider value (position) is incorrect"
        # 'high' value
        pos_expected = round((sel[1] - rng[0]) / step)
        assert slider_high.value() == pos_expected, "Higher boundary slider value (position) is incorrect"

    rman.set_range(full_range[0], full_range[1])
    rman.reset()  # Select the full range

    # This variable will hold the parameters that were passed with the signal
    returned_sig_parameters = None

    def _check_signal_params(low, high, name):
        # Instead of verification of parameters, this function copies the parameters
        #   into a variable `returned_sig_parameters`. If the verification is performed
        #   inside the function (as intended), then no nice error message is printed.
        nonlocal returned_sig_parameters
        returned_sig_parameters = (low, high)
        return True

    # Enter lower boundary
    with qtbot.waitSignal(rman.selection_changed, check_params_cb=_check_signal_params):
        enter_text_via_keyboard(qtbot, rman.le_min_value, f"{selection[0]:.10g}", finish=True)
    npt.assert_array_almost_equal(returned_sig_parameters, (selection[0], full_range[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], full_range[1]), full_range)

    # Enter upper boundary
    with qtbot.waitSignal(rman.selection_changed, check_params_cb=_check_signal_params):
        enter_text_via_keyboard(qtbot, rman.le_max_value, f"{selection[1]:.10g}", finish=True)
    npt.assert_array_almost_equal(returned_sig_parameters, (selection[0], selection[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], selection[1]), full_range)

    # Enter out-of-range lower boundary. Shouldn't generate the signal.
    with qtbot.assertNotEmitted(rman.selection_changed):
        enter_text_via_keyboard(qtbot, rman.le_min_value, f"{selection[1] + 1:.10g}", finish=True)
    npt.assert_array_almost_equal(rman.get_selection(), (selection[0], selection[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], selection[1]), full_range)

    # Enter some random string. Shouldn't generate the signal.
    with qtbot.assertNotEmitted(rman.selection_changed):
        enter_text_via_keyboard(qtbot, rman.le_min_value, "random string", finish=True)
    npt.assert_array_almost_equal(rman.get_selection(), (selection[0], selection[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], selection[1]), full_range)

    # Enter out-of-range upper boundary. Shouldn't generate the signal.
    with qtbot.assertNotEmitted(rman.selection_changed):
        enter_text_via_keyboard(qtbot, rman.le_max_value, f"{selection[0] - 1:.10g}", finish=True)
    npt.assert_array_almost_equal(rman.get_selection(), (selection[0], selection[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], selection[1]), full_range)

    # Enter some random string. Shouldn't generate the signal.
    with qtbot.assertNotEmitted(rman.selection_changed):
        enter_text_via_keyboard(qtbot, rman.le_max_value, "random string", finish=True)
    npt.assert_array_almost_equal(rman.get_selection(), (selection[0], selection[1]))
    _verify_sliders(rman.sld_min_value, rman.sld_max_value, (selection[0], selection[1]), full_range)


# fmt: off
@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-49, 90), (-20, 60), "int"),
])
# fmt: on
def test_RangeManager_9(qtbot, full_range, selection, value_type):
    """Entering invalid value in an entry field (especially the case when text contains `,`)"""

    slider_steps = 1000
    selection_to_range_min = 0.01

    rman = RangeManager(slider_steps=slider_steps, selection_to_range_min=selection_to_range_min)
    qtbot.addWidget(rman)
    rman.show()

    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])

    assert rman.get_selection() == selection, "Incorrect selection"

    enter_text_via_keyboard(qtbot, rman.le_min_value, "abc", finish=True)
    assert rman.get_selection() == selection, "Incorrect selection"

    enter_text_via_keyboard(qtbot, rman.le_min_value, "10,", finish=True)
    assert rman.get_selection() == selection, "Incorrect selection"

    enter_text_via_keyboard(qtbot, rman.le_max_value, "abc", finish=True)
    assert rman.get_selection() == selection, "Incorrect selection"

    enter_text_via_keyboard(qtbot, rman.le_max_value, "10.0,", finish=True)
    assert rman.get_selection() == selection, "Incorrect selection"
