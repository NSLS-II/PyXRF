import pytest
import numpy.testing as npt
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import Qt
from pyxrf.gui_module.useful_widgets import (
    IntValidatorRelaxed, DoubleValidatorRelaxed, RangeManager)


def enter_text_via_keyboard(qtbot, widget, text, *, finish=True):
    """Use keyboard to enter values line edit box"""
    qtbot.mouseDClick(widget, Qt.LeftButton)
    qtbot.keyClicks(widget, text)
    if finish:
        qtbot.keyClick(widget, Qt.Key_Enter)


# ==============================================================
#   Class IntValidatorRelaxed

@pytest.mark.parametrize("text, range, result", [
    ("", None, IntValidatorRelaxed.Intermediate),
    ("123", None, IntValidatorRelaxed.Acceptable),
    ("1&34", None, IntValidatorRelaxed.Intermediate),
    ("2", (0, 10), IntValidatorRelaxed.Acceptable),
    ("2", (10, 20), IntValidatorRelaxed.Intermediate),
    ("-2", (0, 10), IntValidatorRelaxed.Intermediate),
    ("12", (0, 10), IntValidatorRelaxed.Intermediate),
])
def test_IntValidatorRelaxed_1(text, range, result):
    validator = IntValidatorRelaxed()
    if range is not None:
        validator.setRange(range[0], range[1])
    assert validator.validate(text, 0)[0] == result, "Validation failed"


@pytest.mark.parametrize("text, range, valid", [
    ("", None, False),
    ("123", None, True),
    ("12&3", None, False),
    ("2", (0, 10), True),
    ("2", (10, 20), False),
    ("-2", (0, 10), False),
    ("12", (0, 10), False),

])
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
#   Class DoubleValidatorRelaxed

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
def test_DoubleValidatorRelaxed_1(text, range, result):
    validator = DoubleValidatorRelaxed()
    if range is not None:
        validator.setRange(range[0], range[1], decimals=10)
    assert validator.validate(text, 0)[0] == result, "Validation failed"


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
        assert len(returned) == len(expected), \
            "Returned selection has wrong number of elements"
        for v in returned:
            assert isinstance(v, v_type), f"Returned value has wrong type: {type(v)})"
        npt.assert_array_almost_equal(
            returned, expected, err_msg="Returned and original selection are not identical")

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
def test_RangeManager_2(qtbot, full_range, selection, value_type):
    """Test the `reset()` method: resetting the selection"""
    rman = RangeManager()
    qtbot.addWidget(rman)
    rman.show()

    rman.set_value_type(value_type)
    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])

    npt.assert_array_almost_equal(rman.get_range(), full_range,
                                  err_msg="Range is set incorrectly")

    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{selection[0]:.10g}", \
        "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{selection[1]:.10g}", \
        "Upper boundary is displayed incorrectly"
    npt.assert_array_almost_equal(rman.get_selection(), selection,
                                  err_msg="Selection is set incorreclty")

    selection_changed = rman.reset()
    assert selection_changed is True, "Change of selection is incorrectly reported"

    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{full_range[0]:.10g}", \
        "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{full_range[1]:.10g}", \
        "Upper boundary is displayed incorrectly"
    npt.assert_array_almost_equal(rman.get_selection(), full_range,
                                  err_msg="Selection is set incorreclty")

    # Attemtp to reset again (selection shouldn't change)
    selection_changed = rman.reset()
    assert selection_changed is False, "Change of selection is incorrectly reported"


@pytest.mark.parametrize("full_range, selection, value_type", [
    ((-0.254, 37.45), (-0.123, 20.45), "float"),
    ((-0.254, 37.45), (-0.5, 90.45), "float"),
    ((-0.254, 37.45), (10.25, 10.25), "float"),
    ((-0.254, 37.45), (10.25, 9.25), "float"),
    ((-0.254, 37.45), (37.45, 37.45), "float"),
    ((-0.254, 37.45), (-0.254, -0.254), "float"),
    ((-49, 90), (-20, 60), "int"),
    ((-49, 90), (-100, 100), "int"),
    ((-49, 90), (60, 60), "int"),
    ((-49, 90), (60, 40), "int"),
    ((-49, 90), (90, 90), "int"),
    ((-49, 90), (-49, -49), "int"),
])
def test_RangeManager_3(qtbot, full_range, selection, value_type):
    """Test `set_selection()` method"""

    slider_steps = 1000
    selection_to_range_min = 0.01

    rman = RangeManager(slider_steps=slider_steps,
                        selection_to_range_min=selection_to_range_min)
    qtbot.addWidget(rman)
    rman.show()

    rman.set_value_type(value_type)
    rman.set_range(full_range[0], full_range[1])
    rman.set_selection(value_low=selection[0], value_high=selection[1])

    # Clip the selection values
    sel = [max(min(_, full_range[1]), full_range[0]) for _ in selection]
    if value_type == "float":
        min_diff = (full_range[1] - full_range[0]) * selection_to_range_min
    else:
        min_diff = 1
    if sel[1] - sel[0] < min_diff:
        sel[1] = sel[0] + min_diff
        if sel[1] > full_range[1]:
            sel = [full_range[1] - min_diff, full_range[1]]

    npt.assert_array_almost_equal(rman.get_selection(), sel,
                                  err_msg="Selection is set incorrectly")
    # Verify that selection is displayed correctly
    assert rman.le_min_value.text() == f"{sel[0]:.10g}", \
        "Lower boundary is displayed incorrectly"
    assert rman.le_max_value.text() == f"{sel[1]:.10g}", \
        "Upper boundary is displayed incorrectly"

    # Check positions of the sliders
    step = (full_range[1] - full_range[0]) / slider_steps
    assert rman.sld_min_value.value() == slider_steps - round((sel[0] - full_range[0]) / step), \
        "Slider position (min. value) is incorrect"
    assert rman.sld_max_value.value() == round((sel[1] - full_range[0]) / step), \
        "Slider position (max. value) is incorrect"
