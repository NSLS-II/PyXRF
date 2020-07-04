import pytest
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import Qt
from pyxrf.gui_module.useful_widgets import (
    IntValidatorRelaxed, DoubleValidatorRelaxed)


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
