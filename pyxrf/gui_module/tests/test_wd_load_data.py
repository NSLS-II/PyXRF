import os
import pytest
from pyxrf.gui_module.tab_wd_load_data import DialogLoadMask
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt


# =====================================================================
#                 class DialogLoadMask (QDialogBox)


def test_DialogLoadMask_1(qtbot):
    """Set/Read values from ROI widget group of DialogLoadMask object"""
    dlg = DialogLoadMask()
    qtbot.addWidget(dlg)
    dlg.show()

    # Set image size
    n_rows, n_columns = 15, 17
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)
    assert f"{n_rows} rows, {n_columns} columns" in dlg.label_map_size.text(), "Map size was not set correctly"

    def _check_roi_widgets(expected_values):
        l_edit_widgets = (dlg.le_roi_start_row, dlg.le_roi_start_col, dlg.le_roi_end_row, dlg.le_roi_end_col)
        for n, v in enumerate(expected_values):
            if n < 2:  # First 2 entries (are incremented when displayed)
                v += 1
            s = f"{v}" if v > 0 else ""
            assert l_edit_widgets[n].text() == s, "Line Edit text does not match the expected text"

    # Check if the default values were set correctly when image size was set
    #   (try to activate/deactivate the ROI selection group box)
    _check_roi_widgets([0, 0, n_rows, n_columns])  # Default values
    dlg.set_roi_active(True)  # Activate ROI group
    _check_roi_widgets([0, 0, n_rows, n_columns])  # Default values
    dlg.set_roi_active(False)
    _check_roi_widgets([0, 0, n_rows, n_columns])  # Default values

    # Set ROI. Initially the ROI group is inactive, so no text is displayed
    roi = (2, 3, 14, 15)
    dlg.set_roi(row_start=roi[0], column_start=roi[1], row_end=roi[2], column_end=roi[3])
    assert not dlg.gb_roi.isChecked(), "ROI group is checked, while it must be unchecked"
    assert dlg.get_roi_active() is False, "ROI 'enable' status was not read correctly"
    _check_roi_widgets(roi)
    # Activate ROI group
    dlg.set_roi_active(True)
    assert dlg.gb_roi.isChecked(), "ROI group is unchecked, while it must be checked"
    assert dlg.get_roi_active() is True, "ROI 'enable' status was not read correctly"
    _check_roi_widgets(roi)

    # Set the values outside the range. The values must be clipped
    roi = (0, 0, 50, 50)
    dlg.set_roi(row_start=roi[0], column_start=roi[1], row_end=roi[2], column_end=roi[3])
    assert dlg.get_roi() == (roi[0], roi[1], n_rows, n_columns), "Indexes were not clipped correctly"

    dlg.close()


def test_DialogLoadMask_2(qtbot):
    """
    DialogLoadMask - ROI selection group.
    In this test, the ROI selection is modified using keyboard clicks.
    Checked features: initalization, enabling/disabling ROI selection group, editing the ROI entries,
    synchronization of entered and stored values, showing/hiding the displayed values as
    the group is enabled/disabled.
    """
    dlg = DialogLoadMask()
    qtbot.addWidget(dlg)
    dlg.show()

    def _check_roi_widgets(expected_values):
        l_edit_widgets = (dlg.le_roi_start_row, dlg.le_roi_start_col, dlg.le_roi_end_row, dlg.le_roi_end_col)
        for n, v in enumerate(expected_values):
            if n < 2:  # First 2 entries (are incremented when displayed)
                v += 1
            s = f"{v}" if v > 0 else ""
            assert l_edit_widgets[n].text() == s, "Line Edit text does not match the expected text"

    # Set image size
    n_rows, n_columns = 25, 37
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)

    assert not dlg.gb_roi.isChecked(), "ROI selection group must be active"
    _check_roi_widgets([0, 0, n_rows, n_columns])
    dlg.gb_roi.setChecked(True)  # There seems to be no way to change the state of the group box by clicking
    assert dlg.gb_roi.isChecked(), "Failed to activate ROI group"
    _check_roi_widgets([0, 0, n_rows, n_columns])

    # Use keyboard to enter values in each box
    def _enter_value(edit_box, value):
        qtbot.mouseDClick(edit_box, Qt.LeftButton)
        qtbot.keyClicks(edit_box, f"{value}")
        qtbot.keyClick(edit_box, Qt.Key_Enter)

    # Enter a new ROI using keyboard
    sel = (11, 8, 23, 19)
    _enter_value(dlg.le_roi_start_row, sel[0] + 1)
    _enter_value(dlg.le_roi_start_col, sel[1] + 1)
    _enter_value(dlg.le_roi_end_row, sel[2])
    _enter_value(dlg.le_roi_end_col, sel[3])
    _check_roi_widgets(sel)
    # Now check if the values were set correctly
    assert dlg.get_roi() == sel, "ROI was incorrectly set"

    # Disable and then enable the ROI group again, the entered values should remain
    dlg.gb_roi.setChecked(False)
    assert dlg.get_roi_active() is False, "ROI group must be inactive"
    _check_roi_widgets(sel)  # Values must be hidden now
    assert dlg.get_roi() == sel, "Unexpected ROI selection values"
    dlg.gb_roi.setChecked(True)
    assert dlg.get_roi_active() is True, "ROI group must be active"
    _check_roi_widgets(sel)  # Values must be displayed
    assert dlg.get_roi() == sel, "Unexpected ROI selection values"


@pytest.mark.parametrize("use_keys_and_mouse", [False, True])
def test_DialogLoadMask_3(qtbot, use_keys_and_mouse):
    """
    DialogLoadMask - ROI selection group.
    Check if validation of dialog controls works as expected.
    In this test the values are set using dialog box functions, not using mouse/keyboard input.
    """

    dlg = DialogLoadMask()
    qtbot.addWidget(dlg)
    dlg.show()

    # Set image size
    n_rows, n_columns = 25, 37
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)

    def _set_roi_group_active(status):
        if use_keys_and_mouse:
            dlg.set_roi_active(status)
        else:
            dlg.set_roi_active(status)

    def _enter_value(edit_box, value):
        """Use keyboard to enter values line edit box"""
        qtbot.mouseDClick(edit_box, Qt.LeftButton)
        qtbot.keyClicks(edit_box, f"{value}")
        qtbot.keyClick(edit_box, Qt.Key_Enter)

    def _set_roi(roi):
        if use_keys_and_mouse:
            dlg.set_roi(row_start=roi[0], column_start=roi[1], row_end=roi[2], column_end=roi[3])
        else:
            _enter_value(dlg.le_roi_start_row, roi[0])
            _enter_value(dlg.le_roi_start_col, roi[1])
            _enter_value(dlg.le_roi_end_row, roi[2])
            _enter_value(dlg.le_roi_end_col, roi[3])

    def _check_controls(status):
        assert dlg.le_roi_start_row.isValid() is status[0], "Invalid status: ROI control - row start"
        assert dlg.le_roi_start_col.isValid() is status[1], "Invalid status: ROI control - column start"
        assert dlg.le_roi_end_row.isValid() is status[2], "Invalid status: ROI control - row end"
        assert dlg.le_roi_end_col.isValid() is status[3], "Invalid status: ROI control - column end"
        assert dlg.button_ok.isEnabled() is status[4], "Invalid status: Button Ok"

    # Activate ROI selection group
    _set_roi_group_active(True)

    # Set valid selection
    sel = (11, 8, 23, 19)
    _set_roi(sel)
    _check_controls([True, True, True, True, True])

    # Set invalid selection for rows
    sel = (23, 8, 4, 19)
    _set_roi(sel)
    _check_controls([False, True, False, True, False])

    # Set invalid selection for columns
    sel = (11, 19, 23, 5)
    _set_roi(sel)
    _check_controls([True, False, True, False, False])

    # Set invalid selection for both
    sel = (23, 19, 11, 5)
    _set_roi(sel)
    _check_controls([False, False, False, False, False])

    # Deactivate ROI selection group
    _set_roi_group_active(False)
    # Check only the status of Ok button
    assert dlg.button_ok.isEnabled() is True, "Invalid status: Button Ok"


@pytest.mark.parametrize("use_keys_and_mouse", [False, True])
def test_DialogLoadMask_4(qtbot, use_keys_and_mouse):
    """
    DialogLoadMask - ROI selection group.
    Check if validation of dialog controls works as expected.
    In this test the values are set using dialog box functions, not using mouse/keyboard input.
    """

    dlg = DialogLoadMask()
    qtbot.addWidget(dlg)
    dlg.show()

    # Set image size
    n_rows, n_columns = 25, 37
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)

    def _activate_mask_selection_group(state):
        if use_keys_and_mouse:
            dlg.gb_mask.setChecked(state)
        else:
            dlg.set_mask_file_active(state)
        assert dlg.get_mask_file_active() == state, "Mask selection group status is incorrect"
        assert dlg.gb_mask.isChecked() == state, "Mask selection group check status is incorrect"

    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(True)
    assert dlg.le_load_mask.isValid() is False, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(False)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"

    fpath = "/file/path/file_name.bin"  # File path doesn't have to be real

    # Set file name during 'inactive' state
    dlg.set_mask_file_path(fpath)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(True)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(False)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"

    # Set file name during 'inactive' state
    dlg.set_mask_file_path("")  # Clear the file path
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(True)
    assert dlg.le_load_mask.isValid() is False, "Validation status of the mask file name is incorrect"
    dlg.set_mask_file_path(fpath)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"
    _activate_mask_selection_group(False)
    assert dlg.le_load_mask.isValid() is True, "Validation status of the mask file name is incorrect"


def test_DialogLoadMask_5(qtbot):
    """Setting home directory"""

    dlg = DialogLoadMask()
    qtbot.addWidget(dlg)
    dlg.show()

    # Set image size
    n_rows, n_columns = 25, 37
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)

    assert dlg._compute_home_directory() == ".", "Home directory is computed incorrectly"
    d_name = os.path.join("file", "directory")
    d_name = os.path.abspath(d_name)
    dlg.set_default_directory(d_name)
    assert dlg.get_default_directory() == d_name, "Returned default directory is incorrect"
    assert dlg._compute_home_directory() == d_name, "Home directory is computed incorrectly"

    d_name2 = os.path.join("file", "directory")
    d_name2 = os.path.abspath(d_name2)
    dlg.set_default_directory(d_name2)
    f_path = os.path.join(d_name2, "file_name.bin")
    dlg.set_mask_file_path(f_path)
    assert dlg._compute_home_directory() == d_name2, "Home directory is computed incorrectly"

    # Now clear the default directory
    dlg.set_default_directory(d_name)
    assert dlg._compute_home_directory() == d_name2, "Home directory is computed incorrectly"


def test_DialogLoadMask_6(qtbot, monkeypatch):
    """Setting home directory"""

    dlg = DialogLoadMask()
    dlg.show()
    qtbot.addWidget(dlg)
    qtbot.waitForWindowShown(dlg)

    # Set image size
    n_rows, n_columns = 25, 37
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)

    d_name = os.path.join("file", "directory")
    d_name = os.path.abspath(d_name)
    f_name = "file_name.bin"
    d_path = os.path.join(d_name, f_name)
    dlg.set_default_directory(d_name)

    # Activate the group
    dlg.gb_mask.setChecked(True)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: [""])
    qtbot.mouseClick(dlg.pb_load_mask, Qt.LeftButton)
    assert dlg.le_load_mask.isValid() is False, "Incorrect state of the line edit widget"
    assert dlg.le_load_mask.text() == dlg._le_load_mask_default_text, "Line Edit text is set incorrectly"
    assert dlg.get_mask_file_path() == "", "Unexpected returned path for the mask file"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: [d_path])
    qtbot.mouseClick(dlg.pb_load_mask, Qt.LeftButton)
    assert dlg.le_load_mask.isValid() is True, "Incorrect state of the line edit widget"
    assert dlg.le_load_mask.text() == d_path, "Line Edit text is set incorrectly"
    assert dlg.get_mask_file_path() == d_path, "Unexpected returned path for the mask file"
