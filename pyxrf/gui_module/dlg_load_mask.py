import os
import numpy as np

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
)
from qtpy.QtGui import QIntValidator

from .useful_widgets import LineEditReadOnly, LineEditExtended, set_tooltip

import logging

logger = logging.getLogger(__name__)


class DialogLoadMask(QDialog):
    """
    Dialog box for selecting spatial ROI and mask.

    Typical use:

    default_dir = "/home/user/data"
    n_rows, n_columns = 15, 20  # Some values

    # Values that are changed by the dialog box
    roi = (2, 3, 11, 9)
    use_roi = True
    mask_f_path = ""
    use_mask = False

    dlg = DialogLoadMask()
    dlg.set_image_size(n_rows=n_rows, n_columns=n_columns)
    dlg.set_roi(row_start=roi[0], column_start=roi[1], row_end=roi[2], column_end=roi[3])
    dlg.set_roi_active(use_roi)

    dlg.set_default_directory(default_dir)
    dlg.set_mask_file_path(mask_f_path)
    dlg.set_mask_file_active(use_mask)

    if dlg.exec() == QDialog.Accepted:
        # If success, then read the values back. Discard changes if rejected.
        roi = dlg.get_roi()
        use_roi = dlg.get_roi_active()
        mask_f_path = dlg.get_mask_file_path()
        use_mask = dlg.get_mask_file_active()
    """

    def __init__(self):

        super().__init__()

        self._validation_enabled = False

        self.resize(500, 300)
        self.setWindowTitle("Load Mask or Select ROI")

        # Initialize variables used with ROI selection group
        self._n_rows = 0
        self._n_columns = 0
        self._roi_active = False
        self._row_start = -1
        self._column_start = -1
        self._row_end = -1
        self._column_end = -1
        # ... with Mask group
        self._mask_active = False
        self._mask_file_path = ""
        self._default_directory = ""

        # Fields for entering spatial ROI coordinates
        self.validator_rows = QIntValidator()
        self.validator_rows.setBottom(1)
        self.validator_cols = QIntValidator()
        self.validator_cols.setBottom(1)

        self.le_roi_start_row = LineEditExtended()
        self.le_roi_start_row.setValidator(self.validator_rows)
        self.le_roi_start_row.editingFinished.connect(self.le_roi_start_row_editing_finished)
        self.le_roi_start_row.textChanged.connect(self.le_roi_start_row_text_changed)
        self.le_roi_start_col = LineEditExtended()
        self.le_roi_start_col.setValidator(self.validator_cols)
        self.le_roi_start_col.editingFinished.connect(self.le_roi_start_col_editing_finished)
        self.le_roi_start_col.textChanged.connect(self.le_roi_start_col_text_changed)
        self.le_roi_end_row = LineEditExtended()
        self.le_roi_end_row.setValidator(self.validator_rows)
        self.le_roi_end_row.editingFinished.connect(self.le_roi_end_row_editing_finished)
        self.le_roi_end_row.textChanged.connect(self.le_roi_end_row_text_changed)
        self.le_roi_end_col = LineEditExtended()
        self.le_roi_end_col.setValidator(self.validator_cols)
        self.le_roi_end_col.editingFinished.connect(self.le_roi_end_col_editing_finished)
        self.le_roi_end_col.textChanged.connect(self.le_roi_end_col_text_changed)
        self._text_map_size_base = "   * Map size: "
        self.label_map_size = QLabel(self._text_map_size_base + "not set")

        # Group box for spatial ROI selection
        self.gb_roi = QGroupBox("Select ROI (in pixels)")
        set_tooltip(
            self.gb_roi,
            "Select rectangular <b>spatial ROI</b>. If <b>mask</b> is "
            "loaded, then ROI is applied to the masked data.",
        )
        self.gb_roi.setCheckable(True)
        self.gb_roi.toggled.connect(self.gb_roi_toggled)
        self.gb_roi.setChecked(self._roi_active)
        vbox = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("Start position(*):"), 0, 0)
        grid.addWidget(QLabel("row"), 0, 1)
        grid.addWidget(self.le_roi_start_row, 0, 2)
        grid.addWidget(QLabel("column"), 0, 3)
        grid.addWidget(self.le_roi_start_col, 0, 4)
        grid.addWidget(QLabel("End position(*):"), 1, 0)
        grid.addWidget(QLabel("row"), 1, 1)
        grid.addWidget(self.le_roi_end_row, 1, 2)
        grid.addWidget(QLabel("column"), 1, 3)
        grid.addWidget(self.le_roi_end_col, 1, 4)
        vbox.addLayout(grid)
        vbox.addWidget(self.label_map_size)
        self.gb_roi.setLayout(vbox)

        # Widgets for loading mask
        self.pb_load_mask = QPushButton("Load Mask ...")
        self.pb_load_mask.clicked.connect(self.pb_load_mask_clicked)
        self._le_load_mask_default_text = "select 'mask' file"
        self.le_load_mask = LineEditReadOnly(self._le_load_mask_default_text)

        # Group box for setting mask
        self.gb_mask = QGroupBox("Set mask")
        set_tooltip(
            self.gb_mask,
            "Load <b>mask</b> from file. Active pixels in the mask are "
            "represented by positive integers. If <b>spatial ROI</b> is "
            "selected, then it is applied to the masked data.",
        )
        self.gb_mask.setCheckable(True)
        self.gb_mask.toggled.connect(self.gb_mask_toggled)
        self.gb_mask.setChecked(self._mask_active)
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_mask)
        hbox.addWidget(self.le_load_mask)
        self.gb_mask.setLayout(hbox)

        # Yes/No button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_ok = self.button_box.button(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.gb_roi)
        vbox.addStretch(1)
        vbox.addWidget(self.gb_mask)
        vbox.addStretch(2)
        vbox.addWidget(self.button_box)
        self.setLayout(vbox)

        self._validation_enabled = True
        self._validate_all_widgets()

    def _compute_home_directory(self):
        dir_name = "."
        if self._default_directory:
            dir_name = self._default_directory
        if self._mask_file_path:
            d, _ = os.path.split(self._mask_file_path)
            dir_name = d if d else dir_name
        return dir_name

    def pb_load_mask_clicked(self):
        dir_name = self._compute_home_directory()
        file_name = QFileDialog.getOpenFileName(self, "Load Mask From File", dir_name, "All (*)")
        file_name = file_name[0]
        if file_name:
            self._mask_file_path = file_name
            self._show_mask_file_path()
            logger.debug(f"Mask file is selected: '{file_name}'")

    def gb_roi_toggled(self, state):
        self._roi_activate(state)

    def _read_le_roi_value(self, line_edit, v_default):
        """
        Attempt to read value from line edit box as int, return `v_default` if not successful.

        Parameters
        ----------
        line_edit: QLineEdit
            reference to QLineEdit object
        v_default: int
            default value returned if the value read from edit box is incorrect
        """
        try:
            val = int(line_edit.text())
            if val < 1:
                raise Exception()
        except Exception:
            val = v_default
        return val

    def le_roi_start_row_editing_finished(self):
        self._row_start = self._read_le_roi_value(self.le_roi_start_row, self._row_start + 1) - 1

    def le_roi_end_row_editing_finished(self):
        self._row_end = self._read_le_roi_value(self.le_roi_end_row, self._row_end)

    def le_roi_start_col_editing_finished(self):
        self._column_start = self._read_le_roi_value(self.le_roi_start_col, self._column_start + 1) - 1

    def le_roi_end_col_editing_finished(self):
        self._column_end = self._read_le_roi_value(self.le_roi_end_col, self._column_end)

    def le_roi_start_row_text_changed(self):
        self._validate_all_widgets()

    def le_roi_end_row_text_changed(self):
        self._validate_all_widgets()

    def le_roi_start_col_text_changed(self):
        self._validate_all_widgets()

    def le_roi_end_col_text_changed(self):
        self._validate_all_widgets()

    def gb_mask_toggled(self, state):
        self._mask_file_activate(state)

    def _validate_all_widgets(self):
        """
        Validate the values and state of all widgets, update the 'valid' state
        of the widgets and enable/disable Ok button.
        """

        if not self._validation_enabled:
            return

        # Check if all fields have valid input values
        def _check_valid_input(line_edit, flag_valid):
            val = self._read_le_roi_value(line_edit, -1)
            state = val > 0
            line_edit.setValid(state)
            flag_valid = flag_valid if state else False
            return val, flag_valid

        # Set all line edits to 'valid' state
        self.le_roi_start_row.setValid(True)
        self.le_roi_end_row.setValid(True)
        self.le_roi_start_col.setValid(True)
        self.le_roi_end_col.setValid(True)
        self.le_load_mask.setValid(True)

        flag_valid = True

        if self._roi_active:
            # Perform the following checks only if ROI group is active
            rs, flag_valid = _check_valid_input(self.le_roi_start_row, flag_valid)
            re, flag_valid = _check_valid_input(self.le_roi_end_row, flag_valid)
            cs, flag_valid = _check_valid_input(self.le_roi_start_col, flag_valid)
            ce, flag_valid = _check_valid_input(self.le_roi_end_col, flag_valid)

            # Check if start
            if (rs > 0) and (re > 0) and (rs > re):
                self.le_roi_start_row.setValid(False)
                self.le_roi_end_row.setValid(False)
                flag_valid = False
            if (cs > 0) and (ce > 0) and (cs > ce):
                self.le_roi_start_col.setValid(False)
                self.le_roi_end_col.setValid(False)
                flag_valid = False

        if self._mask_active:
            if not self._mask_file_path:
                self.le_load_mask.setValid(False)
                flag_valid = False

        self.button_ok.setEnabled(flag_valid)

    def set_image_size(self, *, n_rows, n_columns):
        """
        Set the image size. Image size is used to for input validation. When image
        size is set, the selection is reset to cover the whole image.

        Parameters
        ----------
        n_rows: int
            The number of rows in the image: (1..)
        n_columns: int
            The number of columns in the image: (1..)
        """
        if n_rows < 1 or n_columns < 1:
            raise ValueError(
                "DialogLoadMask: image size have zero rows or zero columns: "
                f"n_rows={n_rows} n_columns={n_columns}. "
                "Report the error to the development team."
            )

        self._n_rows = n_rows
        self._n_columns = n_columns

        self._row_start = 0
        self._row_end = n_rows
        self._column_start = 0
        self._column_end = n_columns

        self._show_selection(True)
        self._validate_all_widgets()

        self.validator_rows.setTop(n_rows)
        self.validator_cols.setTop(n_columns)
        # Set label
        self.label_map_size.setText(f"{self._text_map_size_base}{self._n_rows} rows, {self._n_columns} columns.")

    def _show_selection(self, visible):
        """visible: True - show values, False - hide values"""

        def _show_number(l_edit, value):
            if value > 0:
                l_edit.setText(f"{value}")
            else:
                # This would typically mean incorrect initialization of the dialog box
                l_edit.setText("")

        _show_number(self.le_roi_start_row, self._row_start + 1)
        _show_number(self.le_roi_start_col, self._column_start + 1)
        _show_number(self.le_roi_end_row, self._row_end)
        _show_number(self.le_roi_end_col, self._column_end)

    def set_roi(self, *, row_start, column_start, row_end, column_end):
        """
        Set the values of fields that define selection of the image region. First set
        the image size (`set_image_size()`) and then set the selection.

        Parameters
        ----------
        row_start: int
            The row number of the first pixel in the selection (0..n_rows-1).
            Negative (-1) - resets value to 0.
        column_start: int
            The column number of the first pixel in the selection (0..n_columns-1).
            Negative (-1) - resets value to 0.
        row_end: int
            The row number following the last pixel in the selection (1..n_rows).
            This row is not included in the selection. Negative (-1) - resets value to n_rows.
        column_end: int
            The column number following the last pixel in the selection (1..n_columns).
            This column is not included in the selection. Negative (-1) - resets value to n_columns.
        """
        # The variables holding the selected region are following Python conventions for the
        #   selections: row_start, column_start are in the range 0..n_rows-1, 0..n_columns-1
        #   and row_end, column_end are in the range 1..n_rows, 1..n_columns.
        #   The values displayed in the dialog box are just pixels numbers in the range
        #   1..n_rows, 1..n_columns that define the rectangle in the way intuitive to the user.
        self._row_start = int(np.clip(row_start, a_min=0, a_max=self._n_rows - 1))
        self._column_start = int(np.clip(column_start, a_min=0, a_max=self._n_columns - 1))

        def _adjust_last_index(index, n_elements):
            if index < 0:
                index = n_elements
            index = int(np.clip(index, 1, n_elements))
            return index

        self._row_end = _adjust_last_index(row_end, self._n_rows)
        self._column_end = _adjust_last_index(column_end, self._n_columns)
        self._show_selection(self._roi_active)
        self._validate_all_widgets()

    def _roi_activate(self, state):
        self._roi_active = state
        self._show_selection(state)
        self._validate_all_widgets()

    def set_roi_active(self, state):
        self._roi_activate(state)
        self.gb_roi.setChecked(self._roi_active)

    def get_roi(self):
        return self._row_start, self._column_start, self._row_end, self._column_end

    def get_roi_active(self):
        return self._roi_active

    def _show_mask_file_path(self):
        fpath = self._mask_file_path if self._mask_file_path else self._le_load_mask_default_text
        self.le_load_mask.setText(fpath)
        self._validate_all_widgets()

    def _mask_file_activate(self, state):
        self._mask_active = state
        self._show_mask_file_path()
        self._validate_all_widgets()

    def set_mask_file_active(self, state):
        self._mask_file_activate(state)
        self.gb_mask.setChecked(self._mask_active)

    def get_mask_file_active(self):
        return self._mask_active

    def set_mask_file_path(self, fpath):
        self._mask_file_path = fpath
        self._show_mask_file_path()

    def get_mask_file_path(self):
        return self._mask_file_path

    def set_default_directory(self, dir_name):
        self._default_directory = dir_name

    def get_default_directory(self):
        return self._default_directory
