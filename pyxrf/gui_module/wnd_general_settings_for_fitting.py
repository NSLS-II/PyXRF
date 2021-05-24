from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QLabel,
    QGridLayout,
    QMessageBox,
)
from qtpy.QtCore import Qt, Slot, Signal, QThreadPool, QRunnable
from .useful_widgets import (
    set_tooltip,
    SecondaryWindow,
    LineEditExtended,
    IntValidatorStrict,
    DoubleValidatorStrict,
)

import logging

logger = logging.getLogger(__name__)


class WndGeneralFittingSettings(SecondaryWindow):

    # Signal that is sent (to main window) to update global state of the program
    update_global_state = Signal()
    computations_complete = Signal(object)

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self._dialog_data = {}

        self._validator_int = IntValidatorStrict()
        self._validator_float = DoubleValidatorStrict()

        # Reference to the main window. The main window will hold
        #   references to all non-modal windows that could be opened
        #   from multiple places in the program.
        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.update_global_state.connect(self.ref_main_window.update_widget_state)

        self.initialize()

        self._data_changed = False

    def initialize(self):
        self.setWindowTitle("General Settings for Fitting Algorithm")

        self.setMinimumHeight(330)
        self.setMinimumWidth(500)
        self.resize(650, 330)

        self.pb_apply = QPushButton("Apply")
        self.pb_apply.setEnabled(False)
        self.pb_apply.clicked.connect(self.pb_apply_clicked)
        self.pb_cancel = QPushButton("Cancel")
        self.pb_cancel.setEnabled(False)
        self.pb_cancel.clicked.connect(self.pb_cancel_clicked)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.pb_apply)
        hbox.addWidget(self.pb_cancel)
        vbox.addLayout(hbox)

        hbox = self._setup_table()
        vbox.addLayout(hbox)

        vbox.addStretch(1)

        self.setLayout(vbox)

        self._set_tooltips()
        self.update_form_data()

    def _setup_table(self):

        vbox_left = QVBoxLayout()

        # ===== Top-left section of the dialog box =====
        self.le_max_iterations = LineEditExtended()
        self.le_tolerance_stopping = LineEditExtended()
        self.le_escape_ratio = LineEditExtended()
        grid = QGridLayout()
        grid.addWidget(QLabel("Iterations (max):"), 0, 0)
        grid.addWidget(self.le_max_iterations, 0, 1)
        grid.addWidget(QLabel("Tolerance (stopping):"), 1, 0)
        grid.addWidget(self.le_tolerance_stopping, 1, 1)
        grid.addWidget(QLabel("Escape peak ratio:"), 2, 0)
        grid.addWidget(self.le_escape_ratio, 2, 1)
        self.group_total_spectrum_fitting = QGroupBox("Fitting of Total Spectrum (Model)")
        self.group_total_spectrum_fitting.setLayout(grid)
        vbox_left.addWidget(self.group_total_spectrum_fitting)

        # ===== Bottom-left section of the dialog box =====

        # Incident energy and the selected range
        self.le_incident_energy = LineEditExtended()
        self.le_range_low = LineEditExtended()
        self.le_range_high = LineEditExtended()
        self.group_energy_range = QGroupBox("Incident Energy and Selected Range")
        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, keV"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(QLabel("Range (low), keV"), 1, 0)
        grid.addWidget(self.le_range_low, 1, 1)
        grid.addWidget(QLabel("Range (high), keV"), 2, 0)
        grid.addWidget(self.le_range_high, 2, 1)
        self.group_energy_range.setLayout(grid)
        vbox_left.addWidget(self.group_energy_range)

        vbox_right = QVBoxLayout()

        # ===== Top-right section of the dialog box =====

        self.cb_linear_baseline = QCheckBox("Subtract linear baseline")
        self.cb_snip_baseline = QCheckBox("Subtract baseline using SNIP")

        # This option is not supported. In the future it may be removed
        #   if not needed or implemented.
        self.cb_add_const_to_data = QCheckBox("Add const. bias to data")
        self.cb_add_const_to_data.setEnabled(False)
        self.lb_add_const_to_data = QLabel("Constant bias:")
        self.lb_add_const_to_data.setEnabled(False)
        self.le_add_const_to_data = LineEditExtended()
        self.le_add_const_to_data.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.cb_linear_baseline)
        vbox.addWidget(self.cb_snip_baseline)
        vbox.addWidget(self.cb_add_const_to_data)
        hbox = QHBoxLayout()
        hbox.addWidget(self.lb_add_const_to_data)
        hbox.addWidget(self.le_add_const_to_data)
        vbox.addLayout(hbox)

        self.group_pixel_fitting = QGroupBox("Fitting of Single Spectra (XRF Maps)")
        self.group_pixel_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_pixel_fitting)

        # ===== Bottom-right section of the dialog box =====

        self.le_snip_window_size = LineEditExtended()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("SNIP window size(*):"))
        hbox.addWidget(self.le_snip_window_size)
        vbox.addLayout(hbox)
        vbox.addWidget(QLabel("*Total spectrum fitting always includes \n    SNIP baseline subtraction"))

        self.group_all_fitting = QGroupBox("All Fitting")
        self.group_all_fitting.setLayout(vbox)

        vbox_right.addWidget(self.group_all_fitting)
        vbox_right.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)

        self.le_max_iterations.textChanged.connect(self.le_max_iterations_text_changed)
        self.le_max_iterations.editingFinished.connect(self.le_max_iterations_editing_finished)
        self.le_tolerance_stopping.textChanged.connect(self.le_tolerance_stopping_text_changed)
        self.le_tolerance_stopping.editingFinished.connect(self.le_tolerance_stopping_editing_finished)
        self.le_escape_ratio.textChanged.connect(self.le_escape_ratio_text_changed)
        self.le_escape_ratio.editingFinished.connect(self.le_escape_ratio_editing_finished)

        self.le_incident_energy.textChanged.connect(self.le_incident_energy_text_changed)
        self.le_incident_energy.editingFinished.connect(self.le_incident_energy_editing_finished)
        self.le_range_low.textChanged.connect(self.le_range_low_text_changed)
        self.le_range_low.editingFinished.connect(self.le_range_low_editing_finished)
        self.le_range_high.textChanged.connect(self.le_range_high_text_changed)
        self.le_range_high.editingFinished.connect(self.le_range_high_editing_finished)

        self.le_snip_window_size.textChanged.connect(self.le_snip_window_size_text_changed)
        self.le_snip_window_size.editingFinished.connect(self.le_snip_window_size_editing_finished)

        self.cb_linear_baseline.stateChanged.connect(self.cb_linear_baseline_state_changed)
        self.cb_snip_baseline.stateChanged.connect(self.cb_snip_baseline_state_changed)

        return hbox

    def update_widget_state(self, condition=None):
        # Update the state of the menu bar
        state = not self.gui_vars["gui_state"]["running_computations"]
        self.setEnabled(state)

        if condition == "tooltips":
            self._set_tooltips()

    def _set_tooltips(self):
        set_tooltip(self.pb_apply, "Save changes and <b>update plots</b>.")
        set_tooltip(self.pb_cancel, "<b>Discard</b> all changes.")
        set_tooltip(self.le_max_iterations, "<b>Maximum number of iterations</b> used for total spectrum fitting.")
        set_tooltip(self.le_tolerance_stopping, "<b>Tolerance</b> setting for total spectrum fitting.")
        set_tooltip(self.le_escape_ratio, "Parameter for total spectrum fitting: <b>escape ration</b>")
        set_tooltip(self.le_incident_energy, "<b>Incident energy</b> in keV")
        set_tooltip(self.le_range_low, "<b>Lower boundary</b> of the selected range in keV.")
        set_tooltip(self.le_range_high, "<b>Upper boundary</b> of the selected range in keV.")
        set_tooltip(
            self.cb_linear_baseline,
            "Subtract baseline as represented as a constant. <b>XRF Map generation</b>. "
            "Baseline subtraction is performed as part of NNLS fitting.",
        )
        set_tooltip(
            self.cb_snip_baseline,
            "Subtract baseline using SNIP method. <b>XRF Map generation</b>. "
            "This is a separate step of processing and can be used together with "
            "'linear' baseline subtraction if needed.",
        )
        set_tooltip(
            self.le_snip_window_size,
            "Window size for <b>SNIP</b> algorithm. Used both for total spectrum fitting "
            "and XRF Map generation. SNIP baseline subtraction is always performed while "
            "fitting total spectrum, but its effect may be reduced or eliminated by setting "
            "the window size to some large value.",
        )

    def pb_apply_clicked(self):
        """Save dialog data and update plots"""
        self.save_form_data()

    def pb_cancel_clicked(self):
        """Reload data (discard all changes)"""
        self.update_form_data()

    def le_max_iterations_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_max_iterations_editing_finished(self):
        if self._validate_max_interations():
            self._dialog_data["max_iterations"] = int(self._read_le_value(self.le_max_iterations))
        else:
            self._show_max_iterations()

    def le_tolerance_stopping_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_tolerance_stopping_editing_finished(self):
        if self._validate_tolerance_stopping():
            self._dialog_data["tolerance"] = self._read_le_value(self.le_tolerance_stopping)
        else:
            self._show_tolerance_stopping()

    def le_escape_ratio_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_escape_ratio_editing_finished(self):
        if self._validate_escape_ratio():
            self._dialog_data["escape_peak_ratio"] = self._read_le_value(self.le_escape_ratio)
        else:
            self._show_escape_ratio()

    def le_incident_energy_text_changed(self, text):
        self._data_changed = True
        if self._validate_incident_energy(text):
            val = float(text)
            val_range_high = val + 0.8
            self._show_range_high(val_range_high)
        self._validate_all()

    def le_incident_energy_editing_finished(self):
        if self._validate_incident_energy():
            self._dialog_data["incident_energy"] = self._read_le_value(self.le_incident_energy)
        else:
            self._show_incident_energy()
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()
        self._validate_all()

    def le_range_low_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_range_low_editing_finished(self):
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()
        self._validate_all()

    def le_range_high_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_range_high_editing_finished(self):
        if self._validate_range():
            self._dialog_data["range_low"] = self._read_le_value(self.le_range_low)
            self._dialog_data["range_high"] = self._read_le_value(self.le_range_high)
        else:
            self._show_range_low()
            self._show_range_high()

    def le_snip_window_size_text_changed(self, text):
        self._data_changed = True
        self._validate_all()

    def le_snip_window_size_editing_finished(self):
        if self._validate_snip_window_size():
            self._dialog_data["snip_window_size"] = self._read_le_value(self.le_snip_window_size)
        else:
            self._show_snip_window_size()

    def cb_linear_baseline_state_changed(self, state):
        self._dialog_data["subtract_baseline_linear"] = state == Qt.Checked
        self._data_changed = True
        self._validate_all()

    def cb_snip_baseline_state_changed(self, state):
        self._dialog_data["subtract_baseline_snip"] = state == Qt.Checked
        self._data_changed = True
        self._validate_all()

    def update_form_data(self):
        self._dialog_data = self.gpc.get_general_fitting_params()
        self._show_all()
        self._data_changed = False
        self._validate_all()

    def save_form_data(self):
        if self._data_changed:

            def cb(dialog_data):
                try:
                    self.gpc.set_general_fitting_params(dialog_data)
                    success, msg = True, ""
                except Exception as ex:
                    success, msg = False, str(ex)
                return {"success": success, "msg": msg}

            self._compute_in_background(cb, self.slot_save_form_data, dialog_data=self._dialog_data)

    @Slot(object)
    def slot_save_form_data(self, result):
        self._recover_after_compute(self.slot_save_form_data)

        if not result["success"]:
            msg = result["msg"]
            msgbox = QMessageBox(
                QMessageBox.Critical, "Failed to Apply Fit Parameters", msg, QMessageBox.Ok, parent=self
            )
            msgbox.exec()
        else:
            self._data_changed = False
            self._validate_all()

        self.gui_vars["gui_state"]["state_model_fit_exists"] = False
        self.update_global_state.emit()

    def _show_all(self):
        self._show_max_iterations()
        self._show_tolerance_stopping()
        self._show_escape_ratio()
        self._show_incident_energy()
        self._show_range_high()
        self._show_range_low()
        self._show_linear_baseline()
        self._show_snip_baseline()
        self._show_snip_window_size()

    def _show_max_iterations(self):
        val = self._dialog_data["max_iterations"]
        self.le_max_iterations.setText(f"{val}")

    def _show_tolerance_stopping(self):
        val = self._dialog_data["tolerance"]
        self.le_tolerance_stopping.setText(self._format_float(val))

    def _show_escape_ratio(self):
        val = self._dialog_data["escape_peak_ratio"]
        self.le_escape_ratio.setText(self._format_float(val))

    def _show_incident_energy(self):
        val = self._dialog_data["incident_energy"]
        self.le_incident_energy.setText(self._format_float(val))

    def _show_range_high(self, val=None):
        val = self._dialog_data["range_high"] if val is None else val
        self.le_range_high.setText(self._format_float(val))

    def _show_range_low(self):
        val = self._dialog_data["range_low"]
        self.le_range_low.setText(self._format_float(val))

    def _show_linear_baseline(self):
        val = self._dialog_data["subtract_baseline_linear"]
        self.cb_linear_baseline.setChecked(Qt.Checked if val else Qt.Unchecked)

    def _show_snip_baseline(self):
        val = self._dialog_data["subtract_baseline_snip"]
        self.cb_snip_baseline.setChecked(Qt.Checked if val else Qt.Unchecked)

    def _show_snip_window_size(self):
        val = self._dialog_data["snip_window_size"]
        self.le_snip_window_size.setText(self._format_float(val))

    def _validate_all(self):
        valid = (
            self._validate_max_interations()
            and self._validate_tolerance_stopping()
            and self._validate_escape_ratio()
            and self._validate_incident_energy()
            and self._validate_range()
            and self._validate_snip_window_size()
        )

        self.pb_apply.setEnabled(valid and self._data_changed)
        self.pb_cancel.setEnabled(valid and self._data_changed)

    def _validate_max_interations(self, text=None):
        if text is None:
            text = self.le_max_iterations.text()

        valid = self._validate_int(text, v_min=1)
        self.le_max_iterations.setValid(valid)

        return valid

    def _validate_tolerance_stopping(self, text=None):
        if text is None:
            text = self.le_tolerance_stopping.text()

        valid = self._validate_float(text, v_min=1e-30)
        self.le_tolerance_stopping.setValid(valid)

        return valid

    def _validate_escape_ratio(self, text=None):
        if text is None:
            text = self.le_escape_ratio.text()

        valid = self._validate_float(text, v_min=0)
        self.le_escape_ratio.setValid(valid)

        return valid

    def _validate_incident_energy(self, text=None):
        if text is None:
            text = self.le_incident_energy.text()

        valid = self._validate_float(text, v_min=0)
        self.le_incident_energy.setValid(valid)

        return valid

    def _validate_range(self, low_text=None, high_text=None):
        if low_text is None:
            low_text = self.le_range_low.text()
        if high_text is None:
            high_text = self.le_range_high.text()

        valid = False
        if self._validate_float(low_text, v_min=0) and self._validate_float(high_text, v_min=0):
            v_low = float(low_text)
            v_high = float(high_text)
            if v_low < v_high:
                valid = True

        self.le_range_high.setValid(valid)
        self.le_range_low.setValid(valid)

        return valid

    def _validate_snip_window_size(self, text=None):
        if text is None:
            text = self.le_snip_window_size.text()

        valid = self._validate_float(text, v_min=1e-30)
        self.le_snip_window_size.setValid(valid)

        return valid

    def _validate_int(self, text, *, v_min=None, v_max=None):
        valid = False
        if self._validator_int.validate(text, 0)[0] == IntValidatorStrict.Acceptable:
            valid = True
            v_int = int(text)
            if (v_min is not None) and (v_int < v_min):
                valid = False
            if (v_max is not None) and (v_int > v_max):
                valid = False
        return valid

    def _validate_float(self, text, *, v_min=None, v_max=None):
        valid = False
        if self._validator_float.validate(text, 0)[0] == DoubleValidatorStrict.Acceptable:
            valid = True
            v_float = float(text)
            if (v_min is not None) and (v_float < v_min):
                valid = False
            if (v_max is not None) and (v_float > v_max):
                valid = False
        return valid

    def _format_float(self, val):
        return f"{val:.10g}"

    def _read_le_value(self, line_edit):
        """It is assumed that the value is validated before the function is called"""
        return float(line_edit.text())

    def _compute_in_background(self, func, slot, *args, **kwargs):
        """
        Run function `func` in a background thread. Send the signal
        `self.computations_complete` once computation is finished.

        Parameters
        ----------
        func: function
            Reference to a function that is supposed to be executed at the background.
            The function return value is passed as a signal parameter once computation is
            complete.
        slot: qtpy.QtCore.Slot or None
            Reference to a slot. If not None, then the signal `self.computation_complete`
            is connected to this slot.
        args, kwargs
            arguments of the function `func`.
        """
        signal_complete = self.computations_complete

        def func_to_run(func, *args, **kwargs):
            class RunTask(QRunnable):
                def run(self):
                    result_dict = func(*args, **kwargs)
                    signal_complete.emit(result_dict)

            return RunTask()

        if slot is not None:
            self.computations_complete.connect(slot)
        self.gui_vars["gui_state"]["running_computations"] = True
        self.update_global_state.emit()
        QThreadPool.globalInstance().start(func_to_run(func, *args, **kwargs))

    def _recover_after_compute(self, slot):
        """
        The function should be called after the signal `self.computations_complete` is
        received. The slot should be the same as the one used when calling
        `self.compute_in_background`.
        """
        if slot is not None:
            self.computations_complete.disconnect(slot)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()
