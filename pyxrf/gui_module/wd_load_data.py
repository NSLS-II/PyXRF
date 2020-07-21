import os
from threading import Thread
import numpy as np

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QMessageBox)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

from .useful_widgets import (LineEditReadOnly, LineEditExtended, adjust_qlistwidget_height,
                             global_gui_parameters, PushButtonMinimumWidth,
                             set_tooltip, clear_gui_state)
from .form_base_widget import FormBaseWidget

import logging
logger = logging.getLogger(__name__)


class LoadDataWidget(FormBaseWidget):

    update_main_window_title = pyqtSignal()
    update_global_state = pyqtSignal()
    computations_complete = pyqtSignal()

    update_preview_map_range = pyqtSignal(str)
    signal_new_run_loaded = pyqtSignal(bool)  # True/False - success/failed

    def __init__(self,  *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.update_global_state.connect(self.ref_main_window.update_widget_state)

        # Reference to background thread used to run computations. The reference is
        #   meaningful only when the computations are run.
        self.bckg_thread = None

        self.initialize()

    def initialize(self):

        v_spacing = global_gui_parameters["vertical_spacing_in_tabs"]

        vbox = QVBoxLayout()

        self._setup_wd_group()
        vbox.addWidget(self.group_wd)
        vbox.addSpacing(v_spacing)

        self._setup_load_group()
        vbox.addWidget(self.group_file)
        vbox.addSpacing(v_spacing)

        self._setup_sel_channel_group()
        vbox.addWidget(self.group_sel_channel)
        vbox.addSpacing(v_spacing)

        self._setup_spec_settings_group()
        vbox.addWidget(self.group_spec_settings)
        vbox.addSpacing(v_spacing)

        self._setup_preview_group()
        vbox.addWidget(self.group_preview)

        vbox.addStretch(1)

        self.setLayout(vbox)

        self._set_tooltips()

    def _setup_wd_group(self):
        self.group_wd = QGroupBox("Working Directory")

        self.pb_set_wd = PushButtonMinimumWidth("..")
        self.pb_set_wd.clicked.connect(self.pb_set_wd_clicked)

        self.le_wd = LineEditReadOnly()

        # Initial working directory. Set to the HOME directory for now
        current_dir = os.path.expanduser(self.gpc.io_model.working_directory)
        self.le_wd.setText(current_dir)

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_set_wd)
        hbox.addWidget(self.le_wd)

        self.group_wd.setLayout(hbox)

    def _setup_load_group(self):

        self.group_file = QGroupBox("Load Data")

        self.pb_file = QPushButton("Read File ...")
        self.pb_file.clicked.connect(self.pb_file_clicked)

        self.pb_dbase = QPushButton("Load Run ...")
        self.pb_dbase.setEnabled(self.gui_vars["gui_state"]["databroker_available"])
        self.pb_dbase.clicked.connect(self.pb_dbase_clicked)

        self.cb_file_all_channels = QCheckBox("All channels")
        self.cb_file_all_channels.setChecked(self.gpc.io_model.load_each_channel)
        self.cb_file_all_channels.toggled.connect(self.cb_file_all_channels_toggled)

        self.le_file_default = "No data is loaded"
        self.le_file = LineEditReadOnly(self.le_file_default)

        self.pb_view_metadata = QPushButton("View Metadata ...")
        self.pb_view_metadata.setEnabled(False)
        self.pb_view_metadata.clicked.connect(self.pb_view_metadata_clicked)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_file)
        hbox.addWidget(self.pb_dbase)
        hbox.addWidget(self.cb_file_all_channels)
        vbox.addLayout(hbox)

        vbox.addWidget(self.le_file)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.pb_view_metadata)
        vbox.addLayout(hbox)

        self.group_file.setLayout(vbox)

    def _setup_sel_channel_group(self):

        self.group_sel_channel = QGroupBox("Select Channel For Processing")

        self.cbox_channel = QComboBox()
        self.cbox_channel.currentIndexChanged.connect(self.cbox_channel_index_changed)
        self._set_cbox_channel_items(items=[])

        hbox = QHBoxLayout()
        hbox.addWidget(self.cbox_channel)

        self.group_sel_channel.setLayout(hbox)

    def _setup_spec_settings_group(self):

        self.group_spec_settings = QGroupBox("Total Spectrum Settings")

        self.pb_apply_mask = QPushButton("Apply Mask ...")
        self.pb_apply_mask.clicked.connect(self.pb_apply_mask_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_apply_mask)
        hbox.addStretch(1)

        self.group_spec_settings.setLayout(hbox)

    def _setup_preview_group(self):

        self.group_preview = QGroupBox("Preview")

        self.list_preview = QListWidget()
        self.list_preview.itemChanged.connect(self.list_preview_item_changed)
        self.list_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_preview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._set_list_preview_items(items=[])

        hbox = QHBoxLayout()
        hbox.addWidget(self.list_preview)

        self.group_preview.setLayout(hbox)

    def _set_tooltips(self):
        set_tooltip(self.pb_set_wd,
                    "Select <b>Working Directory</b>. The Working Directory is "
                    "used as <b>default</b> for loading and saving data and "
                    "configuration files.")
        set_tooltip(self.le_wd, "Currently selected <b>Working Directory</b>")
        set_tooltip(self.pb_file, "Load data from a <b>file on disk</b>.")
        set_tooltip(self.pb_dbase, "Load data from a <b>database</b> (Databroker).")
        set_tooltip(self.cb_file_all_channels,
                    "Load <b>all</b> available data channels (checked) or "
                    "only the <b>sum</b> of the channels")
        set_tooltip(self.le_file,
                    "The <b>name</b> of the loaded file or <b>ID</b> of the loaded run.")
        set_tooltip(self.pb_view_metadata, "View scan <b>metadata</b> (if available)")
        set_tooltip(
            self.cbox_channel,
            "Select channel for processing. Typically the <b>sum</b> channel is used.")
        set_tooltip(
            self.pb_apply_mask,
            "Load the mask from file and/or select spatial ROI. The mask and ROI "
            "are used in run <b>Preview</b> and fitting of the <b>total spectrum</b>.")
        set_tooltip(
            self.list_preview,
            "Data for the selected channels is displayed in <b>Preview</b> tab. "
            "The displayed <b>total spectrum</b> is computed for the selected "
            "ROI and using the loaded mask. If no mask or ROI are enabled, then "
            "total spectrum is computed over all pixels of the image.")

    def update_widget_state(self, condition=None):
        if condition == "tooltips":
            self._set_tooltips()

        state = self.gui_vars["gui_state"]["state_file_loaded"]
        self.group_sel_channel.setEnabled(state)
        self.group_spec_settings.setEnabled(state)
        self.group_preview.setEnabled(state)

    def _set_cbox_channel_items(self, *, items=None):
        """
        Set items of the combo box for item selection. If the list of items is not specified,
        then it is loaded from the respective data structure.

        Parameters
        ----------
        items: list(str)
            The list of items. The list may be cleared by calling
            `self._set_cbox_channel_items(items=[])`

        """
        self.cbox_channel.currentIndexChanged.disconnect(self.cbox_channel_index_changed)

        self.cbox_channel.clear()
        if items is None:
            items = list(self.gpc.io_model.file_channel_list)
        self.cbox_channel.addItems(items)

        self.cbox_channel.currentIndexChanged.connect(self.cbox_channel_index_changed)

        if len(items):
            # Select the first item (if there is at least one item)
            self.cbox_channel.setCurrentIndex(0)

    def _set_list_preview_items(self, *, items=None):
        """
        Set items of the list for selecting channels in preview tab. If the list of items is
        not specified, then it is loaded from the respective data structure.

        Parameters
        ----------
        items: list(str)
            The list of items. The list may be cleared by calling
            `self._set_cbox_channel_items(items=[])`

        """
        self.list_preview.itemChanged.disconnect(self.list_preview_item_changed)

        self.list_preview.clear()
        if items is None:
            items = list(self.gpc.io_model.file_channel_list)
        for s in items:
            wi = QListWidgetItem(s, self.list_preview)
            wi.setFlags(wi.flags() | Qt.ItemIsUserCheckable)
            wi.setFlags(wi.flags() & ~Qt.ItemIsSelectable)
            wi.setCheckState(Qt.Unchecked)

        # Adjust height so that it fits all the elements
        adjust_qlistwidget_height(self.list_preview, other_widgets=[self.group_preview, self])

        # This will cause the preview data to be plotted (the plot is expected to be hidden,
        #   since no channels were selected). Here we select the first channel in the list.
        for n, item in enumerate(items):
            state = Qt.Checked if self.gpc.io_model.data_sets[item].selected_for_preview \
                else Qt.Unchecked
            self.list_preview.item(n).setCheckState(state)

        self.list_preview.itemChanged.connect(self.list_preview_item_changed)

    def pb_set_wd_clicked(self):
        dir_current = self.le_wd.text()
        dir = QFileDialog.getExistingDirectory(
            self, "Select Working Directory", dir_current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            self.gpc.io_model.working_directory = dir
            self.le_wd.setText(dir)

    def pb_file_clicked(self):
        dir_current = self.gpc.io_model.working_directory
        file_paths = QFileDialog.getOpenFileName(self, "Open Data File",
                                                 dir_current,
                                                 "HDF5 (*.h5);; All (*)")
        file_path = file_paths[0]
        if file_path:
            self._result = {}

            def cb(file_path, result_dict):
                try:
                    msg = self.gpc.open_data_file(file_path)
                    status = True
                except Exception as ex:
                    msg = str(ex)
                    status = False
                result_dict.update({
                    "status": status,
                    "msg": msg,
                    "file_path": file_path
                })
                self.computations_complete.emit()

            self.computations_complete.connect(self.slot_file_clicked)
            self.gui_vars["gui_state"]["running_computations"] = True
            self.update_global_state.emit()
            self.bckg_thread = Thread(target=cb, kwargs={"file_path": file_path,
                                                         "result_dict": self._result})
            self.bckg_thread.start()

    @pyqtSlot()
    def slot_file_clicked(self):
        self.computations_complete.disconnect(self.slot_file_clicked)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()

        status = self._result["status"]
        msg = self._result["msg"]  # Message is empty if file loading failed
        file_path = self._result["file_path"]
        if status:
            file_text = f"'{self.gpc.io_model.file_name}'"
            if self.gpc.io_model.scan_metadata_available:
                file_text += f": ID#{self.gpc.io_model.scan_metadata['scan_id']}"
            self.le_file.setText(file_text)

            self.gui_vars["gui_state"]["state_file_loaded"] = True
            # Invalidate fit. Fit must be rerun for new data.
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            # Check if any datasets were loaded.
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = self.gpc.io_model.is_xrf_maps_available()

            # Disable the button for changing working directory. This is consistent
            #   with the behavior of the old PyXRF, but will be changed in the future.
            self.pb_set_wd.setEnabled(False)

            # Enable/disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(self.gpc.io_model.scan_metadata_available)
            self.le_wd.setText(self.gpc.io_model.working_directory)

            self.update_main_window_title.emit()
            self.update_global_state.emit()

            self._set_cbox_channel_items()
            self._set_list_preview_items()

            # Here we want to reset the range in the Total Count Map preview
            self.update_preview_map_range.emit("reset")

            self.signal_new_run_loaded.emit(True)  # Data is loaded successfully

            if msg:
                # Display warning message if it was generated
                msgbox = QMessageBox(QMessageBox.Warning, "Warning",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

        else:
            self.le_file.setText(self.le_file_default)

            # Disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(False)
            self.le_wd.setText(self.gpc.io_model.working_directory)

            # Clear flags: the state now is "No data is loaded".
            clear_gui_state(self.gui_vars)
            self.update_global_state.emit()

            self.update_main_window_title.emit()
            self.update_global_state.emit()

            self._set_cbox_channel_items(items=[])
            self._set_list_preview_items(items=[])

            # Here we want to clear the range in the Total Count Map preview
            self.update_preview_map_range.emit("clear")

            self.signal_new_run_loaded.emit(False)  # Failed to load data

            msg_str = f"Incorrect format of input file '{file_path}': " \
                      f"PyXRF accepts only custom HDF (.h5) files." \
                      f"\n\nError message: {msg}"
            msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                 msg_str, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def pb_dbase_clicked(self):

        dlg = DialogSelectScan()
        if dlg.exec() == QDialog.Accepted:
            print("Dialog exit: Ok button")

    def pb_apply_mask_clicked(self):
        map_size = self.gpc.io_model.get_dataset_map_size()
        map_size = map_size if (map_size is not None) else (0, 0)

        dlg = DialogLoadMask()
        dlg.set_image_size(n_rows=map_size[0], n_columns=map_size[1])
        dlg.set_roi(row_start=self.gpc.io_model.roi_row_start,
                    column_start=self.gpc.io_model.roi_col_start,
                    row_end=self.gpc.io_model.roi_row_end,
                    column_end=self.gpc.io_model.roi_col_end)
        dlg.set_roi_active(self.gpc.io_model.roi_selection_active)

        dlg.set_default_directory(self.gpc.io_model.working_directory)
        dlg.set_mask_file_path(self.gpc.io_model.mask_file_path)
        dlg.set_mask_file_active(self.gpc.io_model.mask_active)

        if dlg.exec() == QDialog.Accepted:
            self.gpc.io_model.roi_row_start, self.gpc.io_model.roi_col_start, \
                self.gpc.io_model.roi_row_end, self.gpc.io_model.roi_col_end = dlg.get_roi()
            self.gpc.io_model.roi_selection_active = dlg.get_roi_active()
            self.gpc.io_model.mask_file_path = dlg.get_mask_file_path()
            # At this point the mask name is just the file name
            self.gpc.io_model.mask_name = os.path.split(self.gpc.io_model.mask_file_path)[-1]
            self.gpc.io_model.mask_active = dlg.get_mask_file_active()

            def _cb():
                try:
                    # TODO: proper error processing is needed here (exception RuntimeError)
                    self.gpc.io_model.apply_mask_to_datasets()
                    self.gpc.plot_model.data_sets = self.gpc.io_model.data_sets
                    self.gpc.plot_model.update_preview_spectrum_plot()
                except Exception as ex:
                    logger.error(f"Error occurred while applying the mask: {str(ex)}")
                self.computations_complete.emit()

            self.computations_complete.connect(self.slot_apply_mask_clicked)
            self.gui_vars["gui_state"]["running_computations"] = True
            self.update_global_state.emit()
            self.bckg_thread = Thread(target=_cb)
            self.bckg_thread.start()

    @pyqtSlot()
    def slot_apply_mask_clicked(self):
        # Here we want to expand the range in the Total Count Map preview if needed
        self.update_preview_map_range.emit("update")

        self.computations_complete.disconnect(self.slot_apply_mask_clicked)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()

    def pb_view_metadata_clicked(self):

        dlg = DialogViewMetadata()
        metadata_string = self.gpc.io_model.scan_metadata.get_formatted_output()
        dlg.setText(metadata_string)
        dlg.exec()

    def cb_file_all_channels_toggled(self, state):
        self.gpc.io_model.load_each_channel = state

    def cbox_channel_index_changed(self, index):
        self.gpc.io_model.file_opt = index
        # Redraw the plot
        self.gpc.plot_model.plot_exp_opt = True

    def list_preview_item_changed(self, list_item):

        # Find the index of the list item that was checked/unchecked
        ind = -1
        for n in range(self.list_preview.count()):
            if self.list_preview.item(n) == list_item:
                ind = n

        # Get the state of the checkbox (checked -> 2, unchecked -> 0)
        state = list_item.checkState()

        # The name of the dataset
        dset_name = self.gpc.io_model.file_channel_list[ind]

        def cb():
            self.gpc.select_preview_dataset(dset_name=dset_name, is_visible=bool(state))
            self.computations_complete.emit()

        self.computations_complete.connect(self.slot_preview_items_changed)
        self.gui_vars["gui_state"]["running_computations"] = True
        self.update_global_state.emit()
        self.bckg_thread = Thread(target=cb)
        self.bckg_thread.start()

    @pyqtSlot()
    def slot_preview_items_changed(self):
        # Here we want to expand the range in the Total Count Map preview if needed
        self.update_preview_map_range.emit("expand")

        self.computations_complete.disconnect(self.slot_preview_items_changed)
        self.gui_vars["gui_state"]["running_computations"] = False
        self.update_global_state.emit()


class DialogSelectScan(QDialog):

    def __init__(self):

        super().__init__()

        self.resize(400, 200)
        self.setWindowTitle("Load Run From Database")

        label = QLabel("Enter run ID or UID:")
        self.le_id_uid = QLineEdit()
        set_tooltip(self.le_id_uid, "Enter <b>Run ID</b> or <b>Run UID</b>.")

        self.rb_id = QRadioButton("Run ID")
        set_tooltip(self.rb_id, "The value in the line edit box is <b>Run ID</b>")
        self.rb_id.setChecked(True)
        self.rb_uid = QRadioButton("Run UID")
        set_tooltip(self.rb_uid, "The value in the line edit box is <b>Run UID</b>")

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.rb_id)
        self.btn_group.addButton(self.rb_uid)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()

        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.le_id_uid)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.rb_id)
        hbox.addStretch(1)
        hbox.addWidget(self.rb_uid)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        vbox.addStretch(1)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

        # This is how the button from QDialogButtonBox can be disabled
        # button_box.button(QDialogButtonBox.Ok).setEnabled(False)


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
        set_tooltip(self.gb_roi,
                    "Select rectangular <b>spatial ROI</b>. If <b>mask</b> is "
                    "loaded, then ROI is applied to the masked data.")
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
        set_tooltip(self.gb_mask,
                    "Load <b>mask</b> from file. Active pixels in the mask are "
                    "represented by positive integers. If <b>spatial ROI</b> is "
                    "selected, then it is applied to the masked data.")
        self.gb_mask.setCheckable(True)
        self.gb_mask.toggled.connect(self.gb_mask_toggled)
        self.gb_mask.setChecked(self._mask_active)
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_mask)
        hbox.addWidget(self.le_load_mask)
        self.gb_mask.setLayout(hbox)

        # Yes/No button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
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
        file_name = QFileDialog.getOpenFileName(self, "Load Mask From File",
                                                dir_name,
                                                "All (*)")
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
            state = (val > 0)
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
            raise ValueError("DialogLoadMask: image size have zero rows or zero columns: "
                             f"n_rows={n_rows} n_columns={n_columns}. "
                             "Report the error to the development team.")

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
        self.label_map_size.setText(
            f"{self._text_map_size_base}{self._n_rows} rows, {self._n_columns} columns.")

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


class DialogViewMetadata(QDialog):

    def __init__(self):

        super().__init__()

        self.resize(500, 500)
        self.setWindowTitle("Run Metadata")

        self.te_meta = QTextEdit()
        self.te_meta.setReadOnly(True)

        # 'Close' button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.te_meta)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def setText(self, text):
        self.te_meta.setText(text)
