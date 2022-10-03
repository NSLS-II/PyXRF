import os

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QFileDialog,
    QMessageBox,
)
from qtpy.QtCore import Qt, Signal, Slot, QThreadPool, QRunnable

from .useful_widgets import (
    LineEditReadOnly,
    adjust_qlistwidget_height,
    global_gui_parameters,
    PushButtonMinimumWidth,
    set_tooltip,
    clear_gui_state,
)
from .form_base_widget import FormBaseWidget
from .dlg_load_mask import DialogLoadMask
from .dlg_select_scan import DialogSelectScan
from .dlg_view_metadata import DialogViewMetadata
import logging

logger = logging.getLogger(__name__)


class LoadDataWidget(FormBaseWidget):

    update_main_window_title = Signal()
    update_global_state = Signal()
    computations_complete = Signal(object)

    update_preview_map_range = Signal(str)
    signal_new_run_loaded = Signal(bool)  # True/False - success/failed
    signal_loading_new_run = Signal()  # Emitted before new run is loaded

    signal_data_channel_changed = Signal(bool)

    def __init__(self, *, gpc, gui_vars):
        super().__init__()

        # Global processing classes
        self.gpc = gpc
        # Global GUI variables (used for control of GUI state)
        self.gui_vars = gui_vars

        self.ref_main_window = self.gui_vars["ref_main_window"]

        self.update_global_state.connect(self.ref_main_window.update_widget_state)

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
        current_dir = os.path.expanduser(self.gpc.get_current_working_directory())
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
        self.cb_file_all_channels.setChecked(self.gpc.get_load_each_channel())
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
        set_tooltip(
            self.pb_set_wd,
            "Select <b>Working Directory</b>. The Working Directory is "
            "used as <b>default</b> for loading and saving data and "
            "configuration files.",
        )
        set_tooltip(self.le_wd, "Currently selected <b>Working Directory</b>")
        set_tooltip(self.pb_file, "Load data from a <b>file on disk</b>.")
        set_tooltip(self.pb_dbase, "Load data from a <b>database</b> (Databroker).")
        set_tooltip(
            self.cb_file_all_channels,
            "Load <b>all</b> available data channels (checked) or only the <b>sum</b> of the channels",
        )
        set_tooltip(self.le_file, "The <b>name</b> of the loaded file or <b>ID</b> of the loaded run.")
        set_tooltip(self.pb_view_metadata, "View scan <b>metadata</b> (if available)")
        set_tooltip(self.cbox_channel, "Select channel for processing. Typically the <b>sum</b> channel is used.")
        set_tooltip(
            self.pb_apply_mask,
            "Load the mask from file and/or select spatial ROI. The mask and ROI "
            "are used in run <b>Preview</b> and fitting of the <b>total spectrum</b>.",
        )
        set_tooltip(
            self.list_preview,
            "Data for the selected channels is displayed in <b>Preview</b> tab. "
            "The displayed <b>total spectrum</b> is computed for the selected "
            "ROI and using the loaded mask. If no mask or ROI are enabled, then "
            "total spectrum is computed over all pixels of the image.",
        )

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
            items = list(self.gpc.get_file_channel_list())
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
            items = list(self.gpc.get_file_channel_list())
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
            state = Qt.Checked if self.gpc.is_dset_item_selected_for_preview(item) else Qt.Unchecked
            self.list_preview.item(n).setCheckState(state)

        self.list_preview.itemChanged.connect(self.list_preview_item_changed)

    def pb_set_wd_clicked(self):
        dir_current = self.le_wd.text()
        dir = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            dir_current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if dir:
            self.gpc.set_current_working_directory(dir)
            self.le_wd.setText(dir)

    def pb_file_clicked(self):
        dir_current = self.gpc.get_current_working_directory()
        file_paths = QFileDialog.getOpenFileName(self, "Open Data File", dir_current, "HDF5 (*.h5);; All (*)")
        file_path = file_paths[0]
        if file_path:
            self.signal_loading_new_run.emit()

            def cb(file_path):
                result_dict = {}
                try:
                    msg = self.gpc.open_data_file(file_path)
                    status = True
                except Exception as ex:
                    msg = str(ex)
                    status = False
                result_dict.update({"status": status, "msg": msg, "file_path": file_path})
                return result_dict

            self._compute_in_background(cb, self.slot_file_clicked, file_path=file_path)

    @Slot(object)
    def slot_file_clicked(self, result):
        self._recover_after_compute(self.slot_file_clicked)

        status = result["status"]
        msg = result["msg"]  # Message is empty if file loading failed
        file_path = result["file_path"]
        if status:
            file_text = f"'{self.gpc.get_loaded_file_name()}'"
            if self.gpc.is_scan_metadata_available():
                file_text += f": ID#{self.gpc.get_metadata_scan_id()}"
            self.le_file.setText(file_text)

            self.gui_vars["gui_state"]["state_file_loaded"] = True
            # Invalidate fit. Fit must be rerun for new data.
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            # Check if any datasets were loaded.
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = self.gpc.is_xrf_maps_available()

            # Disable the button for changing working directory. This is consistent
            #   with the behavior of the old PyXRF, but will be changed in the future.
            # self.pb_set_wd.setEnabled(False)

            # Enable/disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(self.gpc.is_scan_metadata_available())
            self.le_wd.setText(self.gpc.get_current_working_directory())

            self.update_main_window_title.emit()
            self.update_global_state.emit()

            self._set_cbox_channel_items()
            self._set_list_preview_items()

            # Here we want to reset the range in the Total Count Map preview
            self.update_preview_map_range.emit("reset")

            self.signal_new_run_loaded.emit(True)  # Data is loaded successfully

            if msg:
                # Display warning message if it was generated
                msgbox = QMessageBox(QMessageBox.Warning, "Warning", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

        else:
            self.le_file.setText(self.le_file_default)

            # Disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(False)
            self.le_wd.setText(self.gpc.get_current_working_directory())

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

            msg_str = (
                f"Incorrect format of input file '{file_path}': "
                f"PyXRF accepts only custom HDF (.h5) files."
                f"\n\nError message: {msg}"
            )
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg_str, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def pb_dbase_clicked(self):

        dlg = DialogSelectScan()
        if dlg.exec() == QDialog.Accepted:
            mode, id_uid = dlg.get_id_uid()

            self.signal_loading_new_run.emit()

            def cb(id_uid):
                result_dict = {}
                try:
                    msg, file_name = self.gpc.load_run_from_db(id_uid)
                    status = True
                except Exception as ex:
                    msg, file_name = str(ex), ""
                    status = False
                result_dict.update({"status": status, "msg": msg, "id_uid": id_uid, "file_name": file_name})
                return result_dict

            self._compute_in_background(cb, self.slot_dbase_clicked, id_uid=id_uid)

    @Slot(object)
    def slot_dbase_clicked(self, result):
        self._recover_after_compute(self.slot_dbase_clicked)

        status = result["status"]
        msg = result["msg"]  # Message is empty if file loading failed
        id_uid = result["id_uid"]
        # file_name = result["file_name"]
        if status:
            file_text = f"'{self.gpc.get_loaded_file_name()}'"
            if self.gpc.is_scan_metadata_available():
                file_text += f": ID#{self.gpc.get_metadata_scan_id()}"
            self.le_file.setText(file_text)

            self.gui_vars["gui_state"]["state_file_loaded"] = True
            # Invalidate fit. Fit must be rerun for new data.
            self.gui_vars["gui_state"]["state_model_fit_exists"] = False
            # Check if any datasets were loaded.
            self.gui_vars["gui_state"]["state_xrf_map_exists"] = self.gpc.is_xrf_maps_available()

            # Disable the button for changing working directory. This is consistent
            #   with the behavior of the old PyXRF, but will be changed in the future.
            # self.pb_set_wd.setEnabled(False)

            # Enable/disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(self.gpc.is_scan_metadata_available())
            self.le_wd.setText(self.gpc.get_current_working_directory())

            self.update_main_window_title.emit()
            self.update_global_state.emit()

            self._set_cbox_channel_items()
            self._set_list_preview_items()

            # Here we want to reset the range in the Total Count Map preview
            self.update_preview_map_range.emit("reset")

            self.signal_new_run_loaded.emit(True)  # Data is loaded successfully

            if msg:
                # Display warning message if it was generated
                msgbox = QMessageBox(QMessageBox.Warning, "Warning", msg, QMessageBox.Ok, parent=self)
                msgbox.exec()

        else:
            self.le_file.setText(self.le_file_default)

            # Disable 'View Metadata' button
            self.pb_view_metadata.setEnabled(False)
            self.le_wd.setText(self.gpc.get_current_working_directory())

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

            msg_str = f"Failed to load scan '{id_uid}'.\n\nError message: {msg}"
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg_str, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def pb_apply_mask_clicked(self):
        map_size = self.gpc.get_dataset_map_size()
        map_size = map_size if (map_size is not None) else (0, 0)

        dlg = DialogLoadMask()
        dlg.set_image_size(n_rows=map_size[0], n_columns=map_size[1])
        roi = self.gpc.get_preview_spatial_roi()
        dlg.set_roi(
            row_start=roi["row_start"],
            column_start=roi["col_start"],
            row_end=roi["row_end"],
            column_end=roi["col_end"],
        )
        dlg.set_roi_active(self.gpc.is_roi_selection_active())

        dlg.set_default_directory(self.gpc.get_current_working_directory())
        dlg.set_mask_file_path(self.gpc.get_mask_selection_file_path())
        dlg.set_mask_file_active(self.gpc.is_mask_selection_active())

        if dlg.exec() == QDialog.Accepted:
            roi_keys = ("row_start", "col_start", "row_end", "col_end")
            roi_list = dlg.get_roi()
            roi_selected = {k: roi_list[n] for n, k in enumerate(roi_keys)}
            self.gpc.set_preview_spatial_roi(roi_selected)

            self.gpc.set_roi_selection_active(dlg.get_roi_active())

            self.gpc.set_mask_selection_file_path(dlg.get_mask_file_path())
            self.gpc.set_mask_selection_active(dlg.get_mask_file_active())

            def cb():
                try:
                    # TODO: proper error processing is needed here (exception RuntimeError)
                    self.gpc.apply_mask_to_datasets()
                    success = True
                    msg = ""
                except Exception as ex:
                    success = False
                    msg = str(ex)
                return {"success": success, "msg": msg}

            self._compute_in_background(cb, self.slot_apply_mask_clicked)

    @Slot(object)
    def slot_apply_mask_clicked(self, result):
        if not result["success"]:
            msg = f"Error occurred while applying the ROI selection:\nException: {result['msg']}"
            logger.error(f"{msg}")
            mb_error = QMessageBox(QMessageBox.Critical, "Error", f"{msg}", QMessageBox.Ok, parent=self)
            mb_error.exec()

        # Here we want to expand the range in the Total Count Map preview if needed
        self.update_preview_map_range.emit("update")
        self._recover_after_compute(self.slot_apply_mask_clicked)

    def pb_view_metadata_clicked(self):

        dlg = DialogViewMetadata()
        metadata_string = self.gpc.get_formatted_metadata()
        dlg.setText(metadata_string)
        dlg.exec()

    def cb_file_all_channels_toggled(self, state):
        self.gpc.set_load_each_channel(state)

    def cbox_channel_index_changed(self, index):
        def cb(index):
            try:
                self.gpc.set_data_channel(index)
                success, msg = True, ""
            except Exception as ex:
                success = False
                msg = str(ex)
            return {"success": success, "msg": msg}

        self._compute_in_background(cb, self.slot_channel_index_changed, index=index)

    @Slot(object)
    def slot_channel_index_changed(self, result):
        self._recover_after_compute(self.slot_channel_index_changed)

        if result["success"]:
            self.signal_data_channel_changed.emit(True)
        else:
            self.signal_data_channel_changed.emit(False)
            msg = result["msg"]
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

    def list_preview_item_changed(self, list_item):
        # Find the index of the list item that was checked/unchecked
        ind = -1
        for n in range(self.list_preview.count()):
            if self.list_preview.item(n) == list_item:
                ind = n

        # Get the state of the checkbox (checked -> 2, unchecked -> 0)
        state = list_item.checkState()

        # The name of the dataset
        dset_name = self.gpc.get_file_channel_list()[ind]

        def cb():
            try:
                self.gpc.select_preview_dataset(dset_name=dset_name, is_visible=bool(state))
                success, msg = True, ""
            except Exception as ex:
                success = False
                msg = str(ex)
            return {"success": success, "msg": msg}

        self._compute_in_background(cb, self.slot_preview_items_changed)

    @Slot(object)
    def slot_preview_items_changed(self, result):
        if not result["success"]:
            # The error shouldn't actually happen here. This is to prevent potential crashes.
            msg = f"Error occurred: {result['msg']}.\nData may need to be reloaded to continue processing."
            msgbox = QMessageBox(QMessageBox.Critical, "Error", msg, QMessageBox.Ok, parent=self)
            msgbox.exec()

        # Here we want to expand the range in the Total Count Map preview if needed
        self.update_preview_map_range.emit("expand")
        self._recover_after_compute(self.slot_preview_items_changed)

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
            class LoadFile(QRunnable):
                def run(self):
                    result_dict = func(*args, **kwargs)
                    signal_complete.emit(result_dict)

            return LoadFile()

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
