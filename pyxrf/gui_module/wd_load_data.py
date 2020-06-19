import os

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal

from .useful_widgets import (LineEditReadOnly, adjust_qlistwidget_height,
                             global_gui_parameters, PushButtonMinimumWidth,
                             set_tooltip, clear_gui_state)
from .form_base_widget import FormBaseWidget


class LoadDataWidget(FormBaseWidget):

    update_main_window_title = pyqtSignal()
    update_global_state = pyqtSignal()

    def __init__(self,  *, gpc, gui_vars):
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
        self._set_cbox_channel_items(items=[])
        self.cbox_channel.currentIndexChanged.connect(self.cbox_channel_index_changed)

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

        # Set list items
        #sample_items = ["channel_name_sum", "channel_name_det1",
        #                "channel_name_det2", "channel_name_det3"]
        #for s in sample_items:
        #    wi = QListWidgetItem(s, self.list_preview)
        #    wi.setFlags(wi.flags() | Qt.ItemIsUserCheckable)
        #    wi.setFlags(wi.flags() & ~Qt.ItemIsSelectable)
        #    wi.setCheckState(Qt.Unchecked)
        # Adjust height so that it fits all the elements
        #adjust_qlistwidget_height(self.list_preview)

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
        self.cbox_channel.clear()
        if items is None:
            items = list(self.gpc.io_model.file_channel_list)
        self.cbox_channel.addItems(items)
        if len(items):
            # Select the first item (if there is at least one item)
            self.cbox_channel.setCurrentIndex(0)
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
        adjust_qlistwidget_height(self.list_preview)
        self.group_preview.adjustSize()
        self.group_preview.updateGeometry()  # Necessary to keep correct width of the groupbox
        self.adjustSize()
        self.updateGeometry()

        self.list_preview.itemChanged.connect(self.list_preview_item_changed)

        if len(items):
            self.list_preview.item(0).setCheckState(Qt.Checked)

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
            try:

                msg = self.gpc.open_data_file(file_path)

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
                if msg:
                    # Display warning message if it was generated
                    msgbox = QMessageBox(QMessageBox.Warning, "Warning",
                                         msg, QMessageBox.Ok, parent=self)
                    msgbox.exec()

            except Exception as ex:
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

                msg = f"Incorrect format of input file '{file_path}': "\
                      f"PyXRF accepts only custom HDF (.h5) files."\
                      f"\n\nError message: {ex}"
                msgbox = QMessageBox(QMessageBox.Critical, "Error",
                                     msg, QMessageBox.Ok, parent=self)
                msgbox.exec()


    def pb_dbase_clicked(self):

        dlg = DialogSelectScan()
        if dlg.exec() == QDialog.Accepted:
            print("Dialog exit: Ok button")

    def pb_apply_mask_clicked(self):

        dlg = DialogLoadMask()
        if dlg.exec() == QDialog.Accepted:
            print("Dialog exit: Ok button")

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
        ind = -1
        for n in range(self.list_preview.count()):
            if self.list_preview.item(n) == list_item:
                ind = n
        #ind = self.list_preview.itemWidget(list_item)
        sel = list_item.checkState()
        print(f"Item {ind}: state {sel}")

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

    def __init__(self):

        super().__init__()

        self.resize(500, 300)
        self.setWindowTitle("Load Mask or Select ROI")

        # Fields for entering spatial ROI coordinates
        self.le_roi_start_row = QLineEdit()
        self.le_roi_start_col = QLineEdit()
        self.le_roi_end_row = QLineEdit()
        self.le_roi_end_col = QLineEdit()

        # Group box for spatial ROI selection
        self.gb_roi = QGroupBox("Select ROI (in pixels)")
        set_tooltip(self.gb_roi,
                    "Select rectangular <b>spatial ROI</b>. If <b>mask</b> is "
                    "loaded, then ROI is applied to the masked data.")
        self.gb_roi.setCheckable(True)
        self.gb_roi.setChecked(False)  # Should be set based on data
        vbox = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("Start position:"), 0, 0)
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
        vbox.addWidget(QLabel("* The point is not included in the selection"))
        self.gb_roi.setLayout(vbox)

        # Widgets for loading mask
        self.pb_load_mask = QPushButton("Load Mask ...")
        self.pb_load_mask.clicked.connect(self.pb_load_mask_clicked)
        self.le_load_mask = LineEditReadOnly("No mask is loaded")

        # Group box for setting mask
        self.gb_mask = QGroupBox("Set mask")
        set_tooltip(self.gb_mask,
                    "Load <b>mask</b> from file. Active pixels in the mask are "
                    "represented by positive integers. If <b>spatial ROI</b> is "
                    "selected, then it is applied to the masked data.")
        self.gb_mask.setCheckable(True)
        self.gb_mask.setChecked(False)  # Should be set based on data
        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_load_mask)
        hbox.addWidget(self.le_load_mask)
        self.gb_mask.setLayout(hbox)

        # Yes/No button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.gb_roi)
        vbox.addStretch(1)
        vbox.addWidget(self.gb_mask)
        vbox.addStretch(2)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def pb_load_mask_clicked(self):
        # TODO: Propagate current directory here and use it in the dialog call
        file_name = QFileDialog.getOpenFileName(self, "Load Mask From File",
                                                ".",
                                                "All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Loading mask from file: {file_name}")


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
