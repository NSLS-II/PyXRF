import os

from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog,
                             QRadioButton, QButtonGroup, QGridLayout,
                             QTextEdit)
from PyQt5.QtCore import Qt

from .useful_widgets import (LineEditReadOnly, adjust_qlistwidget_height,
                             global_gui_parameters, PushButtonMinimumWidth,
                             set_tooltip)
from .form_base_widget import FormBaseWidget


class LoadDataWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
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
        current_dir = os.path.expanduser("~")
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
        self.pb_dbase.clicked.connect(self.pb_dbase_clicked)

        self.cb_file_all_channels = QCheckBox("All channels")

        self.le_file = LineEditReadOnly("No data is loaded")

        self.pb_view_metadata = QPushButton("View Metadata ...")
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
        self.cbox_channel.addItems(["channel_name_sum", "channel_name_det1",
                                    "channel_name_det2", "channel_name_det3"])

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
        self.list_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_preview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Set list items
        sample_items = ["channel_name_sum", "channel_name_det1",
                        "channel_name_det2", "channel_name_det3"]
        for s in sample_items:
            wi = QListWidgetItem(s, self.list_preview)
            wi.setFlags(wi.flags() | Qt.ItemIsUserCheckable)
            wi.setFlags(wi.flags() & ~Qt.ItemIsSelectable)
            wi.setCheckState(Qt.Unchecked)
        # Adjust height so that it fits all the elements
        adjust_qlistwidget_height(self.list_preview)

        hbox = QHBoxLayout()
        hbox.addWidget(self.list_preview)

        self.group_preview.setLayout(hbox)

    def _set_tooltips(self):
        set_tooltip(self.pb_set_wd,
                    "Select Working Directory. The Working Directory is "
                    "used as default for loading and save data and "
                    "configuration files.")
        set_tooltip(self.le_wd, "Currently selected Working Directory")
        set_tooltip(self.pb_file, "Load data from a file on disk.")
        set_tooltip(self.pb_dbase, "Load data from a database (Databroker).")
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

    def pb_set_wd_clicked(self):
        dir_current = self.le_wd.text()
        dir = QFileDialog.getExistingDirectory(
            self, "Select Working Directory", dir_current,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if dir:
            self.le_wd.setText(dir)

    def pb_file_clicked(self):
        dir_current = self.le_wd.text()
        file_name = QFileDialog.getOpenFileName(self, "Open Data File",
                                                dir_current,
                                                "HDF5 (*.h5);; All (*)")
        file_name = file_name[0]
        if file_name:
            print(f"Opening file: {file_name}")

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
        dlg.exec()


class DialogSelectScan(QDialog):

    def __init__(self):

        super().__init__()

        self.resize(400, 200)
        self.setWindowTitle("Load Run From Database")

        label = QLabel("Enter run ID or UID:")
        self.le_id_uid = QLineEdit()
        self.rb_id = QRadioButton("Run ID")
        self.rb_id.setChecked(True)
        self.rb_uid = QRadioButton("Run UID")

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
        txt = "This field will contain metadata for the loaded run.\n"\
              "QTextEdit may be later replaced by the QTable widget."
        self.te_meta.setText(txt)

        # 'Close' button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.te_meta)
        vbox.addWidget(button_box)
        self.setLayout(vbox)