from PyQt5.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLineEdit, QCheckBox, QLabel,
                             QComboBox, QListWidget, QListWidgetItem,
                             QDialog, QDialogButtonBox, QFileDialog)
from PyQt5.QtCore import Qt

from .form_base_widget import FormBaseWidget


class LoadDataWidget(FormBaseWidget):

    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):

        vbox = QVBoxLayout()

        self._setup_wd_group()
        vbox.addWidget(self.group_wd)

        self._setup_file_group()
        vbox.addWidget(self.group_file)

        self._setup_dbase_group()
        vbox.addWidget(self.group_dbase)

        self._setup_sel_channel_group()
        vbox.addWidget(self.group_sel_channel)

        self._setup_spec_settings_group()
        vbox.addWidget(self.group_spec_settings)

        self._setup_preview_group()
        vbox.addWidget(self.group_preview)

        vbox.addStretch(1)

        self.setLayout(vbox)

    def _setup_wd_group(self):
        self.group_wd = QGroupBox("Working Directory")

        self.pb_set_wd = QPushButton("..")
        pb_size = self.pb_set_wd.sizeHint()
        self.pb_set_wd.setMaximumSize(pb_size.height() * 2 // 3, pb_size.height())
        self.pb_set_wd.clicked.connect(self.pb_set_wd_clicked)

        self.le_wd = QLineEdit()
        self.le_wd.setReadOnly(True)

        self.le_wd.setText("/Example/Of/Some/Long/Directory/Name")

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_set_wd)
        hbox.addWidget(self.le_wd)

        self.group_wd.setLayout(hbox)

    def _setup_file_group(self):

        self.group_file = QGroupBox("Load Data From File")

        self.pb_file = QPushButton("Read File ...")
        self.pb_file.clicked.connect(self.pb_file_clicked)

        self.cb_file_all_channels = QCheckBox("All channels")
        self.lb_file = QLabel("No file data is loaded")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_file)
        hbox.addWidget(self.cb_file_all_channels)
        vbox.addLayout(hbox)

        vbox.addWidget(self.lb_file)

        self.group_file.setLayout(vbox)

    def _setup_dbase_group(self):

        self.group_dbase = QGroupBox("Load Data From Database")

        self.pb_dbase = QPushButton("Load Data ...")
        self.pb_dbase.clicked.connect(self.pb_dbase_clicked)

        self.cb_dbase_all_channels = QCheckBox("All channels")
        self.lb_dbase = QLabel("No scan data is loaded")

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_dbase)
        hbox.addWidget(self.cb_dbase_all_channels)
        vbox.addLayout(hbox)

        vbox.addWidget(self.lb_dbase)

        self.group_dbase.setLayout(vbox)

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

        hbox = QHBoxLayout()
        hbox.addWidget(self.pb_apply_mask)
        hbox.addStretch(1)

        self.group_spec_settings.setLayout(hbox)

    def _setup_preview_group(self):

        self.group_preview = QGroupBox("Preview")

        self.list_preview = QListWidget()
        sample_items = ["channel_name_sum", "channel_name_det1",
                        "channel_name_det2", "channel_name_det3"]
        for s in sample_items:
            wi = QListWidgetItem(s, self.list_preview)
            wi.setFlags(wi.flags() | Qt.ItemIsUserCheckable)
            wi.setCheckState(Qt.Unchecked)
        self.list_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_preview.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        hbox = QHBoxLayout()
        hbox.addWidget(self.list_preview)

        self.group_preview.setLayout(hbox)

    def paintEvent(self, event):

        # Adjust height of the list for spectrum previews
        list_size = self.list_preview.size()
        n_list_elements = self.list_preview.count()
        if n_list_elements:
            # Compute the height necessary to accommodate all the elements
            height = self.list_preview.sizeHintForRow(0) * n_list_elements + \
                    2 * self.list_preview.frameWidth() + 3
        else:
            # Set some visually pleasing height if the list contains no elements
            height = 40
        self.list_preview.setFixedSize(list_size.width(), height)

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


class DialogSelectScan(QDialog):

    def __init__(self):

        super().__init__()

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(button_box)
        self.setLayout(vbox)

        self.setWindowTitle("Open Scan")
