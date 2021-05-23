import os

from qtpy.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QTextEdit,
)
from qtpy.QtCore import Qt

from .useful_widgets import LineEditReadOnly, PushButtonMinimumWidth, set_tooltip

import logging

logger = logging.getLogger(__name__)


class DialogExportToTiffAndTxt(QDialog):
    def __init__(self, parent=None, *, dir_path=""):

        super().__init__(parent)

        self.__dir_path = ""
        self.__save_tiff = True
        self.__save_txt = False
        self.__interpolate_on = False
        self.__quant_norm_on = False
        self.__dset_list = []
        self.__dset_sel = 0
        self.__scaler_list = []
        self.__scaler_sel = 0

        self.setWindowTitle("Export XRF Maps to TIFF and/or TXT Files")
        self.setMinimumHeight(600)
        self.setMinimumWidth(600)
        self.resize(600, 600)

        self.te_saved_files = QTextEdit()
        set_tooltip(self.te_saved_files, "The list of <b>data file groups</b> about to be created.")
        self.te_saved_files.setReadOnly(True)

        self.combo_select_dataset = QComboBox()
        self.combo_select_dataset.currentIndexChanged.connect(self.combo_select_dataset_current_index_changed)
        self._fill_dataset_combo()
        set_tooltip(
            self.combo_select_dataset,
            "Select <b>dataset</b>. Initially, the selection matches the dataset activated "
            "in <b>XRF Maps</b> tab, but the selection may be changed if different dataset "
            "needs to be saved.",
        )

        self.combo_normalization = QComboBox()
        self.combo_normalization.currentIndexChanged.connect(self.combo_normalization_current_index_changed)
        self._fill_scaler_combo()
        set_tooltip(
            self.combo_normalization,
            "Select <b>scaler</b> used for data normalization. Initially, the selection matches "
            "the scaler activated in <b>XRF Maps</b> tab, but the selection may be changed "
            "if needed",
        )

        self.cb_interpolate = QCheckBox("Interpolate to uniform grid")
        self.cb_interpolate.setChecked(Qt.Checked if self.__interpolate_on else Qt.Unchecked)
        self.cb_interpolate.stateChanged.connect(self.cb_interpolate_state_changed)
        set_tooltip(
            self.cb_interpolate,
            "Interpolate pixel coordinates to <b>uniform grid</b>. The initial choice is "
            "copied from <b>XRF Maps</b> tab.",
        )

        self.cb_quantitative = QCheckBox("Quantitative normalization")
        self.cb_quantitative.setChecked(Qt.Checked if self.__quant_norm_on else Qt.Unchecked)
        self.cb_quantitative.stateChanged.connect(self.cb_quantitative_state_changed)
        set_tooltip(
            self.cb_quantitative,
            "Apply <b>quantitative normalization</b> before saving the maps. "
            "The initial choice is copied from <b>XRF Maps</b> tab.",
        )

        self.group_settings = QGroupBox("Settings (selections from XRF Maps tab)")
        grid = QGridLayout()
        grid.addWidget(self.combo_select_dataset, 0, 0)
        grid.addWidget(self.combo_normalization, 0, 1)
        grid.addWidget(self.cb_interpolate, 1, 0)
        grid.addWidget(self.cb_quantitative, 1, 1)
        self.group_settings.setLayout(grid)

        self.le_dir_path = LineEditReadOnly()
        set_tooltip(
            self.le_dir_path,
            "<b>Root directory</b> for saving TIFF and TXT files. The files will be saved "
            "in subdirectories inside the root directory.",
        )
        self.pb_dir_path = PushButtonMinimumWidth("..")
        set_tooltip(self.pb_dir_path, "Change to <b>root directory</b> for TIFF and TXT files.")
        self.pb_dir_path.clicked.connect(self.pb_dir_path_clicked)
        self.pb_dir_path.setDefault(False)
        self.pb_dir_path.setAutoDefault(False)

        self.cb_save_tiff = QCheckBox("Save TIFF")
        set_tooltip(self.cb_save_tiff, "Save XRF Maps as <b>TIFF</b> files.")
        self.cb_save_tiff.setChecked(Qt.Checked if self.__save_tiff else Qt.Unchecked)
        self.cb_save_tiff.stateChanged.connect(self.cb_save_tiff_state_changed)
        self.cb_save_txt = QCheckBox("Save TXT")
        self.cb_save_txt.setChecked(Qt.Checked if self.__save_txt else Qt.Unchecked)
        set_tooltip(self.cb_save_txt, "Save XRF Maps as <b>TXT</b> files.")
        self.cb_save_txt.stateChanged.connect(self.cb_save_txt_state_changed)

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.group_settings)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("The following groups of files will be created:"))
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(self.te_saved_files)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Directory: "))
        hbox.addWidget(self.pb_dir_path)
        hbox.addWidget(self.le_dir_path)
        vbox.addLayout(hbox)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.pb_save = self.button_box.button(QDialogButtonBox.Save)
        self.pb_save.setDefault(False)
        self.pb_save.setAutoDefault(False)
        self.pb_cancel = self.button_box.button(QDialogButtonBox.Cancel)
        self.pb_cancel.setDefault(True)
        self.pb_cancel.setAutoDefault(True)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.cb_save_tiff)
        hbox.addWidget(self.cb_save_txt)
        hbox.addStretch(1)
        hbox.addWidget(self.button_box)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # Set and display file path
        dir_path = os.path.expanduser(dir_path)
        self.dir_path = dir_path

        self._update_pb_save()
        self._update_saved_file_groups()

    @property
    def dir_path(self):
        return self.__dir_path

    @dir_path.setter
    def dir_path(self, dir_path):
        self.__dir_path = dir_path
        self.le_dir_path.setText(dir_path)

    @property
    def save_tiff(self):
        return self.__save_tiff

    @save_tiff.setter
    def save_tiff(self, save_tiff):
        self.__save_tiff = save_tiff
        self.cb_save_tiff.setChecked(Qt.Checked if save_tiff else Qt.Unchecked)
        self._update_pb_save()

    @property
    def save_txt(self):
        return self.__save_txt

    @save_txt.setter
    def save_txt(self, save_txt):
        self.__save_txt = save_txt
        self.cb_save_txt.setChecked(Qt.Checked if save_txt else Qt.Unchecked)
        self._update_pb_save()

    @property
    def interpolate_on(self):
        return self.__interpolate_on

    @interpolate_on.setter
    def interpolate_on(self, interpolate_on):
        self.__interpolate_on = interpolate_on
        self.cb_interpolate.setChecked(Qt.Checked if interpolate_on else Qt.Unchecked)

    @property
    def quant_norm_on(self):
        return self.__quant_norm_on

    @quant_norm_on.setter
    def quant_norm_on(self, quant_norm_on):
        self.__quant_norm_on = quant_norm_on
        self.cb_quantitative.setChecked(Qt.Checked if quant_norm_on else Qt.Unchecked)

    @property
    def dset_list(self):
        return self.__dset_list

    @dset_list.setter
    def dset_list(self, dset_list):
        self.__dset_list = dset_list
        self._fill_dataset_combo()

    @property
    def dset_sel(self):
        return self.__dset_sel

    @dset_sel.setter
    def dset_sel(self, dset_sel):
        self.__dset_sel = dset_sel
        self.combo_select_dataset.setCurrentIndex(dset_sel - 1)

    @property
    def scaler_list(self):
        return self.__scaler_list

    @scaler_list.setter
    def scaler_list(self, scaler_list):
        self.__scaler_list = scaler_list
        self._fill_scaler_combo()

    @property
    def scaler_sel(self):
        return self.__scaler_sel

    @scaler_sel.setter
    def scaler_sel(self, scaler_sel):
        self.__scaler_sel = scaler_sel
        self.combo_normalization.setCurrentIndex(scaler_sel)

    def pb_dir_path_clicked(self):
        # Note: QFileDialog.ShowDirsOnly is not set on purpose, so that the dialog
        #   could be used to inspect directory contents. Files can not be selected anyway.
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Root Directory for TIFF and TXT Files",
            self.dir_path,
            options=QFileDialog.DontResolveSymlinks,
        )
        if dir_path:
            self.dir_path = dir_path

    def _update_pb_save(self):
        state = self.__save_tiff or self.__save_txt
        self.pb_save.setEnabled(state)

    def cb_save_tiff_state_changed(self, state):
        self.__save_tiff = state == Qt.Checked
        self._update_pb_save()
        self._update_saved_file_groups()

    def cb_save_txt_state_changed(self, state):
        self.__save_txt = state == Qt.Checked
        self._update_pb_save()
        self._update_saved_file_groups()

    def cb_interpolate_state_changed(self, state):
        self.__interpolate_on = state == Qt.Checked

    def cb_quantitative_state_changed(self, state):
        self.__quant_norm_on = state == Qt.Checked
        self._update_saved_file_groups()

    def combo_select_dataset_current_index_changed(self, index):
        self.__dset_sel = index + 1
        self._update_saved_file_groups()

    def combo_normalization_current_index_changed(self, index):
        self.__scaler_sel = index
        self._update_saved_file_groups()

    def get_selected_dset_name(self):
        index = self.__dset_sel - 1
        n = len(self.__dset_list)
        if index < 0 or index >= n:
            return None
        else:
            return self.__dset_list[index]

    def get_selected_scaler_name(self):
        index = self.__scaler_sel - 1
        n = len(self.__scaler_list)
        if index < 0 or index >= n:
            return None
        else:
            return self.__scaler_list[index]

    def _fill_dataset_combo(self):
        self.combo_select_dataset.clear()
        self.combo_select_dataset.addItems(self.__dset_list)
        self.combo_select_dataset.setCurrentIndex(self.__dset_sel - 1)

    def _fill_scaler_combo(self):
        scalers = ["Normalize by ..."] + self.__scaler_list
        self.combo_normalization.clear()
        self.combo_normalization.addItems(scalers)
        self.combo_normalization.setCurrentIndex(self.__scaler_sel)

    def _update_saved_file_groups(self):
        dset_name = self.get_selected_dset_name()
        scaler_name = self.get_selected_scaler_name()
        is_fit, is_roi = False, False
        if dset_name is not None:
            is_fit = dset_name.endswith("fit")
            is_roi = dset_name.endswith("roi")

        text_common = ""
        if is_fit:
            text_common += "  - Fitted XRF maps\n"
        elif is_roi:
            text_common += "  - ROI maps\n"
        if (is_fit or is_roi) and (scaler_name is not None):
            text_common += f"  - Normalized maps (scaler '{scaler_name}')\n"
        if is_fit and self.__quant_norm_on:
            text_common += "  - Quantitative maps (if calibration data is loaded)\n"
        text_common += "  - Scalers\n"
        text_common += "  - Positional coordinates\n"

        text = ""
        if self.__save_tiff:
            text += "TIFF FORMAT:\n" + text_common
        if self.__save_txt:
            if text:
                text += "\n"
            text += "TXT_FORMAT:\n" + text_common
        if not text:
            text = "No files will be saved"
        self.te_saved_files.setText(text)
