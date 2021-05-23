from qtpy.QtWidgets import QLabel, QVBoxLayout, QComboBox, QGridLayout, QDialog, QDialogButtonBox, QGroupBox

from .useful_widgets import LineEditReadOnly, set_tooltip


class DialogPlotEscapePeak(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Escape Peak Settings")

        self.grp_show_escape_peak = QGroupBox("Show Escape Peak")
        self.grp_show_escape_peak.setCheckable(True)
        self.grp_show_escape_peak.setChecked(False)  # Set based on data !!!

        self.le_incident_energy = LineEditReadOnly()
        set_tooltip(
            self.le_incident_energy,
            "<b>Incident energy</b>. Use <b>General...</b> button of <b>Model</b> tab "
            "to change the value if needed.",
        )
        self.combo_detector_type = QComboBox()
        set_tooltip(self.combo_detector_type, "Select <b>detector</b> material. The typical choice is <b>Si</b>")

        self._detector_types = ["Si", "Ge"]
        self.combo_detector_type.addItems(self._detector_types)

        grid = QGridLayout()
        grid.addWidget(QLabel("Incident energy, kev:"), 0, 0)
        grid.addWidget(self.le_incident_energy, 0, 1)
        grid.addWidget(QLabel("Detectory type:"), 1, 0)
        grid.addWidget(self.combo_detector_type, 1, 1)
        self.grp_show_escape_peak.setLayout(grid)

        # Yes/No button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.grp_show_escape_peak)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def set_parameters(self, plot_escape_peak, incident_energy, detector_material):
        index_material = -1
        if detector_material in self._detector_types:
            index_material = self._detector_types.index(detector_material)
        self.combo_detector_type.setCurrentIndex(index_material)

        self.le_incident_energy.setText(f"{incident_energy:.10g}")

        self.grp_show_escape_peak.setChecked(plot_escape_peak)

    def get_parameters(self):
        plot_escape_peak = self.grp_show_escape_peak.isChecked()

        index = self.combo_detector_type.currentIndex()
        if index >= 0:
            detector_material = self._detector_types[self.combo_detector_type.currentIndex()]
        else:
            detector_material = ""

        return plot_escape_peak, detector_material
