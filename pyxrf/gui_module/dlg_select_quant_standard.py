import textwrap

from qtpy.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetSelectionRange,
    QTableWidgetItem,
    QHeaderView,
)
from qtpy.QtGui import QBrush, QColor
from qtpy.QtCore import Qt

from .useful_widgets import set_tooltip

"""
# This table may be useful for unit tests that are yet to be written. So keep it for now.
sample_table_contents = [
    ["41147", "Micromatter 41147",
     "GaP 21.2 (Ga=15.4, P=5.8) / CaF2 14.6 / V 26.4 / Mn 17.5 / Co 21.7 / Cu 20.3"],
    ["41148", "Micromatter 41148",
     "GaP 21.1 (Ga=15.4, P=5.7) / CaF2 14.6 / V 25.1 / Mn 18.1 / Co 20.6 / Cu 20.5"],
    ["41151", "Micromatter 41151",
     "KCl 21.9 / Ti 21.3 / Fe 24.4 / ZnTe 20.3 / Pb 22.3"],
    ["41164", "Micromatter 41164",
     "CeF3 21.1 / Au 20.6"],
]
"""


class DialogSelectQuantStandard(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)

        self._qe_param_built_in = []
        self._qe_param_custom = []
        self._qe_standard_selected = None
        self._qe_param = []  # The list of all standards
        self._custom_label = []  # The list of booleans: True - the standard is custom, False -built-in

        self.setWindowTitle("Load Quantitative Standard")

        self.setMinimumHeight(500)
        self.setMinimumWidth(600)
        self.resize(600, 500)

        self.selected_standard_index = -1

        labels = ("C", "Serial #", "Name", "Description")
        col_stretch = (
            QHeaderView.ResizeToContents,
            QHeaderView.ResizeToContents,
            QHeaderView.ResizeToContents,
            QHeaderView.Stretch,
        )
        self.table = QTableWidget()
        set_tooltip(
            self.table,
            "To <b> load the sta"
            "ndard</b>, double-click on the table row or "
            "select the table row and then click <b>Ok</b> button.",
        )
        self.table.setMinimumHeight(200)
        self.table.setColumnCount(len(labels))
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.item_selection_changed)
        self.table.itemDoubleClicked.connect(self.item_double_clicked)

        self.table.setStyleSheet(
            "QTableWidget::item{color: black;}"
            "QTableWidget::item:selected{background-color: red;}"
            "QTableWidget::item:selected{color: white;}"
        )

        header = self.table.horizontalHeader()
        for n, col_stretch in enumerate(col_stretch):
            # Set stretching for the columns
            header.setSectionResizeMode(n, col_stretch)

        self.lb_info = QLabel()
        self.lb_info.setText("Column 'C': * means that the standard is user-defined.")

        button_box = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Cancel).setDefault(True)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.pb_open = button_box.button(QDialogButtonBox.Open)
        self.pb_open.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.table)
        vbox.addWidget(self.lb_info)
        vbox.addWidget(button_box)
        self.setLayout(vbox)

    def _fill_table(self, table_contents):
        self.table.setRowCount(len(table_contents))

        for nr, row in enumerate(table_contents):
            for nc, entry in enumerate(row):
                s = textwrap.fill(entry, width=40)
                item = QTableWidgetItem(s)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if not nc:
                    item.setTextAlignment(int(Qt.AlignHCenter | Qt.AlignVCenter))
                self.table.setItem(nr, nc, item)
        self.table.resizeRowsToContents()

        brightness = 220
        for nr in range(self.table.rowCount()):
            for nc in range(self.table.columnCount()):
                self.table.item(nr, nc)
                if nr % 2:
                    color = QColor(255, brightness, brightness)
                else:
                    color = QColor(brightness, 255, brightness)
                self.table.item(nr, nc).setBackground(QBrush(color))

        try:
            index = self._qe_param.index(self._qe_standard_selected)
            self.selected_standard_index = index
            n_columns = self.table.columnCount()
            self.table.setRangeSelected(QTableWidgetSelectionRange(index, 0, index, n_columns - 1), True)
        except ValueError:
            pass

    def item_selection_changed(self):
        sel_ranges = self.table.selectedRanges()
        # The table is configured to have one or no selected ranges
        # 'Open' button should be enabled only if a range (row) is selected
        if sel_ranges:
            self.selected_standard_index = sel_ranges[0].topRow()
            self.pb_open.setEnabled(True)
        else:
            self.selected_standard_index = -1
            self.pb_open.setEnabled(False)

    def item_double_clicked(self):
        self.accept()

    def set_standards(self, qe_param_built_in, qe_param_custom, qe_standard_selected):
        self._qe_standard_selected = qe_standard_selected
        self._qe_param = qe_param_custom + qe_param_built_in
        custom_label = [True] * len(qe_param_custom) + [False] * len(qe_param_built_in)

        table_contents = []
        for n, param in enumerate(self._qe_param):
            custom = "*" if custom_label[n] else ""
            serial = param["serial"]
            name = param["name"]
            description = param["description"]
            table_contents.append([custom, serial, name, description])

        self._fill_table(table_contents)

    def get_selected_standard(self):
        if self.selected_standard_index >= 0:
            return self._qe_param[self.selected_standard_index]
        else:
            return None
