from qtpy.QtWidgets import QWidget, QSizePolicy


class FormBaseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(400)
        self.setMinimumWidth(400)

        sp = self.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(sp)
