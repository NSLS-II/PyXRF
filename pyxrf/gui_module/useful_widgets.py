from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QPalette


global_gui_parameters = {
    "vertical_spacing_in_tabs": 5
}


class LineEditReadOnly(QLineEdit):
    """
    Read-only version of QLineEdit with background set to the same color
    as the background of the disabled QLineEdit, but font color the same
    as active QLineEdit.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        # Set background color the same as for disabled window.
        p = self.palette()
        p.setColor(QPalette.Base, p.color(QPalette.Disabled, QPalette.Base))
        self.setPalette(p)


def adjust_qlistwidget_height(list_widget, min_height=40):
    """
    Adjust the height of QListWidget so that it fits the items exactly.
    If the computed height is smaller than `min_height`, then the height
    is set to `min_height`. If the list is empty, then the height is set
    to `min_height` (for pleasing look).

    The width of the widget is still adjusted automatically.
    Call this function each time the number of items in the list
    is changed.

    Parameters
    ----------
    list_widget: QListWidget
        reference to the list widget that needs to be adjusted.
    min_height: int
        minimum height of the widget.
    """

    # Compute and set the height of the list
    height = 0
    n_list_elements = list_widget.count()
    if n_list_elements:
        # Compute the height necessary to accommodate all the elements
        height = list_widget.sizeHintForRow(0) * n_list_elements + \
                2 * list_widget.frameWidth() + 3
    # Set some visually pleasing height if the list contains no elements
    height = max(height, min_height)
    list_widget.setMinimumHeight(height)
    list_widget.setMaximumHeight(height)
