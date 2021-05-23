import copy
import pytest

from pyxrf.gui_module.useful_widgets import global_gui_variables
from pyxrf.gui_module.wnd_image_wizard import WndImageWizard

# =====================================================================
#                 class WndImageWizard (SecondaryWindow)

# fmt: off
sample_table_content = [
    ["Ar_K", 4.277408, 307.452117],
    ["Ca_K", 0.000000, 1.750295e+03],
    ["Fe_K", 17.902211, 4.576803e+04],
    ["Tb_L", 0.000000, 2.785734e+03],
    ["Ti_K", 0.055362, 1.227637e+04],
    ["Zn_K", 0.000000, 892.008670],
    ["compton", 0.000000, 249.055352],
    ["elastic", 0.000000, 163.153881],
    ["i0", 1.715700e+04, 1.187463e+06],
    ["i0_time", 3.066255e+06, 1.727313e+08],
]
# fmt: on


def _get_gpc_sim(maps_info_table):
    """Simulated simplified 'GlobalProcessingClasses` class for testing"""

    class SimGlobalProcessingClasses:
        def __init__(self):
            self.range_table = copy.deepcopy(maps_info_table)
            # Selection limit table may be the same
            self.limit_table = copy.deepcopy(maps_info_table)
            # Show status table
            self.show_table = [[_[0], False] for _ in self.range_table]

        def get_maps_info_table(self):
            # TODO: implementation needed
            return self.range_table, self.limit_table, self.show_table

    return SimGlobalProcessingClasses()


# fmt: off
@pytest.mark.parametrize("sample_table_data", [
    [],  # Empty table
    [sample_table_content[0]],  # Only one table row
    sample_table_content])  # Full table
# fmt: on
def test_WndImageWizard_1(qtbot, sample_table_data):
    """
    Fill the table and show the window.
    Test if the table data is set successfully and the table contains
    the expected number of rows.
    """
    gpc = _get_gpc_sim(sample_table_data)
    gui_vars = global_gui_variables

    wnd = WndImageWizard(gpc=gpc, gui_vars=gui_vars)
    qtbot.addWidget(wnd)
    wnd.show()

    wnd.slot_update_table()
    # Check that the table was saved correctly
    assert wnd._range_data == sample_table_data, "Table data was set incorrectly"
    # Check the number of table rows
    assert wnd.table.rowCount() == len(sample_table_data), "Incorrect number of table rows"
