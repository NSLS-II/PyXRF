import numpy as np
import numpy.testing as npt
import pytest
from pyxrf.gui_support.gpc_class import GlobalProcessingClasses

# =====================================================================
#                 class GlobalProcessingClasses (SecondaryWindow)


# fmt: off
@pytest.mark.parametrize("eline_keys", [
    [],
    ["Ca_K"],
    ["K_K", "Ca_K", "Fe_K", "positions", "some_key"]
])
# fmt: on
def test_gpc_get_maps_info_table_1(eline_keys):

    gpc = GlobalProcessingClasses()
    gpc.initialize()

    def _get_rand(v_low, v_high):
        return v_low + np.random.rand() * (v_high - v_low)

    # Generate tables
    range_table, limit_table, limit_table_norm, show_table = [], [], [], []
    for key in eline_keys:
        rng_min = _get_rand(0, 10)
        rng_max = _get_rand(20, 30)
        sel_min = _get_rand(rng_min, rng_min + (rng_max - rng_min) / 3)
        sel_max = _get_rand(rng_min + 2 * (rng_max - rng_min) / 3, rng_max)
        sel_min_norm = (sel_min - rng_min) / (rng_max - rng_min) * 100.0
        sel_max_norm = (sel_max - rng_min) / (rng_max - rng_min) * 100.0
        range_table.append([key, rng_min, rng_max])
        limit_table.append([key, sel_min, sel_max])
        limit_table_norm.append([key, sel_min_norm, sel_max_norm])
        show_table.append([key, np.random.rand() > 0.5])

    # 'Artificially' setup some fileds of 'gpc.img_model_adv'
    gpc.img_model_adv.map_keys = eline_keys
    gpc.img_model_adv.range_dict = {}
    gpc.img_model_adv.limit_dict = {}
    gpc.img_model_adv.stat_dict = {}
    for n, key in enumerate(eline_keys):
        gpc.img_model_adv.range_dict[key] = {}
        gpc.img_model_adv.range_dict[key]["low"] = range_table[n][1]
        gpc.img_model_adv.range_dict[key]["high"] = range_table[n][2]
        gpc.img_model_adv.limit_dict[key] = {}
        gpc.img_model_adv.limit_dict[key]["low"] = limit_table_norm[n][1]
        gpc.img_model_adv.limit_dict[key]["high"] = limit_table_norm[n][2]
        gpc.img_model_adv.stat_dict[key] = show_table[n][1]

    # Now run the test
    rng_table, lim_table, sh_table = gpc.get_maps_info_table()
    assert rng_table == range_table, "Returned table of ranges does not match the expected."
    assert [_[0] for _ in lim_table] == [
        _[0] for _ in limit_table
    ], "Keys in the returned table of limits do not match the expected."
    npt.assert_array_almost_equal(
        [_[1:] for _ in lim_table],
        [_[1:] for _ in limit_table],
        err_msg="Returned limits don't match the expected",
    )
    assert sh_table == show_table, "'show' status table don't match the expected table"


def test_gpc_get_maps_info_table_2():
    """Failing cases for `GlobalProcessingClasses.get_maps_info_table()`"""

    gpc = GlobalProcessingClasses()
    gpc.initialize()

    eline_keys = ["abc", "def"]  # Some random keys
    gpc.img_model_adv.map_keys = eline_keys
    correct_dict = {_: {"low": 0.0, "high": 1.0} for _ in eline_keys}
    correct_stat_dict = {_: True for _ in eline_keys}

    # Modified 'limit_dict'
    modified_dict = correct_dict.copy()
    modified_dict["extra_key"] = {"low": 0.0, "high": 1.0}

    modified_stat_dict = correct_stat_dict.copy()
    modified_stat_dict["extra_key"] = False

    gpc.img_model_adv.range_dict = correct_dict
    gpc.img_model_adv.limit_dict = modified_dict
    gpc.img_model_adv.stat_dict = correct_stat_dict

    with pytest.raises(RuntimeError, match="list of keys in 'limit_dict'"):
        gpc.get_maps_info_table()

    gpc.img_model_adv.range_dict = modified_dict
    gpc.img_model_adv.limit_dict = correct_dict
    gpc.img_model_adv.stat_dict = correct_stat_dict

    with pytest.raises(RuntimeError, match="list of keys in 'range_dict'"):
        gpc.get_maps_info_table()

    gpc.img_model_adv.range_dict = correct_dict
    gpc.img_model_adv.limit_dict = correct_dict
    gpc.img_model_adv.stat_dict = modified_stat_dict

    with pytest.raises(RuntimeError, match="list of keys in 'stat_dict'"):
        gpc.get_maps_info_table()
