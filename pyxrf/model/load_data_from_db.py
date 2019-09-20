from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import h5py
import numpy as np
import os
import json
import multiprocessing
import pandas as pd
import math

import logging
import warnings

logger = logging.getLogger()
warnings.filterwarnings('ignore')

sep_v = os.sep

try:
    config_path = '/etc/pyxrf/pyxrf.json'
    with open(config_path, 'r') as beamline_pyxrf:
        beamline_config_pyxrf = json.load(beamline_pyxrf)
        beamline_name = beamline_config_pyxrf['beamline_name']
    if beamline_name == 'HXN':
        from pyxrf.db_config.hxn_db_config import db
    elif beamline_name == 'SRX':
        from pyxrf.db_config.srx_db_config import db
    elif beamline_name == 'XFM':
        from pyxrf.db_config.xfm_db_config import db
    else:
        db = None
        db_analysis = None
        print('Beamline Database is not used in pyxrf.')
except IOError:
    db = None
    print('Beamline Database is not used in pyxrf.')

# try:
#     import suitcase.hdf5 as sc
# except ImportError:
#     pass


def flip_data(input_data, subscan_dims=None):
    """
    Flip 2D or 3D array. The flip happens on the second index of shape.
    .. warning :: This function mutates the input values.

    Parameters
    ----------
    input_data : 2D or 3D array.

    Returns
    -------
    flipped data
    """
    new_data = np.asarray(input_data)
    data_shape = input_data.shape
    if len(data_shape) == 2:
        if subscan_dims is None:
            new_data[1::2, :] = new_data[1::2, ::-1]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :] = new_data[start:end:2, ::-1]
                i += ny

    if len(data_shape) == 3:
        if subscan_dims is None:
            new_data[1::2, :, :] = new_data[1::2, ::-1, :]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :, :] = new_data[start:end:2, ::-1, :]
                i += ny
    return new_data


def fetch_data_from_db(runid, fpath=None,
                       create_each_det=False,
                       output_to_file=False,
                       save_scalar=True,
                       num_end_lines_excluded=None):
    """
    Read data from databroker.
    This is the place where new beamlines can be easily added
    to pyxrf GUI.
    Save the data from databroker to hdf file if needed.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        id number for given run
    fpath: str, optional
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    output_to_file : bool, optional
        save data to hdf5 file if True
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[-1]
    print('Loading data from database.')

    if hdr.start.beamline_id == 'HXN':
        data = map_data2D_hxn(runid, fpath,
                              create_each_det=create_each_det,
                              output_to_file=output_to_file)
    elif (hdr.start.beamline_id == 'xf05id' or
          hdr.start.beamline_id == 'SRX'):
        data = map_data2D_srx(runid, fpath,
                              create_each_det=create_each_det,
                              output_to_file=output_to_file,
                              save_scalar=save_scalar,
                              num_end_lines_excluded=num_end_lines_excluded)
    elif hdr.start.beamline_id == 'XFM':
        data = map_data2D_xfm(runid, fpath,
                              create_each_det=create_each_det,
                              output_to_file=output_to_file)
    else:
        print("Databroker is not setup for this beamline")
        return
    free_memory_from_handler()
    return data


def make_hdf(start, end=None, fname=None,
             prefix='scan2D_',
             create_each_det=True, save_scalar=True,
             num_end_lines_excluded=None):
    """
    Transfer multiple h5 files.

    Parameters
    ---------
    start : int
        start run id
    end : int, optional
        end run id
    fname : string
        path to save file when start equals to end, in this case only
        one file is transfered.
    prefix : str, optional
        prefix name of the file
    db : databroker
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines. Used at SRX beamline.
    """
    if end is None:
        end = start

    if end == start:
        if fname is None:
            fname = prefix+str(start)+'.h5'
        fetch_data_from_db(start, fpath=fname,
                           create_each_det=create_each_det,
                           output_to_file=True,
                           save_scalar=save_scalar,
                           num_end_lines_excluded=num_end_lines_excluded)
    else:
        datalist = range(start, end+1)
        for v in datalist:
            filename = prefix+str(v)+'.h5'
            try:
                fetch_data_from_db(v, fpath=filename,
                                   create_each_det=create_each_det,
                                   output_to_file=True,
                                   save_scalar=save_scalar,
                                   num_end_lines_excluded=num_end_lines_excluded)
                print('{} is created. \n'.format(filename))
            except Exception:
                print('Can not transfer scan {}. \n'.format(v))


def map_data2D_hxn(runid, fpath,
                   create_each_det=False,
                   output_to_file=True):
    """
    Save the data from databroker to hdf file.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        id number for given run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    output_to_file : bool, optional
        save data to hdf5 file if True
    """
    hdr = db[runid]

    start_doc = hdr['start']
    if 'dimensions' in start_doc:
        datashape = start_doc.dimensions
    elif 'shape' in start_doc:
        datashape = start_doc.shape
    else:
        logger.error('No dimension/shape is defined in hdr.start.')

    datashape = [datashape[1], datashape[0]]  # vertical first, then horizontal
    fly_type = start_doc.get('fly_type', None)
    subscan_dims = start_doc.get('subscan_dims', None)

    if 'motors' in hdr.start:
        pos_list = hdr.start.motors
    elif 'axes' in hdr.start:
        pos_list = hdr.start.axes
    else:
        pos_list = ['zpssx[um]', 'zpssy[um]']

    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = 'hxn_pv_config.json'
    config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)

    keylist = hdr.descriptors[0].data_keys.keys()
    det_list = [v for v in keylist if 'xspress3' in v]  # find xspress3 det with key word matching

    scaler_list_all = config_data['scaler_list']

    all_keys = hdr.descriptors[0].data_keys.keys()
    scaler_list = [v for v in scaler_list_all if v in all_keys]

    # fields = det_list + scaler_list + pos_list
    data = db.get_table(hdr, fill=True)

    data_out = map_data2D(data, datashape,
                          det_list=det_list,
                          pos_list=pos_list,
                          scaler_list=scaler_list,
                          fly_type=fly_type, subscan_dims=subscan_dims,
                          spectrum_len=4096)
    if output_to_file:
        # output to file
        print('Saving data to hdf file.')
        write_db_to_hdf_base(fpath, data_out,
                             create_each_det=create_each_det)
    return data_out
    # write_db_to_hdf(fpath, data, datashape,
    #                 det_list=det_list, pos_list=pos_list,
    #                 scaler_list=scaler_list,
    #                 fly_type=fly_type, subscan_dims=subscan_dims)
    #
    # # use suitcase to save baseline data, and scaler data from primary
    # tmp = set()
    # for descriptor in hdr.descriptors:
    #     # no 3D vector data
    #     xs3 = [key for key in descriptor.data_keys.keys() if 'xspress3' in key]
    #     tmp.update(xs3)
    #     tmp.add('merlin1')
    # fds = sc.filter_fields(hdr, tmp)
    # if full_data == True:
    #     sc.export(hdr, fpath, db.mds, fields=fds, use_uid=False)


def get_total_scan_point(hdr):
    """
    Find the how many data points are recorded. This number may not equal to the total number
    defined at the start of the scan due to scan stop or abort.
    """
    evs = hdr.events()
    n = 0
    try:
        for e in evs:
            n = n+1
    except IndexError:
        pass
    return n


def map_data2D_srx(runid, fpath,
                   create_each_det=False, output_to_file=True,
                   save_scalar=True, num_end_lines_excluded=None):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at SRX beamline for both fly scan and step scan.
    Save to hdf file if needed.

    .. note:: Requires the databroker package from NSLS2

    Parameters
    ----------
    runid : int
        id number for given run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    output_to_file : bool, optional
        save data to hdf5 file if True
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[runid]
    spectrum_len = 4096
    start_doc = hdr['start']
    plan_n = start_doc.get('plan_name')

    # Load configuration file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = 'srx_pv_config.json'
    config_path = sep_v.join(current_dir.split(sep_v)[:-2] + ['configs', config_file])
    with open(config_path, 'r') as json_data:
        config_data = json.load(json_data)

    if 'fly' not in plan_n:  # not fly scan

        print()
        print(f"****************************************")
        print(f"        Loading SRX step scan           ")
        print(f"****************************************")

        fly_type = None

        if num_end_lines_excluded is None:
            # vertical first then horizontal, assuming fast scan on x
            datashape = [start_doc['shape'][1], start_doc['shape'][0]]
        else:
            datashape = [start_doc['shape'][1] - num_end_lines_excluded, start_doc['shape'][0]]

        snake_scan = start_doc.get('snaking')
        if snake_scan[1] is True:
            fly_type = 'pyramid'

        try:
            data = hdr.table(fill=True, convert_times=False)

        except IndexError:
            total_len = get_total_scan_point(hdr) - 2
            evs, _ = zip(*zip(hdr.events(fill=True), range(total_len)))
            namelist = config_data['xrf_detector'] + hdr.start.motors + config_data['scaler_list']
            dictv = {v: [] for v in namelist}
            for e in evs:
                for k, v in six.iteritems(dictv):
                    dictv[k].append(e.data[k])
            data = pd.DataFrame(dictv, index=np.arange(1, total_len+1))  # need to start with 1

        #  Commented by DG: Just use the detector names from .json configuration file. Do not delete the code.
        # express3 detector name changes in databroker
        # if xrf_detector_names[0] not in data.keys():
        #     xrf_detector_names = ['xs_channel'+str(i) for i in range(1,4)]
        #     config_data['xrf_detector'] = xrf_detector_names

        if output_to_file:
            if 'xs' in hdr.start.detectors:
                print('Saving data to hdf file: Xpress3 detector #1 (three channels).')
                root, ext = os.path.splitext(fpath)
                fpath_out = f"{root + '_xs'}{ext}"
                write_db_to_hdf(fpath_out, data,
                                # hdr.start.datashape,
                                datashape,
                                det_list=config_data['xrf_detector'],
                                pos_list=hdr.start.motors,
                                scaler_list=config_data['scaler_list'],
                                fly_type=fly_type,
                                base_val=config_data['base_value'])  # base value shift for ic
            if 'xs2' in hdr.start.detectors:
                print('Saving data to hdf file: Xpress3 detector #2 (single channel).')
                root, ext = os.path.splitext(fpath)
                fpath_out = f"{root}_xs2{ext}"
                write_db_to_hdf(fpath_out, data,
                                datashape,
                                det_list=config_data['xrf_detector2'],
                                pos_list=hdr.start.motors,
                                scaler_list=config_data['scaler_list'],
                                fly_type=fly_type,
                                base_val=config_data['base_value'])  # base value shift for ic
        return data

    else:
        # srx fly scan

        print()
        print(f"****************************************")
        print(f"         Loading SRX fly scan           ")
        print(f"****************************************")

        if save_scalar is True:
            scaler_list = ['i0', 'time', 'i0_time', 'time_diff']
            xpos_name = 'enc1'
            ypos_name = 'hf_stage_y'  # 'hf_stage_x' if fast axis is vertical

        # The dictionary of fields that are used to store data from different detectors (for fly scan only)
        #   key - the name of the field used to store data read from the detector
        #   value - the detector name (probably short abbreviation, attached to the created file name so that
        #           the detector could be identified)
        # A separate data file is created for each detector
        detector_field_dict = config_data['xrf_flyscan_detector_fields']

        num_det = 0  # Some default value (should never be used)

        #  Added by AMK to allow flying of single element on xs2
        # if 'E_tomo' in start_doc['scaninfo']['type']:
        #     num_det = 1
        #     ypos_name = 'e_tomo_y'
        # else:
        #     num_det = 3
        vertical_fast = False  # assuming fast on x as default
        if num_end_lines_excluded is None:
            # vertical first then horizontal, assuming fast scan on x
            datashape = [start_doc['shape'][1], start_doc['shape'][0]]
        else:
            datashape = [start_doc['shape'][1]-num_end_lines_excluded, start_doc['shape'][0]]
        if 'fast_axis' in hdr.start.scaninfo:
            # fast scan along vertical, y is fast scan, x is slow
            if hdr.start.scaninfo['fast_axis'] in ('VER', 'DET2VER'):
                xpos_name = 'enc1'
                ypos_name = 'hf_stage_x'
                if 'E_tomo' in start_doc['scaninfo']['type']:
                    ypos_name = 'e_tomo_x'
                vertical_fast = True
                #   fast vertical scan put shape[0] as vertical direction
                # datashape = [start_doc['shape'][0], start_doc['shape'][1]]

        new_shape = datashape + [spectrum_len]
        # total_points = datashape[0]*datashape[1]

        des = [d for d in hdr.descriptors if d.name == 'stream0'][0]
        #   merlin data doesn't need to be saved.
        # un_used_det = ['merlin', 'im'] # data not to be transfered for pyxrf
        # data_list_used = [v for v in des.data_keys.keys() if 'merlin' not in v.lower()]

        # The number of found detectors for which data exists in the database
        n_detectors_found = 0

        # Try each data field listed in the config file
        for detector_field, detector_name in detector_field_dict.items():

            # Assume that Databroker caches the tables locally, so that data will not be reloaded
            e = hdr.events(fill=True, stream_name=des.name)

            new_data = {}
            data = {}

            if save_scalar is True:
                new_data['scaler_names'] = scaler_list
                scaler_tmp = np.zeros([datashape[0], datashape[1], len(scaler_list)])
                if vertical_fast is True:  # data shape only has impact on scalar data
                    scaler_tmp = np.zeros([datashape[1], datashape[0], len(scaler_list)])
                for v in scaler_list+[xpos_name]:
                    data[v] = np.zeros([datashape[0], datashape[1]])

            # Total number of lines in fly scan
            n_scan_lines_total = new_shape[0]

            detector_field_exists = True

            for m, v in enumerate(e):

                if m == 0:

                    # Check if detector field does not exist. If not, then the file should not be created.
                    if detector_field not in v.data:
                        detector_field_exists = False
                        break

                    print()
                    print(f"Collecting data from detector '{detector_name}' (field '{detector_field}')")

                    # Determine the number of channels from the size of the table with fluorescence data
                    num_det = v.data[detector_field].shape[1]

                    # Now allocate space for fluorescence data
                    if create_each_det is False:
                        new_data['det_sum'] = np.zeros(new_shape)
                    else:
                        for i in range(num_det):
                            new_data[f'det{i + 1}'] = np.zeros(new_shape)

                    print(f"Number of the detector channels: {num_det}")

                if m < datashape[0]:   # scan is not finished
                    if save_scalar is True:
                        for n in scaler_list[:-1] + [xpos_name]:
                            min_len = min(v.data[n].size, datashape[1])
                            data[n][m, :min_len] = v.data[n][:min_len]
                            # position data or i0 has shorter length than fluor data
                            if min_len < datashape[1]:
                                len_diff = datashape[1] - min_len
                                # interpolation on scaler data
                                interp_list = (v.data[n][-1] - v.data[n][-3]) / 2 * \
                                    np.arange(1, len_diff + 1) + v.data[n][-1]
                                data[n][m, min_len:datashape[1]] = interp_list
                    fluor_len = v.data[detector_field].shape[0]
                    if m > 0 and not (m % 10):
                        print(f"Processed {m} of {n_scan_lines_total} lines ...")
                    # print(f"m = {m} Data shape {v.data['fluor'].shape} - {v.data['fluor'].shape[1] }")
                    # print(f"Data keys: {v.data.keys()}")
                    if create_each_det is False:
                        for i in range(num_det):
                            # in case the data length in each line is different
                            new_data['det_sum'][m, :fluor_len, :] += v.data[detector_field][:, i, :]
                    else:
                        for i in range(num_det):
                            # in case the data length in each line is different
                            new_data['det'+str(i+1)][m, :fluor_len, :] = v.data[detector_field][:, i, :]

            # If the detector field does not exist, then try the next one from the list
            if not detector_field_exists:
                continue

            # Modify file name (path) to include data on how many channels are included in the file and how many
            #    channels are used for sum calculation
            root, ext = os.path.splitext(fpath)
            s = f"_{detector_name}_sum({num_det}ch)"
            if create_each_det:
                s += f"+{num_det}ch"
            fpath_out = f'{root}{s}{ext}'

            if vertical_fast is True:  # need to transpose the data, as we scan y first
                if create_each_det is False:
                    new_data['det_sum'] = np.transpose(new_data['det_sum'], axes=(1, 0, 2))
                else:
                    for i in range(num_det):
                        new_data['det'+str(i+1)] = np.transpose(new_data['det'+str(i+1)], axes=(1, 0, 2))

            if save_scalar is True:
                if vertical_fast is False:
                    for i, v in enumerate(scaler_list[:-1]):
                        scaler_tmp[:, :, i] = data[v]
                    scaler_tmp[:, :-1, -1] = np.diff(data['time'], axis=1)
                    scaler_tmp[:, -1, -1] = data['time'][:, -1] - data['time'][:, -2]
                else:
                    for i, v in enumerate(scaler_list[:-1]):
                        scaler_tmp[:, :, i] = data[v].T
                    data_t = data['time'].T
                    scaler_tmp[:-1, :, -1] = np.diff(data_t, axis=0)
                    scaler_tmp[-1, :, -1] = data_t[-1, :] - data_t[-2, :]
                new_data['scaler_data'] = scaler_tmp
                x_pos = np.vstack(data[xpos_name])

                # get y position data, from differet stream name primary
                data1 = hdr.table(fill=True, stream_name='primary')
                if num_end_lines_excluded is not None:
                    data1 = data1[:datashape[0]]
                # if ypos_name not in data1.keys() and 'E_tomo' not in start_doc['scaninfo']['type']:
                # print(f"data1 keys: {data1.keys()}")
                if ypos_name not in data1.keys():
                    ypos_name = 'hf_stage_z'        # vertical along z
                y_pos0 = np.hstack(data1[ypos_name])

                # Original comment (from the previous authors):
                #      y position is more than actual x pos, scan not finished?
                #
                # The following (temporary) fix assumes that incomplete scan contains
                #   at least two completed lines. The step between the scanned lines
                #   may be used to recreate y-coordinates for the lines that were not
                #   scanned: data for those lines will be filled with zeros.
                #   Having some reasonable y-coordinate data for the missed lines
                #   will allow to plot and process the data even if the scan is incomplete.
                #   In the case if scan contain only one line, there is no reliable way
                #   to to generate coordinates, use the same step as for x coordinates
                #   or 1 if the first scannned line contains only one point.

                # First check if the scan of the last line was completed. If not,
                #   then x-coordinates of the scan points are all ZERO
                last_scan_line_no_data = False
                if math.isclose(np.sum(x_pos[x_pos.shape[0] - 1, :]), 0.0, abs_tol=1e-20):
                    last_scan_line_no_data = True

                no_position_data = False
                if len(y_pos0) == 0 or (len(y_pos0) == 1 and last_scan_line_no_data):
                    no_position_data = True
                    print("WARNING: The scan contains no completed scan lines")

                if len(y_pos0) < x_pos.shape[0] and len(y_pos0) > 1:
                    # The number of the lines for which the scan was initiated
                    #   Unfortunately this is not the number of scanned lines,
                    #   so x-axis values need to be restored for the line #'n_scanned_lines - 1' !!!
                    n_scanned_lines = len(y_pos0)
                    print(f"WARNING: The scan is not completed: {n_scanned_lines} "
                          f"out of {x_pos.shape[0]} lines")
                    y_step = 1
                    if n_scanned_lines > 1:
                        y_step = (y_pos0[-1] - y_pos0[0])/(n_scanned_lines - 1)
                    elif x_pos.shape[1] > 1:
                        # Set 'y_step' equal to the absolute value of 'x_step'
                        #    this is just to select some reasonable scale and happens if
                        #    only one line was completed in the unfinished flyscan.
                        #    This is questionable decision, but it should be rarely applied
                        y_step = math.fabs((x_pos[0, -1] - x_pos[0, 0])/(x_pos.shape[1] - 1))
                    # Now use 'y_step' to generate the remaining points
                    n_pts = x_pos.shape[0] - n_scanned_lines
                    v_start = y_pos0[-1] + y_step
                    v_stop = v_start + (n_pts - 1) * y_step
                    y_pos_filled = np.linspace(v_start, v_stop, n_pts)
                    y_pos0 = np.append(y_pos0, y_pos_filled)
                    # Now duplicate x-coordinate values from the last scanned line to
                    #   all the unscanned lines, otherwise they all will be zeros
                    for n in range(n_scanned_lines - 1, x_pos.shape[0]):
                        x_pos[n, :] = x_pos[n_scanned_lines - 2, :]

                elif x_pos.shape[0] > 1 and last_scan_line_no_data:
                    # One possible scenario is that if the scan was interrupted while scanning
                    #   the last line. In this case the condition
                    #                 len(y_pos0) >= x_pos.shape[0]
                    #   will hold, but the last line of x-coordinates will be filleds with
                    #   zeros, which will create a mess if data is plotted with PyXRF
                    #   To fix the problem, fill the last line with values from the previous line
                    x_pos[-1, :] = x_pos[-2, :]

                # The following condition check is left from the existing code. It is still checking
                #   for the case if 0 lines were scanned.
                if len(y_pos0) >= x_pos.shape[0] and not no_position_data:
                    y_pos = y_pos0[:x_pos.shape[0]]
                    x_tmp = np.ones(x_pos.shape[1])
                    xv, yv = np.meshgrid(x_tmp, y_pos)
                    # need to change shape to sth like [2, 100, 100]
                    data_tmp = np.zeros([2, x_pos.shape[0], x_pos.shape[1]])
                    data_tmp[0, :, :] = x_pos
                    data_tmp[1, :, :] = yv
                    new_data['pos_data'] = data_tmp
                    new_data['pos_names'] = ['x_pos', 'y_pos']
                    if vertical_fast is True:  # need to transpose the data, as we scan y first
                        # fast scan on y has impact for scalar data
                        data_tmp = np.zeros([2, x_pos.shape[1], x_pos.shape[0]])
                        data_tmp[1, :, :] = x_pos.T
                        data_tmp[0, :, :] = yv.T
                        new_data['pos_data'] = data_tmp
                else:
                    print("WARNING: Scan was interrupted: x,y positions are not saved")

            n_detectors_found += 1

            if output_to_file:
                # output to file
                print(f"Saving data to hdf file #{n_detectors_found}: Detector: {detector_name}.")
                write_db_to_hdf_base(fpath_out, new_data,
                                     create_each_det=create_each_det)

        print()
        if n_detectors_found == 0:
            print(f"ERROR: no data from known detectors were found in the database:")
            print(f"     Check that appropriate fields are included in 'xrf_fly_scan_detector_fields'")
            print(f"         of configuration file: {config_path}")
        else:
            print(f"Total of {n_detectors_found} detectors were found", end="")
            if output_to_file:
                print(f", {n_detectors_found} data files were created", end="")
            print(".")

        return new_data


def map_data2D_xfm(runid, fpath,
                   create_each_det=False,
                   output_to_file=True):
    """
    Transfer the data from databroker into a correct format following the
    shape of 2D scan.
    This function is used at XFM beamline for step scan.
    Save the new data dictionary to hdf file if needed.

    .. note:: It is recommended to read data from databroker into memory
    directly, instead of saving to files. This is ongoing work.

    Parameters
    ----------
    runid : int
        id number for given run
    fpath: str
        path to save hdf file
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    output_to_file : bool, optional
        save data to hdf5 file if True

    Returns
    -------
    dict of data in 2D format matching x,y scanning positions
    """
    hdr = db[runid]
    # spectrum_len = 4096
    start_doc = hdr['start']
    plan_n = start_doc.get('plan_name')
    if 'fly' not in plan_n:  # not fly scan
        datashape = start_doc['shape']   # vertical first then horizontal
        fly_type = None

        snake_scan = start_doc.get('snaking')
        if snake_scan[1] is True:
            fly_type = 'pyramid'

        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = 'xfm_pv_config.json'
        config_path = sep_v.join(current_dir.split(sep_v)[:-2]+['configs', config_file])
        with open(config_path, 'r') as json_data:
            config_data = json.load(json_data)

        # try except can be added later if scan is not completed.
        data = db.get_table(hdr, fill=True, convert_times=False)

        xrf_detector_names = config_data['xrf_detector']
        data_output = map_data2D(data,
                                 datashape,
                                 det_list=xrf_detector_names,
                                 pos_list=hdr.start.motors,
                                 scaler_list=config_data['scaler_list'],
                                 fly_type=fly_type)
        if output_to_file:
            print('Saving data to hdf file.')
            write_db_to_hdf_base(fpath, data_output,
                                 create_each_det=create_each_det)
        return data_output


def write_db_to_hdf(fpath, data, datashape,
                    det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
                    pos_list=('zpssx[um]', 'zpssy[um]'),
                    scaler_list=('sclr1_ch3', 'sclr1_ch4'),
                    fly_type=None, subscan_dims=None, base_val=None):
    """
    Assume data is obained from databroker, and save the data to hdf file.
    This function can handle stopped/aborted scans.

    .. note:: This function should become part of suitcase

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    """
    interpath = 'xrfmap'
    with h5py.File(fpath, 'a') as f:

        sum_data = None
        new_v_shape = datashape[0]  # to be updated if scan is not completed
        spectrum_len = 4096  # standard

        for n, c_name in enumerate(det_list):
            if c_name in data:
                detname = 'det'+str(n+1)
                dataGrp = f.create_group(interpath+'/'+detname)

                logger.info('read data from %s' % c_name)
                channel_data = data[c_name]

                # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
                new_v_shape = len(channel_data) // datashape[1]

                new_data = np.vstack(channel_data)
                new_data = new_data[:new_v_shape*datashape[1], :]

                new_data = new_data.reshape([new_v_shape, datashape[1],
                                             len(channel_data[1])])
                if new_data.shape[2] != spectrum_len:
                    # merlin detector has spectrum len 2048
                    # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                    new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                    new_tmp[:, :, :new_data.shape[2]] = new_data
                    new_data = new_tmp
                if fly_type in ('pyramid',):
                    new_data = flip_data(new_data, subscan_dims=subscan_dims)

                if sum_data is None:
                    sum_data = new_data
                else:
                    sum_data += new_data
                ds_data = dataGrp.create_dataset('counts', data=new_data, compression='gzip')
                ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)

        # summed data
        dataGrp = f.create_group(interpath+'/detsum')

        if sum_data is not None:
            sum_data = sum_data.reshape([new_v_shape, datashape[1],
                                         spectrum_len])
            ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel sum'

        # position data
        dataGrp = f.create_group(interpath+'/positions')

        pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                     datashape)

        for i in range(len(pos_names)):
            if 'x' in pos_names[i]:
                pos_names[i] = 'x_pos'
            elif 'y' in pos_names[i]:
                pos_names[i] = 'y_pos'
        if 'x_pos' not in pos_names or 'y_pos' not in pos_names:
            pos_names = ['x_pos', 'y_pos']

        # need to change shape to sth like [2, 100, 100]
        data_temp = np.zeros([pos_data.shape[2], pos_data.shape[0], pos_data.shape[1]])
        for i in range(pos_data.shape[2]):
            data_temp[i, :, :] = pos_data[:, :, i]

        if fly_type in ('pyramid',):
            for i in range(data_temp.shape[0]):
                # flip position the same as data flip on det counts
                data_temp[i, :, :] = flip_data(data_temp[i, :, :], subscan_dims=subscan_dims)

        dataGrp.create_dataset('name', data=helper_encode_list(pos_names))
        dataGrp.create_dataset('pos', data=data_temp[:, :new_v_shape, :])

        # scaler data
        dataGrp = f.create_group(interpath+'/scalers')

        scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                           datashape)

        if fly_type in ('pyramid',):
            scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)

        dataGrp.create_dataset('name', data=helper_encode_list(scaler_names))

        if base_val is not None:  # base line shift for detector, for SRX
            base_val = np.array([base_val])
            if len(base_val) == 1:
                scaler_data = np.abs(scaler_data - base_val)
            else:
                for i in scaler_data.shape[2]:
                    scaler_data[:, :, i] = np.abs(scaler_data[:, :, i] - base_val[i])

        dataGrp.create_dataset('val', data=scaler_data[:new_v_shape, :])


def get_name_value_from_db(name_list, data, datashape):
    """
    Get name and data from db.
    """
    pos_names = []
    pos_data = np.zeros([datashape[0], datashape[1], len(name_list)])
    for i, v in enumerate(name_list):
        posv = np.zeros(datashape[0]*datashape[1])  # keep shape unchanged, so stopped/aborted run can be handled.
        data[v] = np.asarray(data[v])  # in case data might be list
        posv[:data[v].shape[0]] = np.asarray(data[v])
        pos_data[:, :, i] = posv.reshape([datashape[0], datashape[1]])
        pos_names.append(str(v))
    return pos_names, pos_data


def map_data2D(data, datashape,
               det_list=('xspress3_ch1', 'xspress3_ch2', 'xspress3_ch3'),
               pos_list=('zpssx[um]', 'zpssy[um]'),
               scaler_list=('sclr1_ch3', 'sclr1_ch4'),
               fly_type=None, subscan_dims=None, spectrum_len=4096):
    """
    Data is obained from databroker. Transfer items from data to a dictionay of
    numpy array, which has 2D shape same as scanning area.

    This function can handle stopped/aborted scans. Raster scan (snake scan) is
    also considered.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple, optional
        list of detector channels
    pos_list : list, tuple, optional
        list of pos pv
    scaler_list : list, tuple, optional
        list of scaler pv
    fly_type : string or optional
        raster scan (snake scan) or normal
    subscan_dims : 1D array or optional
        used at HXN, 2D of a large area is split into small area scans
    spectrum_len : int, optional
        standard spectrum length

    Returns
    -------
    dict of numpy array
    """
    data_output = {}
    sum_data = None
    new_v_shape = datashape[0]  # updated if scan is not completed
    sum_data = None

    for n, c_name in enumerate(det_list):
        if c_name in data:
            detname = 'det'+str(n+1)
            logger.info('read data from %s' % c_name)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]
            new_data = np.vstack(channel_data)
            new_data = new_data[:new_v_shape*datashape[1], :]
            new_data = new_data.reshape([new_v_shape, datashape[1],
                                         len(channel_data[1])])
            if new_data.shape[2] != spectrum_len:
                # merlin detector has spectrum len 2048
                # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                new_tmp[:, :, :new_data.shape[2]] = new_data
                new_data = new_tmp
            if fly_type in ('pyramid',):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)
            data_output[detname] = new_data
            if sum_data is None:
                sum_data = new_data
            else:
                sum_data += new_data
    data_output['det_sum'] = sum_data

    # scanning position data
    pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                 datashape)
    for i in range(len(pos_names)):
        if 'x' in pos_names[i]:
            pos_names[i] = 'x_pos'
        elif 'y' in pos_names[i]:
            pos_names[i] = 'y_pos'
    if 'x_pos' not in pos_names or 'y_pos' not in pos_names:
        pos_names = ['x_pos', 'y_pos']

    if fly_type in ('pyramid',):
        for i in range(pos_data.shape[2]):
            # flip position the same as data flip on det counts
            pos_data[:, :, i] = flip_data(pos_data[:, :, i], subscan_dims=subscan_dims)
    new_p = np.zeros([len(pos_names), pos_data.shape[0], pos_data.shape[1]])
    for i in range(len(pos_names)):
        new_p[i, :, :] = pos_data[:, :, i]
    data_output['pos_names'] = pos_names
    data_output['pos_data'] = new_p

    # scaler data
    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                       datashape)
    if fly_type in ('pyramid',):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)
    data_output['scaler_names'] = scaler_names
    data_output['scaler_data'] = scaler_data
    return data_output


def write_db_to_hdf_base(fpath, data, create_each_det=True):
    """
    Data is obained based on databroker, and save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : dict
        fluorescence data with scaler value and positions
    create_each_det : Bool, optional
        if number of point is too large, only sum data is saved in h5 file
    """
    interpath = 'xrfmap'
    sum_data = None
    xrf_det_list = [n for n in data.keys() if 'det' in n and 'sum' not in n]
    xrf_det_list.sort()

    with h5py.File(fpath, 'a') as f:
        if create_each_det is True:
            for detname in xrf_det_list:
                new_data = data[detname]

                if sum_data is None:
                    sum_data = new_data
                else:
                    sum_data += new_data

                dataGrp = f.create_group(interpath+'/'+detname)
                ds_data = dataGrp.create_dataset('counts', data=new_data,
                                                 compression='gzip')
                ds_data.attrs['comments'] = 'Experimental data from {}'.format(detname)
        else:
            sum_data = data['det_sum']

        # summed data
        if sum_data is not None:
            dataGrp = f.create_group(interpath+'/detsum')
            ds_data = dataGrp.create_dataset('counts', data=sum_data,
                                             compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel sum'

        # add positions
        if 'pos_names' in data:
            dataGrp = f.create_group(interpath+'/positions')
            pos_names = data['pos_names']
            pos_data = data['pos_data']
            dataGrp.create_dataset('name', data=helper_encode_list(pos_names))
            dataGrp.create_dataset('pos', data=pos_data)

        # scaler data
        if 'scaler_data' in data:
            dataGrp = f.create_group(interpath+'/scalers')
            scaler_names = data['scaler_names']
            scaler_data = data['scaler_data']
            dataGrp.create_dataset('name', data=helper_encode_list(scaler_names))
            dataGrp.create_dataset('val', data=scaler_data)


def free_memory_from_handler():
    """Quick way to set 3D dataset at handler to None to release memory.
    """
    for h in db.fs._handler_cache.values():
        setattr(h, '_dataset', None)
    print('Memory is released.')


def export1d(runid, name=None):
    """
    Export all PVs to a file. Do not talk to filestore.

    Parameters
    ----------
    name : str or optional
        name for the file
    runid : int
        run number
    """
    t = db.get_table(db[runid], fill=False)
    if name is None:
        name = 'scan_'+str(runid)+'.txt'
    t.to_csv(name)


def helper_encode_list(data, data_type='utf-8'):
    return [d.encode(data_type) for d in data]


def helper_decode_list(data, data_type='utf-8'):
    return [d.decode(data_type) for d in data]


def get_data_per_event(n, data, e, det_num):
    db.fill_event(e)
    min_len = e.data['fluor'].shape[0]
    for i in range(det_num):
        data[n, :min_len, :] += e.data['fluor'][:, i, :]


def get_data_parallel(data, elist, det_num):
    num_processors_to_use = multiprocessing.cpu_count()-2

    print('cpu count: {}'.format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    # result_pool = [
    #     pool.apply_async(get_data_per_event, (n, data, e, det_num))
    #     for n, e in enumerate(elist)]

    # results = [r.get() for r in result_pool]

    pool.terminate()
    pool.join()
