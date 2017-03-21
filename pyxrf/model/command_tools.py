# basic command line functions to use. Some of those functions can be implemented to
# scikit-beam later.
import os
import time
import json
import glob

from .fileio import output_data, read_hdf_APS, read_MAPS
from .fit_spectrum import single_pixel_fitting_controller, save_fitdata_to_hdf


def fit_pixel_data_and_save(working_directory, file_name,
                            fit_channel_sum=True, param_file_name=None,
                            fit_channel_each=False, param_channel_list=None,
                            incident_energy=None,
                            method='nnls', pixel_bin=0, raise_bg=0,
                            comp_elastic_combine=False,
                            linear_bg=False,
                            use_snip=True,
                            bin_energy=0,
                            spectrum_cut=3000,
                            save_txt=False,
                            save_tiff=True,
                            ic_name=None,
                            data_from='NSLS-II'):
    """
    Do fitting for signle data set, and save data accordingly. Fitting can be performed on
    either summed data or each channel data, or both.

    Parameters
    ----------
    working_directory : str
        path folder
    file_names : str
        selected h5 file
    fit_channel_sum : bool, optional
        fit summed data or not
    param_file_name : str, optional
        param file name for summed data fitting
    fit_channel_each : bool, optional
        fit each channel data or not
    param_channel_list : list, optional
        list of param file names for each channel
    incident_energy : float, optional
        use this energy as incident energy instead of the one in param file, i.e., XANES
    method : str, optional
        fitting method, default as nnls
    pixel_bin : int, optional
        bin pixel as 2by2, or 3by3
    raise_bg : int, optional
        add a constant value to each spectrum, better for fitting
    comp_elastic_combine : bool, optional
        combine elastic and compton as one component for fitting
    linear_bg : bool, optional
        use linear background instead of snip
    use_snip : bool, optional
        use snip method to remove background
    bin_energy : int, optional
        bin spectrum with given value
    spectrum_cut : int, optional
        only use spectrum from, say 0, 3000
    save_txt : bool, optional
        save data to txt or not
    save_tiff : bool, optional
        save data to tiff or not
    ic_name : str, optional
        if given, normalization will be performed
    data_from : str, optional
        where do data come from? Data format includes data from NSLS-II, or 2IDE-APS
    """
    fpath = os.path.join(working_directory, file_name)
    t0 = time.time()
    prefix_fname = file_name.split('.')[0]
    if fit_channel_sum is True:
        if data_from == 'NSLS-II':
            img_dict, data_sets = read_hdf_APS(working_directory, file_name,
                                               spectrum_cut=spectrum_cut,
                                               load_each_channel=False)
        elif data_from == '2IDE-APS':
            img_dict, data_sets = read_MAPS(working_directory,
                                            file_name, channel_num=1)
        else:
            print('Unkonw data sets.')

        try:
            data_all_sum = data_sets[prefix_fname+'_sum'].raw_data
        except KeyError:
            data_all_sum = data_sets[prefix_fname].raw_data

        # load param file
        param_path = os.path.join(working_directory, param_file_name)
        with open(param_path, 'r') as json_data:
            param_sum = json.load(json_data)

        # update incident energy, required for XANES
        if incident_energy is not None:
            param_sum['coherent_sct_energy']['value'] = incident_energy

        result_map_sum, calculation_info = single_pixel_fitting_controller(data_all_sum,
                                                                           param_sum,
                                                                           incident_energy=incident_energy,
                                                                           method=method,
                                                                           pixel_bin=pixel_bin,
                                                                           raise_bg=raise_bg,
                                                                           comp_elastic_combine=comp_elastic_combine,
                                                                           linear_bg=linear_bg,
                                                                           use_snip=use_snip,
                                                                           bin_energy=bin_energy)

        # output to .h5 file
        inner_path = 'xrfmap/detsum'
        fit_name = prefix_fname+'_fit'
        save_fitdata_to_hdf(fpath, result_map_sum, datapath=inner_path)

    if fit_channel_each is True and param_channel_list is not None:
        channel_num = len(param_channel_list)
        img_dict, data_sets = read_hdf_APS(working_directory, file_name,
                                           spectrum_cut=spectrum_cut,
                                           load_each_channel=True)
        for i in range(channel_num):
            filename_det = prefix_fname+'_det'+str(i+1)
            inner_path = 'xrfmap/det'+str(i+1)

            # load param file
            param_file_det = param_channel_list[i]
            param_path = os.path.join(working_directory, param_file_det)
            with open(param_path, 'r') as json_data:
                param_det = json.load(json_data)

            # update incident energy, required for XANES
            if incident_energy is not None:
                param_det['coherent_sct_energy']['value'] = incident_energy

            data_all_det = data_sets[filename_det].raw_data
            result_map_det, calculation_info = single_pixel_fitting_controller(data_all_det,
                                                                               param_det,
                                                                               method=method,
                                                                               pixel_bin=pixel_bin,
                                                                               raise_bg=raise_bg,
                                                                               comp_elastic_combine=comp_elastic_combine,
                                                                               linear_bg=linear_bg,
                                                                               use_snip=use_snip,
                                                                               bin_energy=bin_energy)
            # output to .h5 file
            save_fitdata_to_hdf(fpath, result_map_det, datapath=inner_path)

    t1 = time.time()
    print('Time used for pixel fitting for file {} is : {}'.format(file_name, t1-t0))

    if save_txt is True:
        output_folder = 'output_txt_'+prefix_fname
        output_path = os.path.join(working_directory, output_folder)
        output_data(fpath, output_path, file_format='txt', norm_name=ic_name)
    if save_tiff is True:
        output_folder = 'output_tiff_'+prefix_fname
        output_path = os.path.join(working_directory, output_folder)
        output_data(fpath, output_path, file_format='tiff', norm_name=ic_name)


def fit_multiple_pixel_data(start_id, end_id=None, wd=None, fit_channel_sum=True, param_file_name=None,
                            fit_channel_each=False, param_channel_list=None, incident_energy=None,
                            spectrum_cut=3000, save_txt=False, save_tiff=True, ic_name=None):
    """
    Do fitting for multiple data sets, and save data accordingly. Fitting can be performed on
    either summed data or each channel data, or both. This is based on fit_pixel_data_and_save function.

    Parameters
    ----------
    start_id : int
        starting run id
    end_id : int
        ending run id
    wd : str, or optional
        path folder, default is the current folder
    file_names : str
        selected h5 file
    fit_channel_sum : bool, optional
        fit summed data or not
    param_file_name : str, optional
        param file name for summed data fitting
    fit_channel_each : bool, optional
        fit each channel data or not
    param_channel_list : list, optional
        list of param file names for each channel
    incident_energy : float, optional
        use this energy as incident energy instead of the one in param file, i.e., XANES
    spectrum_cut : int, optional
        only use spectrum from, say 0, 3000
    save_txt : bool, optional
        save data to txt or not
    save_tiff : bool, optional
        save data to tiff or not
    ic_name : str, optional
        if given, normalization will be performed
    """
    if wd is None:
        wd = '.'
    all_files = glob.glob(os.path.join(wd, '*.h5'))

    if end_id is None:
        flist = [fname for fname in all_files if str(start_id) in fname]
        try:
            fpath = flist[0]
        except IndexError:
            print("File with runid {} doesn't exist.".format(start_id))
        fname = fpath.split('/')[-1]
        working_directory = fpath[:-len(fname)]
        fit_pixel_data_and_save(working_directory, fname, fit_channel_sum=fit_channel_sum, param_file_name=param_file_name,
                                fit_channel_each=fit_channel_each, param_channel_list=param_channel_list,
                                incident_energy=incident_energy, spectrum_cut=spectrum_cut,
                                save_txt=save_txt, save_tiff=save_tiff,
                                ic_name=ic_name)
    else:
        flist = []
        for data_id in range(start_id, end_id+1):
            flist += [fname for fname in all_files if str(data_id) in fname]
        print('Number of files to fit: {}'.format(len(flist)))
        print('\n'.join(flist))
        for fpath in flist:
            fname = fpath.split('/')[-1]
            working_directory = fpath[:-len(fname)]
            fit_pixel_data_and_save(working_directory, fname, fit_channel_sum=fit_channel_sum, param_file_name=param_file_name,
                                    fit_channel_each=fit_channel_each, param_channel_list=param_channel_list,
                                    incident_energy=incident_energy, spectrum_cut=spectrum_cut,
                                    save_txt=save_txt, save_tiff=save_tiff,
                                    ic_name=ic_name)
