# cython: language_level=3

import csv
import numpy as np
from collections import OrderedDict

def load_parameters_table(csv_file_path: bytes):
    '''
    Translates a BPLUT CSV file to a Python internal representation
    (OrderedDict instance). Keys are ordered as:

    Parameters
    ----------
    csv_file_path : bytes
        File path to the CSV representation of the BPLUT

    Returns
    -------
    dict
    '''
    def decomment(csv_file):
        for row in csv_file:
            raw = row.split('#')[0].strip()
            if raw:
                yield raw

    header = ('LC_index', 'LC_Label', 'model_code', 'NDVItoFPAR_scale',
        'NDVItoFPAR_offset', 'LUEmax', 'Tmin_min_K', 'Tmin_max_K',
        'VPD_min_Pa', 'VPD_max_Pa', 'SMrz_min', 'SMrz_max', 'FT_min',
        'FT_max', 'SMtop_min', 'SMtop_max', 'Tsoil_beta0', 'Tsoil_beta1',
        'Tsoil_beta2', 'fraut', 'fmet', 'fstr', 'kopt', 'kstr', 'kslw',
        'Nee_QA_Rank_min', 'Nee_QA_Rank_max', 'Nee_QA_Error_min',
        'Nee_QA_Error_max', 'Fpar_QA_Rank_min', 'Fpar_QA_Rank_max',
        'Fpar_QA_Error_min', 'Fpar_QA_Error_max', 'FtMethod_QA_mult',
        'FtAge_QA_Rank_min', 'FtAge_QA_Rank_max', 'FtAge_QA_Error_min',
        'FtAge_QA_Error_max', 'Par_QA_Error', 'Tmin_QA_Error',
        'Vpd_QA_Error', 'Smrz_QA_Error', 'Tsoil_QA_Error', 'Smtop_QA_Error')
    contents = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(decomment(file), fieldnames = header)
        for row in reader:
            contents.append(row)

    # With the horrible CSV files written by L4CSYS, it's sometimes unclear
    #   on which line we should start; this sometimes manifests as an empty
    #   header row with None for values
    if None in contents[0].values():
        contents = contents[1:]

    params = OrderedDict()
    params['LUE'] = np.array([[
        contents[p-1]['LUEmax'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(4)
    params['CUE'] = np.array([[
        (1 - float(contents[p-1]['fraut'])) if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(4)
    params['tmin'] = np.array([
        [contents[p-1]['Tmin_min_K'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['Tmin_max_K'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    params['vpd'] = np.array([
        [contents[p-1]['VPD_min_Pa'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['VPD_max_Pa'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(2)
    params['smrz'] = np.array([
        [contents[p-1]['SMrz_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['SMrz_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    params['smsf'] = np.array([
        [contents[p-1]['SMtop_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['SMtop_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(1)
    params['ft'] = np.array([
        [contents[p-1]['FT_min'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['FT_max'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(3)
    params['tsoil'] = np.array([[
        contents[p-1]['Tsoil_beta0'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(1)
    params['f_metabolic'] = np.array([[
        contents[p-1]['fmet'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(3)
    params['f_structural'] = np.array([[
        contents[p-1]['fstr'] if p in range(1, 9) else np.nan
        for p in range(0, 10)
    ]], dtype = np.float32).round(1)
    params['decay_rate'] = np.array([
        [contents[p-1]['kopt'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['kstr'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
        [contents[p-1]['kslw'] if p in range(1, 9) else np.nan
            for p in range(0, 10)],
    ], dtype = np.float32).round(3)
    # The "kstr" and "kslw" values are really the fraction of kopt
    #   assigned to the second and third pools
    params['decay_rate'][1,:] = np.multiply(
        params['decay_rate'][0,:], params['decay_rate'][1,:])
    params['decay_rate'][2,:] = np.multiply(
        params['decay_rate'][0,:], params['decay_rate'][2,:])
    # Now flatten; i.e., this implements pyl4c's restore_bplut_flat()
    result = OrderedDict()
    for key, value in params.items():
        if key not in ('tmin', 'vpd', 'smsf', 'smrz', 'ft'):
            result[key] = value
            continue
        for i, array in enumerate(value.tolist()):
            result[f'{key}{i}'] = np.array(array).reshape((1,len(array)))
    return result
