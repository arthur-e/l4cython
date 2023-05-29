# cython: language_level=3

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based on
Version 7 state and parameters, at 9-km spatial resolution. The `main()`
routine is optimized for model execution but it may take several seconds to
load the state data.

Required data:

- Surface soil wetness ("SMSF"), in proportion units [0,1]
- Soil temperature, in degrees K
'''

import cython
import datetime
import json
import numpy as np
from tqdm import tqdm
from respiration cimport BPLUT, arrhenius, linear_constraint

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

# The PFT map
cdef unsigned char PFT[SPARSE_N]
cdef:
    float SOC0[SPARSE_N]
    float SOC1[SPARSE_N]
    float SOC2[SPARSE_N]
    float NPP[SPARSE_N]

# L4_C BPLUT Version 7 (Vv7042, Vv7040, Nature Run v10)
# NOTE: BPLUT is initialized here because we *need* it to be a C struct and
#   1) It cannot be a C struct if it is imported from a *.pyx file (it gets
#   converted to a dict); 2) If imported as a Python dictionary and coerced
#   to a BPLUT struct, it's still (inexplicably) a dictionary; and 3) We can't
#   initalize the C struct's state if it is in a *.pxd file
cdef BPLUT PARAMS
# NOTE: Must have an (arbitrary) value in 0th position to avoid overflow of
#   indexing (as PFT=0 is not used and C starts counting at 0)
PARAMS.smsf0[:] = [0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 11.3, 0.0]
PARAMS.smsf1[:] = [0, 30.1, 30.1, 35.1, 30.7, 75.4, 68.0, 30.1, 30.1]
PARAMS.tsoil[:] = [0, 238.17, 422.77, 233.94, 246.48, 154.91, 366.14, 242.47, 265.06]
PARAMS.cue[:] = [0, 0.687, 0.469, 0.755, 0.799, 0.649, 0.572, 0.708, 0.705]
PARAMS.f_metabolic[:] = [0, 0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78]
PARAMS.f_structural[:] = [0, 0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8]
PARAMS.decay_rate[0] = [0, 0.020, 0.022, 0.030, 0.029, 0.012, 0.026, 0.018, 0.031]
for p in range(1, 9):
    PARAMS.decay_rate[1][p] = PARAMS.decay_rate[0][p] * KSTRUCT
    PARAMS.decay_rate[2][p] = PARAMS.decay_rate[0][p] * KRECAL


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config_file = None):
    '''
    Forward run of the L4C soil decomposition and heterotrophic respiration
    algorithm. Starts on March 31, 2015 and continues for the specified
    number of time steps.
    '''
    cdef:
        Py_ssize_t i
        float rh0[SPARSE_N]
        float rh1[SPARSE_N]
        float rh2[SPARSE_N]
        float w_mult[SPARSE_N]
        float t_mult[SPARSE_N]
    # Read in configuration file, then load state data
    if config_file is None:
        config_file = '../data/L4Cython_RECO_M09_config.json'
    with open(config_file) as file:
        config = json.load(file)
    load_state(config)
    # We leave rh_total as a NumPy array because it is one we want to
    #   write to disk
    rh_total = np.full((SPARSE_N,), np.nan, dtype = np.float32)
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date = date.strftime('%Y%m%d')
        # Convert to percentage units
        smsf = 100 * np.fromfile(
            config['data']['drivers']['smsf'] % date, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['drivers']['tsoil'] % date, dtype = np.float32)
        for i in range(0, SPARSE_N):
            pft = int(PFT[i])
            if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                continue
            w_mult[i] = linear_constraint(
                smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
            t_mult[i] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
            rh0[i] = w_mult[i] * t_mult[i] * SOC0[i] * PARAMS.decay_rate[0][pft]
            rh1[i] = w_mult[i] * t_mult[i] * SOC1[i] * PARAMS.decay_rate[1][pft]
            rh2[i] = w_mult[i] * t_mult[i] * SOC2[i] * PARAMS.decay_rate[2][pft]
            # "the adjustment...to account for material transferred into the
            #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
            rh1[i] = rh1[i] * (1 - PARAMS.f_structural[pft])
            rh_total[i] = rh0[i] + rh1[i] + rh2[i]
            # Calculate change in SOC pools; NPP[i] is daily litterfall
            SOC0[i] += (NPP[i] * PARAMS.f_metabolic[pft]) - rh0[i]
            SOC1[i] += (NPP[i] * (1 - PARAMS.f_metabolic[pft])) - rh1[i]
            SOC2[i] += (PARAMS.f_structural[pft] * rh1[i]) - rh2[i]
        # NOTE: Writing more than one array per iteration of this loop will
        #   cause a segmenation fault
        np.array(rh_total).astype(np.float32).tofile(
            '%s/L4Cython_RH_%s_M09land.flt32' % (config['model']['output_dir'], date))


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    PFT[:] = np.fromfile(config['data']['PFT_map'], np.uint8)
    SOC0[:] = np.fromfile(config['data']['SOC'][0], np.float32)
    SOC1[:] = np.fromfile(config['data']['SOC'][1], np.float32)
    SOC2[:] = np.fromfile(config['data']['SOC'][2], np.float32)
    NPP[:] = np.fromfile(config['data']['NPP_annual_sum'], np.float32) / 365
