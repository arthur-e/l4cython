from libc.math cimport log

cdef inline float rescale_smrz(
        float smrz0, float smrz_min, float smrz_max) nogil:
    '''
    Rescales root-zone soil-moisture (SMRZ); original SMRZ is in percent
    saturation units. NOTE: Although Jones et al. (2017) write "SMRZ_wp is
    the plant wilting point moisture level determined by ancillary soil
    texture data provided by L4SM..." in actuality it is just `smrz_min`.

    Parameters
    ----------
    smrz0 : float
        (T x N) array of original SMRZ data, in percent (%) saturation units
        for N sites and T time steps
    smrz_min : float
        Site-level long-term minimum SMRZ (percent saturation)
    smrz_max : float
        Site-level long-term maximum SMRZ (percent saturation)
    '''
    cdef float smrz_norm
    # Clip input SMRZ to the lower, upper bounds
    if smrz0 < smrz_min:
        smrz0 = smrz_min
    elif smrz0 > smrz_max:
        smrz0 = smrz_max
    smrz_norm = 100 * ((smrz0 - smrz_min) / (smrz_max - smrz_min)) + 1
    # Log-transform normalized data and rescale to range between
    #   5.0 and 100% saturation)
    return 95 * (log(smrz_norm) / log(101)) + 5
