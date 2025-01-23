from libc.math cimport log, exp

cdef inline float photosynth_active_radiation(float sw_rad) nogil:
    '''
    Calculates daily total photosynthetically active radiation (PAR) from
    (hourly) incoming short-wave radiation (`sw_rad`). PAR is assumed to
    be 45% of `sw_rad`.

    I make a note here, because this is one place someone would come back to
    when looking for this information: `sw_rad` is a power (energy per unit
    time), so when working with sub-daily source data, we don't take, e.g., a
    24-hour sum but a 24-hour mean. An alternative approach might be to
    convert the hourly data to energy (Joules) first, but that is not what has
    historically been done. This was confirmed by comparing an official
    24-hour MERRA-2 granule with the (apparently averaged) MERRA-2 data used
    previously in L4C V4, as listed here:

        /anx_v2/laj/smap/code/geog2egv2/list/merra2_gran_swgdn.list

    Parameters
    ----------
    sw_rad : int or float or numpy.ndarray
        Incoming short-wave radiation (W m-2)

    NOTE: Assumes that the period over which radiation is measured, in hours,
        is 1 (i.e., once-hourly measurements).

    Returns
    -------
    int or float or numpy.ndarray
        Photosynthetically active radiation (MJ m-2)
    '''
    # Convert SW_rad from [W m-2] to [MJ m-2], then take 45%; because
    #   1 W == 1 J s-1, we multiply 3600 secs hr-1 times
    #   (1 MJ / 1e6 Joules) == 0.0036
    return 0.45 * (0.0036 * 24 * sw_rad)


cdef inline float rescale_smrz(
        float smrz0, float smrz_min, float smrz_max) nogil:
    '''
    Rescales root-zone soil-moisture (SMRZ); original SMRZ is in percent
    saturation units. NOTE: Although Jones et al. (2017) write "SMRZ_wp is
    the plant wilting point moisture level determined by ancillary soil
    texture data provided by L4SM..." in actuality it is just `smrz_min`.

    NOTE: This operation must be done element-wise for array pointers, e.g.:

        for i in prange(N, nogil = True):
            rescale_smrz(...)

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


cdef inline float vapor_pressure_deficit(
        float qv2m, float ps, float temp_k) nogil:
    r'''
    Calculates vapor pressure deficit (VPD); unfortunately, the provenance
    of this formula cannot be properly attributed. It is taken from the
    SMAP L4C Science code base, so it is exactly how L4C calculates VPD.

    $$
    \mathrm{VPD} = 610.7 \times \mathrm{exp}\left(
    \frac{17.38 \times T_C}{239 + T_C}
    \right) - \frac{(P \times [\mathrm{QV2M}]}{0.622 + (0.378 \times [\mathrm{QV2M}])}
    $$

    Where P is the surface pressure (Pa), QV2M is the water vapor mixing
    ratio at 2-meter height, and T is the temperature in degrees C (though
    this function requires units of Kelvin when called).

    NOTE: A variation on this formula can be found in the text:

    Monteith, J. L. and M. H. Unsworth. 1990.
    Principles of Environmental Physics, 2nd. Ed. Edward Arnold Publisher.

    See also:
        https://glossary.ametsoc.org/wiki/Mixing_ratio

    NOTE: This operation must be done element-wise for array pointers, e.g.:

        for i in prange(N, nogil = True):
            vapor_pressure_deficit(...)

    Parameters
    ----------
    qv2m : float
        QV2M, the water vapor mixing ratio at 2-m height
    ps : float
        The surface pressure, in Pascals
    temp_k : float
        The temperature at 2-m height in degrees Kelvin

    Returns
    -------
    float
        VPD in Pascals
    '''
    temp_c = temp_k - 273.15 # Convert temperature to degrees C
    avp = (qv2m * ps) / (0.622 + (0.378 * qv2m))
    x = (17.38 * temp_c) / (239 + temp_c)
    esat = 610.7 * exp(x)
    return esat - avp
