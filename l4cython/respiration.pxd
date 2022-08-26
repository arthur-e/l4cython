cdef struct BPLUT:
    float smsf0[9] # wetness [0-100%]
    float smsf1[9] # wetness [0-100%]
    float tsoil[9] # deg K
    float cue[9]
    float f_metabolic[9]
    float f_structural[9]
    float decay_rate[9]


# Represents the RH flux from three SOC pools
cdef struct rh_flux:
    float rh0
    float rh1
    float rh2


cdef inline rh_flux rh_calc(
        BPLUT params, int pft, float k_mult, float soc0, float soc1, float soc2,
        float kstruct, float krecal):
    '''
    Pass
    '''
    cdef rh_flux rh
    rh.rh0 = k_mult * soc0 * params.decay_rate[pft]
    rh.rh1 = k_mult * soc1 * params.decay_rate[pft] * kstruct
    rh.rh2 = k_mult * soc2 * params.decay_rate[pft] * krecal
    # "the adjustment...to account for material transferred into the
    #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
    rh.rh1 = rh.rh1 * (1 - params.f_structural[pft])
    return rh


cdef inline float linear_constraint(
        float x, float xmin, float xmax, int reversed):
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    x: float
    xmin : float
        Lower bound of the linear ramp function
    xmax : float
        Upper bound of the linear ramp function
    reversed : int
        Type of ramp function: 1 for "reversed," i.e., function decreases
        as x increases

    Returns
    -------
    float
    '''
    assert reversed == 0 or reversed == 1
    if reversed == 1:
        if x >= xmax:
            return 0
        elif x < xmin:
            return 1
        else:
            return 1 - ((x - xmin) / (xmax - xmin))
    # For normal case
    if x >= xmax:
        return 1
    elif x < xmin:
        return 0
    else:
        return (x - xmin) / (xmax - xmin)
