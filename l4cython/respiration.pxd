cdef struct BPLUT:
    float smsf0[9] # wetness [0-100%]
    float smsf1[9] # wetness [0-100%]
    float tsoil[9] # deg K
    float cue[9]
    float f_metabolic[9]
    float f_structural[9]
    float decay_rate[9]


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
