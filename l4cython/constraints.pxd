import numpy as np
from libc.math cimport exp

cdef inline float arrhenius(
        float tsoil, float beta0, float beta1, float beta2) nogil:
    '''
    The Arrhenius equation for response of enzymes to (soil) temperature,
    constrained to lie on the closed interval [0, 1].

    Parameters
    ----------
    tsoil : float
        Array of soil temperature in degrees K
    beta0 : float
        Coefficient for soil temperature (deg K)
    beta1 : float
        Coefficient for ... (deg K)
    beta2 : float
        Coefficient for ... (deg K)
    '''
    cdef float a, b, y0
    a = (1.0 / beta1)
    b = 1 / (tsoil - beta2)
    # This is the simple answer, but it takes on values >1
    y0 = exp(beta0 * (a - b))
    # Constrain the output to the interval [0, 1]
    if y0 > 1:
        return 1
    elif y0 < 0:
        return 0
    else:
        return y0


cdef inline float linear_constraint(
        float x, float xmin, float xmax, int reversed) nogil:
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


cdef inline char is_valid(char pft, float tsoil, float litter) nogil:
    '''
    Checks to see if a given pixel is valid, based on the PFT but also on
    select input data values. Modeled after `tcfModUtil_isInCell()` in the
    TCF code.

    Parameters
    ----------
    pft : char
        The Plant Functional Type (PFT)
    tsoil : float
        The daily mean soil temperature in the surface layer (deg K)
    litter : float
        The daily litterfall input

    Returns
    -------
    char
        A value of 0 indicates the pixel is invalid, otherwise returns 1
    '''
    cdef char valid = 1 # Assume it's a valid pixel
    if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
        valid = 0
    elif tsoil <= 0:
        valid = 0
    elif litter <= 0:
        valid = 0
    return valid
