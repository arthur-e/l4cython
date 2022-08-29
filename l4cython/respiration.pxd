import numpy as np

cdef extern from "math.h":
    double exp(double x)


cdef struct BPLUT:
    float smsf0[9] # wetness [0-100%]
    float smsf1[9] # wetness [0-100%]
    float tsoil[9] # deg K
    float cue[9]
    float f_metabolic[9]
    float f_structural[9]
    float decay_rate[3][9]


cdef inline float arrhenius(
        float tsoil, float beta0, float beta1, float beta2):
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


cdef inline to_numpy(float *ptr, int n):
    '''
    Converts a typed memoryview to a NumPy array.

    Parameters
    ----------
    ptr : float*
        A pointer to the typed memoryview
    n : int
        The number of array elements

    Returns
    -------
    numpy.ndarray
    '''
    cdef int i
    arr = np.full((n,), np.nan, dtype = np.float32)
    for i in range(n):
        arr[i] = ptr[i]
    return arr
