"""
Functions to implement various transforms of complex data
"""
import numpy as np


def complex_to_real_2d(arr: np.ndarray) -> np.ndarray:
    """
    Convert complex array into 2-D real array

    The real parts of the numbers are in the first column, the complex in the
    second.

    Parameters
    ----------
    arr : np.ndarray
        The complex array

    Returns
    -------
    np.ndarray
        The real array

    Examples
    --------
    Convert a 1-D complex array into a 2-D real array with real numbers in the
    first column and imaginary numbers in the second.

    >>> import numpy as np
    >>> from regressioninc.transform import complex_to_real_2d
    >>> arr = np.array([3 + 5j, 2 - 1j, 6 + 7j])
    >>> complex_to_real_2d(arr)
    array([[ 3.,  5.],
           [ 2., -1.],
           [ 6.,  7.]])

    See also
    --------
    real_2d_to_complex : convert 2-D real-valued array to a complex-valued array
    """
    return np.column_stack((arr.real, arr.imag))


def complex_to_real_view(arr: np.ndarray) -> np.ndarray:
    """
    Get a 2-D view of a complex array

    This returns a 2-D view of a complex array with real numbers in the first
    column and complex in the second column. Note that changes made in the view
    will also be reflected in the original array

    Parameters
    ----------
    arr : np.ndarray
        The complex array

    Returns
    -------
    np.ndarray
        The real array

    Examples
    --------
    Convert a 1-D complex array into a 2-D real array with real numbers in the
    first column and imaginary numbers in the second.

    >>> import numpy as np
    >>> from regressioninc.transform import complex_to_real_view
    >>> arr = np.array([3 + 5j, 2 - 1j, 6 + 7j])
    >>> view = complex_to_real_view(arr)
    >>> view
    array([[ 3.,  5.],
           [ 2., -1.],
           [ 6.,  7.]])
    >>> view[0,0] = 9
    >>> view
    array([[ 9.,  5.],
           [ 2., -1.],
           [ 6.,  7.]])
    >>> arr
    array([9.+5.j, 2.-1.j, 6.+7.j])

    See also
    --------
    real_view_to_complex : get a complex-valued view of a 2-D real-valued array
    """
    return arr.view("(2,)float")


def real_2d_to_complex(arr: np.ndarray) -> np.ndarray:
    """
    Convert a real-valued 2-D array to complex values

    Parameters
    ----------
    arr : np.ndarray
        2-D array with real values

    Returns
    -------
    np.ndarray
        Array converted to have complex values

    Examples
    --------
    Start by converting a complex-valued array to a 2-D real-valued array

    >>> import numpy as np
    >>> from regressioninc.transform import complex_to_real_2d, real_2d_to_complex
    >>> arr = np.array([3 + 5j, 2 - 1j, 6 + 7j])
    >>> real2d = complex_to_real_2d(arr)
    >>> real2d
    array([[ 3.,  5.],
           [ 2., -1.],
           [ 6.,  7.]])

    Now let's convert this back

    >>> real_2d_to_complex(real2d)
    array([3.+5.j, 2.-1.j, 6.+7.j])

    See also
    --------
    complex_to_real_2d : convert complex-valued array to 2-D real-valued array
    """
    return arr[..., 0] + 1j * arr[..., 1]


def real_view_to_complex(arr: np.ndarray) -> np.ndarray:
    """
    Get a complex-valued view of a 2-D real array

    Parameters
    ----------
    arr : np.ndarray
        2-D array with real values

    Returns
    -------
    np.ndarray
        Complex-valued view of the real-valued array

    Examples
    --------
    Start by converting a complex-valued array to a 2-D real-valued array

    >>> import numpy as np
    >>> from regressioninc.transform import complex_to_real_2d, real_view_to_complex
    >>> arr = np.array([3 + 5j, 2 - 1j, 6 + 7j])
    >>> real2d = complex_to_real_2d(arr)
    >>> real2d
    array([[ 3.,  5.],
           [ 2., -1.],
           [ 6.,  7.]])

    Now let's get a complex-valued view

    >>> real_view_to_complex(real2d)
    array([[3.+5.j],
           [2.-1.j],
           [6.+7.j]])

    See also
    --------
    complex_to_real_view : get a 2-D real-valued view of a complex-valued array
    """
    return arr.view(dtype=np.complex128)
