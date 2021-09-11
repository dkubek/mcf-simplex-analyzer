from gmpy2 import mpq
import numpy as np
from mcf_simplex_analyzer.fractionarray._fractionarray import FractionArray


def empty(shape, dtype=mpq):
    """
    Return a new FractionArray of given shape and type, without initializing
    entries.

    Args:
        shape (tuple of ints, int): Shape of the new array.
        dtype (dtype_like):
            The desired data-type for the array. (default: numpy.int64)

    Returns:
        FractionArray: A newly created array.

    """

    return FractionArray(np.empty(shape, dtype=dtype))


def empty_like(fa, shape=None):
    """
    Return an empty FractionArray with shape and type of input.

    Args:
        fa (FractionArray):
            Return an array of zeros with the same shape and type as fa given
            array.
        shape:
            Overrides the shape of the result.

    Returns:
        FractionArray: Array of zeros with the same shape and type as fa.

    """

    return FractionArray(data=np.empty_like(fa.data, shape=shape))


def zeros(shape, dtype=mpq):
    """
    Return a new FractionArray of given shape and type, filled with zeros.

    Args:
        shape (tuple of ints, int): Shape of the new array.
        dtype (dtype_like):
            The desired data-type for the array. (default: numpy.int64)

    Returns:
        FractionArray: A newly created array.

    """

    ans = empty(shape, dtype=dtype)
    ans.data.fill(mpq())

    return ans


def zeros_like(fa, shape=None):
    """
    Return an FractionArray of zeros with the same shape and type as a given
    array.

    Args:
        fa (FractionArray):
            Return an array of zeros with the same shape and type as fa given
            array.
        shape:
            Overrides the shape of the result.

    Returns:
        FractionArray: Array of zeros with the same shape and type as fa.

    """

    ans = empty_like(fa, shape=shape)
    ans.data.fill(mpq())

    return ans


def ones(shape, dtype=mpq):
    """
    Return a new FractionArray of given shape and type, filled with ones.

    Args:
        shape (tuple of ints, int): Shape of the new array.
        dtype (dtype_like):
            The desired data-type for the array. (default: numpy.int64)

    Returns:
        FractionArray: A newly created array.

    """

    ans = empty(shape, dtype=dtype)
    ans.data.fill(mpq(1))

    return ans


def ones_like(fa, shape=None):
    """
    Return an FractionArray of ones with the same shape and type as a given
    array.

    Args:
        fa (FractionArray):
            Return an array of zeros with the same shape and type as fa given
            array.
        shape:
            Overrides the shape of the result.

    Returns:
        FractionArray: Array of zeros with the same shape and type as fa.

    """

    ans = empty_like(fa, shape=shape)
    ans.data.fill(mpq(1))

    return ans


def concatenate(farrays, axis=0, dtype=None):
    """
    Join a sequence of arrays along an existing axis.

    Args:
        farrays (sequence of FractionArrays): A sequence of FractionArrays
        axis (int, optional):
            The axis along which the arrays will be joined. If axis is None,
            arrays are flattened before use. (default 0)
        dtype (dtype_like):
            If provided, the destination array will have this dtype.

    Returns:
        FractionArray: The concatenated array.

    """

    return FractionArray(
        np.concatenate([farr.data for farr in farrays], axis=axis, dtype=dtype)
    )


def hstack(farrays):
    """
    Stack arrays in sequence horizontally (column wise).

    Args:
        farrays (sequence of FractionArrays): A sequence of FractionArrays

    Returns:
        FractionArray: The stacked array.

    """

    return FractionArray(np.hstack([farr.data for farr in farrays]))


def vstack(farrays):
    """
    Stack arrays in sequence vertically (row wise).

    Args:
        farrays (sequence of FractionArrays): A sequence of FractionArrays

    Returns:
        FractionArray: The stacked array.

    """

    return FractionArray(np.vstack([farr.data for farr in farrays]))


def dot(fa: FractionArray, fb: FractionArray):
    """
    Dot product of two arrays.

    Args:
        fa (FractionArray): First argument
        fb (FractionArray): Second argument

    Raises:
        ValueError: If the sizes of the input are non-conformant.

    Returns:
        The dot product of fa and fb.

    """

    if fa.shape != fb.shape:
        raise ValueError(
            "Shapes are non-conformant fa: {}, fb: {}".format(
                fa.shape, fb.shape
            )
        )

    if len(fa.shape) != 1:
        raise NotImplementedError("Only 1D array dot product is supported.")

    return np.sum(fa.data * fb.data, initial=mpq())
