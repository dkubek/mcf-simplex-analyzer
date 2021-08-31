import fractions

import numpy as np
from mcf_simplex_analyzer.fractionarray._fractionarray import FractionArray


def empty(shape, dtype=np.int64):
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

    return FractionArray(
        numerator=np.empty(shape, dtype=dtype),
        denominator=np.empty(shape, dtype=dtype),
        _normalize=False,
    )


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

    return FractionArray(
        numerator=np.empty_like(fa.numerator, shape=shape),
        denominator=np.empty_like(fa.denominator, shape=shape),
        _normalize=False,
    )


def zeros(shape, dtype=np.int64):
    """
    Return a new FractionArray of given shape and type, filled with zeros.

    Args:
        shape (tuple of ints, int): Shape of the new array.
        dtype (dtype_like):
            The desired data-type for the array. (default: numpy.int64)

    Returns:
        FractionArray: A newly created array.

    """

    return FractionArray(
        numerator=np.zeros(shape, dtype=dtype),
        denominator=np.ones(shape, dtype=dtype),
    )


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

    return FractionArray(
        numerator=np.zeros_like(fa.numerator, shape=shape),
        denominator=np.ones_like(fa.denominator, shape=shape),
    )


def ones(shape, dtype=np.int64):
    """
    Return a new FractionArray of given shape and type, filled with ones.

    Args:
        shape (tuple of ints, int): Shape of the new array.
        dtype (dtype_like):
            The desired data-type for the array. (default: numpy.int64)

    Returns:
        FractionArray: A newly created array.

    """

    return FractionArray(
        numerator=np.ones(shape, dtype=dtype),
        denominator=np.ones(shape, dtype=dtype),
    )


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

    return FractionArray(
        numerator=np.ones_like(fa.numerator, shape=shape),
        denominator=np.ones_like(fa.denominator, shape=shape),
    )


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
        np.concatenate(
            [farr.numerator for farr in farrays], axis=axis, dtype=dtype
        ),
        np.concatenate(
            [farr.denominator for farr in farrays], axis=axis, dtype=dtype
        ),
    )


def hstack(farrays):
    """
    Stack arrays in sequence horizontally (column wise).

    Args:
        farrays (sequence of FractionArrays): A sequence of FractionArrays

    Returns:
        FractionArray: The stacked array.

    """

    return FractionArray(
        np.hstack([farr.numerator for farr in farrays]),
        np.hstack([farr.denominator for farr in farrays]),
    )


def vstack(farrays):
    """
    Stack arrays in sequence vertically (row wise).

    Args:
        farrays (sequence of FractionArrays): A sequence of FractionArrays

    Returns:
        FractionArray: The stacked array.

    """

    return FractionArray(
        np.vstack([farr.numerator for farr in farrays]),
        np.vstack([farr.denominator for farr in farrays]),
    )


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

    return np.sum(
        fa * fb, dtype=fractions.Fraction, initial=fractions.Fraction(0)
    )
