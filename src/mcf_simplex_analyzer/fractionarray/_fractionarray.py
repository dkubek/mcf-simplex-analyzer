# -*- coding: utf-8 -*-
"""
FractionArray is an array of fractions.

"""

from fractions import Fraction

import numpy as np

from gmpy2 import mpq


def is_integral_array(x):
    """
    Check whether ``x`` is a numpy array containing integral values.

    Args:
        x (Any): The input to check.

    Returns:
        (bool): True if ``x`` is an integral array, otherwise False.

    """

    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer)


def isintlike(x):
    return is_integral_array(x) or np.issubdtype(type(x), np.integer)


def to_array(input_iter, dtype=int):
    """
    Convert ``input_iter`` to an integral array.

    Args:
        input_iter (Iterable): The input iterable to convert.
        dtype (dtype_like): Datatype of the output array.

    Raises:
        ValueError: When the input iterable cannot be converted or is invalid.

    Returns:
        (array_like): The output integral array.

    """

    # Do nothing if the array is already an integral array
    if is_integral_array(input_iter):
        return input_iter

    try:
        iterator = iter(input_iter)
        return np.fromiter(iterator, dtype=dtype)
    except TypeError:
        raise ValueError("Invalid input type: {}".format(type(input_iter)))


def _perform_op(other, op_fractional, op_integral, opname="operation"):
    if isinstance(other, (Fraction, FractionArray)):
        return op_fractional(other)

    if is_integral_array(other) or np.issubdtype(type(other), np.integer):
        return op_integral(other)

    raise ValueError("Cannot perform {}: {}".format(opname, type(other)))


class FractionArray:
    """
    FractionArray represents an array (or matrix, ...) of fractions.

    The fractions are represented as two arrays of numbers: the numerators and
    denominators. It defines a similar interface to numpy arrays and supports
    all basic arithmetic operations.

    """

    # numerator: np.ndarray
    # denominator: np.ndarray
    data: np.ndarray

    @classmethod
    def from_array(cls, input_values):
        """
        Create a new FractionArray from given ``input_values``.

        Args:
            input_values (array-like or iterable): The input values.

            The valid input values are:
                - numpy array
                - iterable containing integral types or fractions (from the
                  python ``fractions`` module)

        Notes:
            If the ``input_values`` are an (numpy) array-like, the resulting
            array will inherit the datatype, otherwise, an ``int`` will be used
            (which defaults to numpy ``int64`` on most systems).

            The array is kept in a normalized state. The numerators carry the
            sign and the GCD of numerators and denominators is always 1.

        Warnings:
            When given an array-like as an input it won't be copied but a
            reference will be stored.

            Division by zero is silently ignored.

        Raises:
            TypeError: The input values are invalid (for example, given input
            contains floating point values or is of invalid shape).

        Returns:
            A new FractionArray.

        """

        # Test input is iterable
        try:
            iter(input_values)
        except TypeError:
            raise TypeError(
                "Input is not an iterable: {}".format(type(input_values))
            )

        original_shape = np.shape(input_values)
        size = np.size(input_values)

        data = np.empty(size, dtype=object)
        for i, val in enumerate(np.ravel(input_values)):
            if isinstance(val, (Fraction, mpq)):
                data[i] = mpq(int(val.numerator), int(val.denominator))
                continue

            if np.issubdtype(type(val), np.integer):
                data[i] = mpq(int(val))
                continue

            raise TypeError(
                "Invalid value: {} of type {}".format(val, type(val))
            )

        return cls(data.reshape(original_shape))

    def __init__(self, data):

        assert isinstance(data, np.ndarray)

        self.data = data
        self.shape = data.shape
        self.size = data.size
        self.dtype = mpq

    def min(self, axis=None, eps=1e-6):
        """
        Return the minimum or minimum along an axis.

        Notes:
            The values are first converted to floating point values and then
            close values determined by given ``eps`` are found and compared
            using exact arithmetic.

        Args:
            axis: Same as for numpy ``amin``
            eps (float): The precision of floating point comparison.

        Returns:
            The minimum of the array.

        """

        return self.data.min(axis=axis)

    def max(self, axis=None, eps=1e-6):
        """
        Return the maximum or maximum along an axis.

        Notes:
            The values are first converted to floating point values and then
            close values determined by given ``eps`` are found and compared
            using exact arithmetic.

        Args:
            axis: Same as for numpy ``amin``
            eps (float): The precision of floating point comparison.

        Returns:
            The maximum of the array.

        """

        return self.data.max(axis=axis)

    def argmin(self, axis=None, eps=1e-6):
        """
        Returns the indices of the minimum values along an axis.

        Notes:
            The values are first converted to floating point values and then
            close values determined by given ``eps`` are found and compared
            using exact arithmetic.

        Args:
            axis (int, optional):
                By default, the index is into the flattened array, otherwise
                along the specified axis.
            eps (float): The precision of floating point comparison.

        Returns:
            (index_array): Array of indices into the array. It has the same
            shape as self.shape with the dimension along axis removed.

        See Also:
            numpy.unravel_index

        """

        return self.data.argmin(axis=axis)

    def argmax(self, axis=None, eps=1e-6):
        """
        Returns the indices of the maximum values along an axis.

        See Also:
            FractionArray.argmin
            numpy.unravel_index

        """

        return self.data.argmax(axis=axis)

    def reshape(self, *args, **kwargs):
        """
        Returns an array containing the same data with a new shape.

        Returns:
            (FractionArray): The same FractionArray with given shape.

        """

        return FractionArray(self.data.reshape(*args, **kwargs))

    def resize(self, shape):
        raise NotImplementedError("Resizing not supported.")

    def copy(self):
        return FractionArray(self.data.copy())

    def __abs__(self):
        """ Return abs(self). """

        return FractionArray(abs(self.data))

    def __eq__(self, other):
        """ Returns self == other. """

        return self.data.__eq__(other)

    def __ne__(self, other):
        """ Returns self != other. """

        return self.data.__ne__(other)

    def __lt__(self, other):
        """ Returns self < other. """

        return self.data.__lt__(other)

    def __gt__(self, other):
        """ Returns self > other. """

        return self.data.__gt__(other)

    def __le__(self, other):
        """ Returns self <= other. """

        return self.data.__le__(other)

    def __ge__(self, other):
        """ Returns self >= other. """

        return self.data.__ge__(other)

    def __add__(self, other):
        """ Returns self + other. """

        return FractionArray(self.data.__add__(other))

    def __radd__(self, other):
        """ Returns other + self. """

        return FractionArray(self.data.__radd__(other))

    def __sub__(self, other):
        """ Returns self - other. """

        return FractionArray(self.data.__sub__(other))

    def __rsub__(self, other):  # other - self
        """ Returns other - self. """

        return self.data.__rsub__(other)

    def __neg__(self):
        """ Returns -self. """

        return FractionArray(-self.data)

    def __mul__(self, other):
        """ Returns self * other. """

        return FractionArray(self.data.__mul__(other))

    def __rmul__(self, other):
        """ Returns other * self. """

        return FractionArray(self.data.__rmul__(other))

    def __div__(self, other):
        return FractionArray(self.data / other)

    def __floordiv__(self, other):
        """ Returns self // other. """

        return self.__div__(other)

    def __truediv__(self, other):
        """ Returns self / other. Performs floordiv. """

        return self.__div__(other)

    def __iadd__(self, other):
        """ Add other to self in-place. """
        self.data = self.data.__iadd__(other)
        return self

    def __isub__(self, other):
        """ Subtract other from self in-place. """
        self.data = self.data.__isub__(other)
        return self

    def __imul__(self, other):
        """ Multiply self by other in-place. """

        self.data = self.data.__imul__(other)
        return self

    def __itruediv__(self, other):
        """ Divide self by other in-place. Performs floordiv. """

        return self.__idiv__(other)

    def __ifloordiv__(self, other):
        """ Divide self by other in-place. """

        return self.__idiv__(other)

    def __idiv__(self, other):
        """ Divide self by other in-place. """

        if isintlike(other) and other == 1:
            return self

        self.data /= other

        return self

    def transpose(self):
        """
        Reverses the dimensions of the sparse matrix.

        Returns:
            FractionArray: The transposed FractionArray.

        """

        return FractionArray(
            self.data.transpose(),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        ans = self.data[key]
        if isinstance(ans, mpq):
            return ans

        return FractionArray(ans)

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            numerator, denominator = value
            self.data[key] = mpq(numerator, denominator)
            return

        if isinstance(value, (Fraction, mpq)):
            self.data[key] = mpq(int(value.numerator), int(value.denominator))
            return

        if np.issubdtype(type(value), np.integer):
            self.data[key] = mpq(int(value))
            return

        if isinstance(value, FractionArray):
            self.data[key] = value.data
            return

        raise ValueError("Cannot assign value of type {}".format(type(value)))

    def __getattr__(self, attribute):
        if attribute == "T":
            return self.transpose()
        else:
            raise AttributeError(attribute + " not found")

    def __iter__(self):
        if len(self.shape) == 1:
            for r in range(self.shape[0]):
                yield self[r]
        else:
            for r in range(self.shape[0]):
                yield self[r, :]

    def __repr__(self):
        return "FractionArray(data={})".format(self.data)

    __str__ = __repr__
