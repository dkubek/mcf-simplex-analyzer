# -*- coding: utf-8 -*-
"""
FractionArray is an array of fractions.

"""

from fractions import Fraction

import operator
import numpy as np

# from scipy.sparse.sputils import isintlike


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

    Attributes:
        numerator: An array of numerators.
        denominator: An array of denominators

    """

    numerator: np.ndarray
    denominator: np.ndarray

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

        if is_integral_array(input_values):
            return cls(input_values, np.ones_like(input_values))

        # Test input is iterable
        try:
            iter(input_values)
        except TypeError:
            raise TypeError(
                "Input is not an iterable: {}".format(type(input_values))
            )

        original_shape = np.shape(input_values)
        size = np.size(input_values)

        numerator, denominator = np.empty(size, dtype=int), np.empty(
            size, dtype=int
        )
        for i, val in enumerate(np.ravel(input_values)):
            if isinstance(val, Fraction):
                numerator[i] = val.numerator
                denominator[i] = val.denominator
                continue

            if np.issubdtype(type(val), np.integer):
                numerator[i] = val
                denominator[i] = 1
                continue

            raise TypeError("Invalid value: {}".format(val))

        return cls(
            numerator.reshape(original_shape),
            denominator.reshape(original_shape),
        )

    def __init__(
        self,
        numerator,
        denominator,
        dtype=np.int64,
        _normalize=True,
        _check_type=True,
    ):
        """
        Create a new FractionArray.

        Args:
            numerator (array_like): Array of numerators
            denominator (array_like): Array of denominators
            dtype (dtype_like):
                An integral datatype of the internal representation.
                (default: int)
            _normalize (bool): Whether to normalize the input. (default: True)

        Warnings:
            When ``numerator`` or ``denominator`` are numpy arrays a reference
            is kept instead of copying.

        Raises:
            ValueError: When the input iterable cannot be converted or is
            invalid.

        Raises:

        """

        if dtype is None:
            dtype = np.int64

        if _check_type:
            numerator = to_array(numerator, dtype=dtype)
            denominator = to_array(denominator, dtype=dtype)

        if numerator.shape != denominator.shape:
            raise ValueError(
                "Different numerator and denominator shapes {} vs {}".format(
                    numerator.shape, denominator.shape
                )
            )

        self.numerator = numerator
        self.denominator = denominator

        self.shape = self.numerator.shape
        self.size = self.numerator.size

        if _normalize:
            self.normalize()

    def normalize(self):
        """ Convert the array to a normalized form. """

        gcds = np.gcd(self.numerator, self.denominator)

        neg_ans = np.logical_xor(self.numerator < 0, self.denominator < 0)

        self.numerator = abs(self.numerator) // gcds
        self.denominator = abs(self.denominator) // gcds

        self.numerator[neg_ans] *= -1

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

        return self[
            np.unravel_index(self.argmin(axis=axis, eps=eps), self.shape)
        ]

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

        return self[
            np.unravel_index(self.argmax(axis=axis, eps=eps), self.shape)
        ]

    def _arg_min_or_max(self, arg_ext_fun, ext_fun, axis=None, eps=1e-6):
        floats = np.true_divide(self.numerator, self.denominator)
        float_extreme = ext_fun(floats)

        valid = np.where(
            (floats <= float_extreme + eps) & (floats >= float_extreme - eps)
        )[0]

        lcm = np.lcm.reduce(self.denominator[valid], initial=1)
        return valid[
            arg_ext_fun(
                (lcm // self.denominator[valid]) * self.numerator[valid],
                axis=axis,
            )
        ]

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

        return self._arg_min_or_max(np.argmin, np.min, axis=axis, eps=eps)

    def argmax(self, axis=None, eps=1e-6):
        """
        Returns the indices of the maximum values along an axis.

        See Also:
            FractionArray.argmin
            numpy.unravel_index

        """

        return self._arg_min_or_max(np.argmax, np.max, axis=axis, eps=eps)

    def reshape(self, *args, **kwargs):
        """
        Returns an array containing the same data with a new shape.

        Returns:
            (FractionArray): The same FractionArray with given shape.

        """

        return FractionArray(
            self.numerator.reshape(*args, **kwargs),
            self.denominator.reshape(*args, **kwargs),
        )

    def resize(self, shape):
        raise NotImplementedError("Resizing not supported.")

    def copy(self):
        return FractionArray(
            self.numerator.copy(), self.denominator.copy(), _normalize=False
        )

    def __abs__(self):
        """ Return abs(self). """

        return FractionArray(abs(self.numerator), self.denominator.copy())

    def _comparison(self, other, compare):
        """
        Compare to ``other`` using a comparison function ``compare``.

        Args:
            other: Value to compare to.
            compare: Comparison function. (For example from the ``operators``
                     library.)

        Returns:
            (bool or array of bool): Array of boolean values of the same
            shape as the FractionArray shape.

        """

        # Quickly compare the sign
        if np.all(other == 0):
            return compare(self.numerator, other)

        if isinstance(other, FractionArray):
            return compare(
                self.numerator * other.denominator,
                other.numerator * self.denominator,
            )

        return compare(self.numerator, other * self.denominator)

    def __eq__(self, other):
        """ Returns self == other. """

        return self._comparison(other, operator.eq)

    def __ne__(self, other):
        """ Returns self != other. """

        return self._comparison(other, operator.ne)

    def __lt__(self, other):
        """ Returns self < other. """

        return self._comparison(other, operator.lt)

    def __gt__(self, other):
        """ Returns self > other. """

        return self._comparison(other, operator.gt)

    def __le__(self, other):
        """ Returns self <= other. """

        return self._comparison(other, operator.le)

    def __ge__(self, other):
        """ Returns self >= other. """

        return self._comparison(other, operator.ge)

    def _add_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator + self.denominator * f.numerator,
            self.denominator * f.denominator,
        )

    def _add_integral(self, x):
        return FractionArray(
            self.numerator + x * self.denominator,
            self.denominator.copy(),
        )

    def __add__(self, other):
        """ Returns self + other. """

        if isintlike(other) and other == 0:
            return self.copy()

        return _perform_op(
            other,
            self._add_fractional,
            self._add_integral,
            opname="addition",
        )

    def __radd__(self, other):
        """ Returns other + self. """

        return self.__add__(other)

    def _sub_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator - self.denominator * f.numerator,
            self.denominator * f.denominator,
        )

    def _sub_integral(self, x):
        return FractionArray(
            self.numerator - x * self.denominator,
            self.denominator.copy(),
        )

    def __sub__(self, other):
        """ Returns self - other. """

        if isintlike(other) and other == 0:
            return self.copy()

        return _perform_op(
            other,
            self._sub_fractional,
            self._sub_integral,
            opname="subtraction (left)",
        )

    def _rsub_fractional(self, f):
        return FractionArray(
            self.denominator * f.numerator - self.numerator * f.denominator,
            self.denominator * f.denominator,
        )

    def _rsub_integral(self, x):
        return FractionArray(
            x * self.denominator - self.numerator,
            self.denominator.copy(),
        )

    def __rsub__(self, other):  # other - self
        """ Returns other - self. """

        if isintlike(other) and other == 0:
            return -self.copy()

        return _perform_op(
            other,
            self._rsub_fractional,
            self._rsub_integral,
            opname="subtraction (right)",
        )

    def __neg__(self):
        """ Returns -self. """

        return FractionArray(
            -self.numerator.copy(), self.denominator.copy(), _normalize=False
        )

    def _mul_fractional(self, f):
        return FractionArray(
            self.numerator * f.numerator, self.denominator * f.denominator
        )

    def _mul_integral(self, x):
        return FractionArray(x * self.numerator, self.denominator.copy())

    def __mul__(self, other):
        """ Returns self * other. """

        if isintlike(other) and other == 1:
            return self.copy()

        return _perform_op(
            other,
            self._mul_fractional,
            self._mul_integral,
            opname="multiplication",
        )

    def __rmul__(self, other):
        """ Returns other * self. """

        return self.__mul__(other)

    def _div_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator, self.denominator * f.numerator
        )

    def _div_integral(self, x):
        return FractionArray(self.numerator, self.denominator * x)

    def __div__(self, other):
        """ Returns self // other. """

        if isintlike(other) and other == 1:
            return self.copy()

        return _perform_op(
            other,
            self._div_fractional,
            self._div_integral,
            opname="division",
        )

    def __floordiv__(self, other):
        """ Returns self // other. """

        return self.__div__(other)

    def __truediv__(self, other):
        """ Returns self / other. Performs floordiv. """

        return self.__div__(other)

    def _iadd_fractional(self, f):
        self.numerator = (
            self.numerator * f.denominator + self.denominator * f.numerator
        )
        self.denominator = self.denominator * f.denominator

        self.normalize()

        return self

    def _iadd_integral(self, x):
        self.numerator = self.numerator + x * self.denominator

        self.normalize()

        return self

    def __iadd__(self, other):
        """ Add other to self in-place. """

        if isintlike(other) and other == 0:
            return self

        return _perform_op(
            other,
            self._iadd_fractional,
            self._iadd_integral,
            opname="in-place addition",
        )

    def _isub_fractional(self, f):
        self.numerator = (
            self.numerator * f.denominator - self.denominator * f.numerator
        )
        self.denominator = self.denominator * f.denominator

        self.normalize()

        return self

    def _isub_integral(self, x):
        self.numerator = self.numerator - x * self.denominator

        self.normalize()

        return self

    def __isub__(self, other):
        """ Subtract other from self in-place. """

        if isintlike(other) and other == 0:
            return self

        return _perform_op(
            other,
            self._isub_fractional,
            self._isub_integral,
            opname="in-place subtraction",
        )

    def _imul_fractional(self, f):
        self.numerator = self.numerator * f.numerator
        self.denominator = self.denominator * f.denominator

        self.normalize()

        return self

    def _imul_integral(self, x):
        new_numerator = x * self.numerator
        self.numerator = new_numerator

        self.normalize()

        return self

    def __imul__(self, other):
        """ Multiply self by other in-place. """

        if isintlike(other) and other == 1:
            return self

        return _perform_op(
            other,
            self._imul_fractional,
            self._imul_integral,
            opname="in-place multiplication",
        )

    def _idiv_fractional(self, f):
        self.numerator = self.numerator * f.denominator
        self.denominator = self.denominator * f.numerator

        self.normalize()

        return self

    def _idiv_integral(self, x):
        self.numerator = self.numerator
        self.denominator = self.denominator * x

        self.normalize()

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

        return _perform_op(
            other,
            self._imul_fractional,
            self._imul_integral,
            opname="in-place division",
        )

    def transpose(self):
        """
        Reverses the dimensions of the sparse matrix.

        Returns:
            FractionArray: The transposed FractionArray.

        """

        return FractionArray(
            self.numerator.transpose(),
            self.denominator.transpose(),
            _normalize=False,
        )

    def __len__(self):
        return len(self.numerator)

    def __getitem__(self, key):
        numerator = self.numerator[key]
        denominator = self.denominator[key]

        if np.shape(numerator) == ():
            return Fraction(numerator, denominator, _normalize=False)

        return FractionArray(numerator, denominator, _normalize=False)

    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            numerator, denominator = value
        elif isinstance(value, (Fraction, FractionArray)):
            numerator = value.numerator
            denominator = value.denominator
        elif np.issubdtype(type(value), np.integer):
            numerator = value
            denominator = 1
        else:
            raise ValueError(
                "Cannot assign value of type {}".format(type(value))
            )

        self.numerator[key] = numerator
        self.denominator[key] = denominator

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
        return "FractionArray(numerator={}, denominator={})".format(
            self.numerator, self.denominator
        )

    __str__ = __repr__
