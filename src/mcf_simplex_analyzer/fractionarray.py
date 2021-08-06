"""
FractionArray is an array of fractions.

TODO:
    - Clearly define types and their interactions
    - Refactor initialization
    - Refactor operation dispatch for operations
    - Rename ``regular`` -> ``integral``
    - Refactor argmin, argmas, min, max ...

"""

from fractions import Fraction

import operator
import numpy as np
from scipy.sparse.sputils import isintlike


def is_integral_array(x):
    """
    Check whether ``x`` is a numpy array containing integral values.

    Args:
        x (Any): The input to check.

    Returns:
        (bool): True if ``x`` is an integral array, otherwise False.

    """

    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer)


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

    def __init__(self, numerator, denominator, dtype=int, _normalize=True):
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
            dtype = int

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

    def minimum(self):
        raise NotImplemented

    def maximum(self):
        raise NotImplemented

    def argmin(self):
        # TODO: This can overflow! Find a safer method.
        #       For example first convert to floats.
        lcm = np.lcm.reduce(self.denominator, initial=1)
        return np.argmin((lcm // self.denominator) * self.numerator)

    def argmax(self):
        # TODO: This can overflow! Find a safer method.
        #       For example first convert to floats.
        lcm = np.lcm.reduce(self.denominator, initial=1)
        return np.argmax((lcm // self.denominator) * self.numerator)

    def reshape(self, *args, **kwargs):
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
        return FractionArray(abs(self.numerator), self.denominator.copy())

    def _comparison(self, other, compare):
        if isinstance(other, FractionArray):
            return compare(
                self.numerator * other.denominator,
                other.numerator * self.denominator,
            )

        return compare(self.numerator, other * self.denominator)

    def __eq__(self, other):
        return self._comparison(other, operator.eq)

    def __ne__(self, other):
        return self._comparison(other, operator.ne)

    def __lt__(self, other):
        return self._comparison(other, operator.lt)

    def __gt__(self, other):
        return self._comparison(other, operator.gt)

    def __le__(self, other):
        return self._comparison(other, operator.le)

    def __ge__(self, other):
        return self._comparison(other, operator.ge)

    def _add_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator + self.denominator * f.numerator,
            self.denominator * f.denominator,
        )

    def _add_regular(self, x):
        return FractionArray(
            self.numerator + x * self.denominator,
            self.denominator.copy(),
        )

    def __add__(self, other):  # self + other
        if isintlike(other) and other == 0:
            return self.copy()

        if isinstance(other, (Fraction, FractionArray)):
            return self._add_fractional(other)

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            return self._add_regular(other)

        raise ValueError("Cannot add: {}".format(other.dtype))

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def _sub_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator - self.denominator * f.numerator,
            self.denominator * f.denominator,
        )

    def _sub_regular(self, x):
        return FractionArray(
            self.numerator - x * self.denominator,
            self.denominator.copy(),
        )

    def __sub__(self, other):  # self - other
        if isintlike(other) and other == 0:
            return self.copy()

        if isinstance(other, (Fraction, FractionArray)):
            return self._sub_fractional(other)

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            return self._sub_regular(other)

        raise ValueError("Cannot subtract: {}".format(other.dtype))

    def _rsub_fractional(self, f):
        return FractionArray(
            self.denominator * f.numerator - self.numerator * f.denominator,
            self.denominator * f.denominator,
        )

    def _rsub_regular(self, x):
        return FractionArray(
            x * self.denominator - self.numerator,
            self.denominator.copy(),
        )

    def __rsub__(self, other):  # other - self
        if isintlike(other) and other == 0:
            return -self.copy()

        if isinstance(other, (Fraction, FractionArray)):
            return self._rsub_fractional(other)

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            return self._rsub_regular(other)

        raise ValueError("Cannot subtract from: {}".format(other.dtype))

    def __neg__(self):
        return FractionArray(
            -self.numerator.copy(), self.denominator.copy(), _normalize=False
        )

    def _mul_fractional(self, f):
        return FractionArray(
            self.numerator * f.numerator, self.denominator * f.denominator
        )

    def _mul_regular(self, x):
        return FractionArray(x * self.numerator, self.denominator.copy())

    def __mul__(self, other):
        if isintlike(other):
            return self._mul_regular(other)

        # other = np.asanyarray(other)

        if is_integral_array(other):
            return self._mul_regular(other)

        if isinstance(other, (Fraction, FractionArray)):
            return self._mul_fractional(other)

        raise ValueError("Cannot multiply: {}".format(other.dtype))

    def __rmul__(self, other):
        return self.__mul__(other)

    def _div_fractional(self, f):
        return FractionArray(
            self.numerator * f.denominator, self.denominator * f.numerator
        )

    def _div_regular(self, x):
        return FractionArray(self.numerator, self.denominator * x)

    def __div__(self, other):  # self // other
        if is_integral_array(other) or isintlike(other):
            return self._div_regular(other)

        if isinstance(other, (Fraction, FractionArray)):
            return self._div_fractional(other)

        raise ValueError("Cannot divide with: {}".format(other.dtype))

    def __floordiv__(self, other):
        return self.__div__(other)

    def __truediv__(self, other):
        return self.__div__(other)

    def _iadd_fractional(self, f):
        self.numerator = (
            self.numerator * f.denominator + self.denominator * f.numerator
        )
        self.denominator = self.denominator * f.denominator

        self.normalize()

    def _iadd_regular(self, x):
        self.numerator = self.numerator + x * self.denominator

        self.normalize()

    def __iadd__(self, other):
        if isintlike(other) and other == 0:
            return self

        if isinstance(other, (Fraction, FractionArray)):
            self._iadd_fractional(other)
            return self

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            self._iadd_regular(other)
            return self

        raise ValueError("Cannot add in-place: {}".format(other.dtype))

    def _isub_fractional(self, f):
        self.numerator = (
            self.numerator * f.denominator - self.denominator * f.numerator
        )
        self.denominator = self.denominator * f.denominator

        self.normalize()

    def _isub_regular(self, x):
        self.numerator = self.numerator - x * self.denominator

        self.normalize()

    def __isub__(self, other):
        if isintlike(other) and other == 0:
            return self

        if isinstance(other, (Fraction, FractionArray)):
            self._isub_fractional(other)
            return self

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            self._isub_regular(other)
            return self

        raise ValueError("Cannot subtract in-place: {}".format(type(other)))

    def _imul_fractional(self, f):
        self.numerator = self.numerator * f.numerator
        self.denominator = self.denominator * f.denominator

        self.normalize()

    def _imul_regular(self, x):
        new_numerator = x * self.numerator
        self.numerator = new_numerator

        self.normalize()

    def __imul__(self, other):
        if isintlike(other):
            self._imul_regular(other)
            return self

        # other = np.asanyarray(other)

        if is_integral_array(other):
            self._imul_regular(other)
            return self

        if isinstance(other, (Fraction, FractionArray)):
            self._imul_fractional(other)
            return self

        raise ValueError("Cannot multiply in-place: {}".format(other.dtype))

    def _idiv_fractional(self, f):
        self.numerator = self.numerator * f.denominator
        self.denominator = self.denominator * f.numerator

        self.normalize()

    def _idiv_regular(self, x):
        self.numerator = self.numerator
        self.denominator = self.denominator * x

        self.normalize()

    def __itruediv__(self, other):
        if is_integral_array(other) or isintlike(other):
            self._idiv_regular(other)
            return self

        if isinstance(other, (Fraction, FractionArray)):
            self._idiv_fractional(other)
            return self

        raise ValueError("Cannot divide in-place: {}".format(other.dtype))

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __len__(self):
        return len(self.numerator)

    def __getitem__(self, key):
        numerator = self.numerator[key]
        denominator = self.denominator[key]

        if np.shape(numerator) == ():
            return Fraction(numerator, denominator, _normalize=False)

        return FractionArray(numerator, denominator, _normalize=False)

    def __setitem__(self, key, value):
        # TODO: Deprecated tuple, remove
        if isinstance(value, tuple):
            numerator, denominator = value
        else:
            numerator = value.numerator
            denominator = value.denominator

        self.numerator[key] = numerator
        self.denominator[key] = denominator

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
