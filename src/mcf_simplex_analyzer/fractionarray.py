from fractions import Fraction

import operator
import numpy as np
from scipy.sparse.sputils import isintlike


def is_integral_array(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer)


def to_array(input_iter, dtype=np.integer):
    if is_integral_array(input_iter):
        return input_iter

    try:
        iterator = iter(input_iter)
        return np.fromiter(iterator, dtype=dtype)
    except TypeError:
        raise ValueError("Invalid input type: {}".format(type(input_iter)))


class FractionArray:
    """ Array of rational numbers. """

    numerator: np.ndarray
    denominator: np.ndarray

    @classmethod
    def from_array(cls, x):
        # Array of integers
        if is_integral_array(x):
            return cls(x, np.ones_like(x))

        try:
            numerator, denominator = [], []
            for val in x:
                if isinstance(val, Fraction):
                    numerator.append(val.numerator)
                    denominator.append(val.denominator)
                    continue

                if np.issubdtype(type(val), np.integer):
                    numerator.append(val)
                    denominator.append(1)
                    continue

                raise TypeError

            return cls(numerator, denominator)
        except TypeError:
            raise ValueError("Invalid input: {}".format(type(x)))

    def __init__(
        self, numerator, denominator, normalize=True, dtype=np.integer
    ):
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

        if normalize:
            self.normalize()

    def normalize(self):
        gcds = np.gcd(self.numerator, self.denominator)

        neg_ans = np.logical_xor(self.numerator < 0, self.denominator < 0)

        self.numerator = abs(self.numerator) // gcds
        self.denominator = abs(self.denominator) // gcds

        self.numerator[neg_ans] *= -1

    def reshape(self, *args, **kwargs):
        return FractionArray(
            self.numerator.reshape(*args, **kwargs),
            self.denominator.reshape(*args, **kwargs),
        )

    def copy(self):
        return FractionArray(
            self.numerator.copy(), self.denominator.copy(), normalize=False
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
        if other == 0:
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
        if other == 0:
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
        if other == 0:
            return -self.copy()

        if isinstance(other, (Fraction, FractionArray)):
            return self._rsub_fractional(other)

        if is_integral_array(other) or np.issubdtype(type(other), np.integer):
            return self._rsub_regular(other)

        raise ValueError("Cannot subtract from: {}".format(other.dtype))

    def __neg__(self):
        return FractionArray(
            -self.numerator.copy(), self.denominator.copy(), normalize=False
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

        other = np.asanyarray(other)

        if is_integral_array(other):
            return self._mul_regular(other)

        if isinstance(other, (Fraction, FractionArray)):
            return self._mul_fractional(other)

        raise ValueError("Cannot multiply: {}".format(other.dtype))

    def __rmul__(self, other):
        self.__mul__(other)

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

    def __len__(self):
        return len(self.numerator)

    def __getitem__(self, key):
        numerator = self.numerator[key]
        denominator = self.denominator[key]

        if np.shape(numerator) == ():
            return Fraction(numerator, denominator)

        return FractionArray(numerator, denominator)

    def __setitem__(self, key, value):
        numerator, denominator = None, None

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
