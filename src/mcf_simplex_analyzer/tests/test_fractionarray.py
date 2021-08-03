# -*- coding: utf-8 -*-
""" Tests for FractionArray. """

import itertools
from fractions import Fraction

import pytest
import numpy as np
import numpy.testing

from mcf_simplex_analyzer.fractionarray import FractionArray


def assert_normal_form(fa: FractionArray):
    gcds = np.gcd(fa.numerator, fa.denominator)

    assert np.all(gcds == 1)


@pytest.mark.fractionarray
@pytest.mark.parametrize("num", [[], np.empty((0,))])
@pytest.mark.parametrize("denom", [[], np.empty((0,))])
def test_create_empty(num, denom):
    """ Create an empty from lists. """

    empty = FractionArray([], [])

    numpy.testing.assert_array_equal(
        empty.numerator, np.empty((0,), dtype=np.integer)
    )
    numpy.testing.assert_array_equal(
        empty.denominator, np.empty((0,), dtype=np.integer)
    )


@pytest.mark.fractionarray
def test_from_integral_array():
    """ Create array from an integral array. """

    fa = FractionArray.from_array(np.arange(10))

    numpy.testing.assert_array_equal(fa.numerator, np.arange(10))


@pytest.mark.fractionarray
def test_from_float_array():
    """ Fail to create a fraction array from floating point values. """

    with pytest.raises(ValueError):
        FractionArray.from_array(np.arange(10, dtype="f"))


@pytest.mark.fractionarray
def test_from_fraction_list():
    """ Create array from an  List of Fractions. """

    fa = FractionArray.from_array(
        [Fraction(1, 3), Fraction(3, 2), Fraction(2, 8)]
    )

    numpy.testing.assert_array_equal(fa.numerator, [1, 3, 1])
    numpy.testing.assert_array_equal(fa.denominator, [3, 2, 4])


@pytest.mark.fractionarray
def test_from_mixed_list():
    """ Create array from an  List of Fractions. """

    fa = FractionArray.from_array(
        [Fraction(1, 3), 5, Fraction(3, 2), Fraction(2, 8), 10]
    )

    numpy.testing.assert_array_equal(fa.numerator, [1, 5, 3, 1, 10])
    numpy.testing.assert_array_equal(fa.denominator, [3, 1, 2, 4, 1])


@pytest.mark.fractionarray
@pytest.mark.parametrize("num_shape", [(0, 1), (1, 0), (2, 3)])
@pytest.mark.parametrize("denom_shape", [(0, 2), (2, 0), (3, 4)])
def test_invalid_shape(num_shape, denom_shape):
    """ Raise error when invalid shapes are supplied. """

    with pytest.raises(ValueError):
        FractionArray(
            np.empty(num_shape, dtype=np.integer),
            np.empty(denom_shape, dtype=np.integer),
        )


@pytest.mark.fractionarray
@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (2, 3), (1,), (100,)])
def test_correct_shape(shape):
    """ Test shape is preserved. """

    fa = FractionArray(
        np.empty(shape, dtype=np.integer),
        np.empty(shape, dtype=np.integer),
    )

    assert fa.shape == shape


@pytest.mark.fractionarray
@pytest.mark.parametrize(
    "num_mult, denom_mult", itertools.combinations(range(1, 11), 2)
)
def test_normalize(num_mult, denom_mult):
    """ Test correct normalization. """

    fa = FractionArray(
        np.arange(10) * num_mult, np.ones(10, dtype=np.integer) * denom_mult
    )

    assert_normal_form(fa)


@pytest.mark.fractionarray
def test_abs():
    """ Test absolute value of FractionArray. """

    fa = FractionArray(np.arange(10) - 10, np.ones(10))

    np.testing.assert_array_equal(fa.numerator, np.arange(10) - 10)
    np.testing.assert_array_equal(fa.denominator, np.ones(10))


@pytest.mark.fractionarray
@pytest.mark.parametrize(
    "this, other, ans",
    [
        ([1, 2, 3], [1, 2, 3], [True, True, True]),
        ([1, 3, 3], [1, 2, 3], [True, False, True]),
        ([1, 3, 4], [1, 2, 3], [True, False, False]),
        ([1, -1, 3], [0, 2, 2], [False, False, False]),
        ([1, -1, 3], 1, [True, False, False]),
        ([1, -1, 3], Fraction(3, 1), [False, False, True]),
        ([1, Fraction(4, 8), 3], Fraction(2, 4), [False, True, False]),
    ],
)
def test_eq_array(this, other, ans):
    fa = FractionArray.from_array(this)

    numpy.testing.assert_array_equal(fa == other, ans)


@pytest.mark.fractionarray
@pytest.mark.parametrize(
    "this, other, ans",
    [
        ([1, 2, 3], [1, 2, 3], np.logical_not([True, True, True])),
        ([1, 3, 3], [1, 2, 3], np.logical_not([True, False, True])),
        ([1, 3, 4], [1, 2, 3], np.logical_not([True, False, False])),
        ([1, -1, 3], [0, 2, 2], np.logical_not([False, False, False])),
        ([1, -1, 3], 1, np.logical_not([True, False, False])),
        ([1, -1, 3], Fraction(3, 1), np.logical_not([False, False, True])),
        (
            [1, Fraction(4, 8), 3],
            Fraction(2, 4),
            np.logical_not([False, True, False]),
        ),
    ],
)
def test_neq_array(this, other, ans):
    fa = FractionArray.from_array(this)

    numpy.testing.assert_array_equal(fa != other, ans)
