# -*- coding: utf-8 -*-
""" Tests for Simplex. """

import pytest
from fractions import Fraction
import numpy as np

from mcf_simplex_analyzer.simplex import (
    Simplex,
    LPFormulation,
    LPFormType,
)
from mcf_simplex_analyzer.fractionarray import FractionArray


@pytest.mark.simplex
def test_instantiate_canonical():
    """ Create an empty from lists. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array(np.empty(0).reshape(0, 0)),
        rhs=FractionArray.from_array([]),
        objective=FractionArray.from_array([]),
    )

    result = Simplex.instantiate(problem)

    assert result is not None


@pytest.mark.simplex
def test_instantiate_standard():
    """ Create an empty from lists. """

    problem = LPFormulation(
        LPFormType.Standard,
        table=FractionArray.from_array(np.empty(0).reshape(0, 0)),
        rhs=FractionArray.from_array([]),
        objective=FractionArray.from_array([]),
    )

    result = Simplex.instantiate(problem)

    assert result is not None


@pytest.mark.simplex
def test_problem_one_canonical():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]),
        rhs=FractionArray.from_array([5, 11, 8]),
        objective=FractionArray.from_array([5, 4, 3]),
    )

    result = Simplex.instantiate(problem).solve()

    expected_value = Fraction(13)
    expected_variables = FractionArray(
        numerator=[
            2,
            0,
            1,
        ],
        denominator=[1, 1, 1],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_one_standard():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Standard,
        table=FractionArray.from_array(
            [[2, 3, 1, 1, 0, 0], [4, 1, 2, 0, 1, 0], [3, 4, 2, 0, 0, 1]]
        ),
        rhs=FractionArray.from_array([5, 11, 8]),
        objective=FractionArray.from_array([5, 4, 3, 0, 0, 0]),
    )

    result = Simplex.instantiate(problem).solve()

    expected_value = Fraction(13)
    expected_variables = FractionArray(
        numerator=[2, 0, 1, 0, 1, 0], denominator=[1, 1, 1, 1, 1, 1]
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_two_canonical():
    """ Test problem from canonical formulation. """

    table = FractionArray.from_array(
        np.array(
            [
                [1, 3, 1],
                [-1, 0, 3],
                [2, -1, 2],
                [2, 3, -1],
            ],
            dtype=np.int64,
        )
    )
    rhs = FractionArray.from_array(np.array([3, 2, 4, 2], dtype=np.int64))
    objective = FractionArray.from_array(np.array([5, 5, 3], dtype=np.int64))

    problem = LPFormulation(LPFormType.Canonical, table, rhs, objective)
    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    expected_value = Fraction(10)
    expected_variables = FractionArray(
        numerator=[
            32,
            8,
            30,
        ],
        denominator=[29, 29, 29],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_two_standard():
    """ Test problem from canonical formulation. """

    table = FractionArray.from_array(
        np.array(
            [
                [1, 3, 1, 1, 0, 0, 0],
                [-1, 0, 3, 0, 1, 0, 0],
                [2, -1, 2, 0, 0, 1, 0],
                [2, 3, -1, 0, 0, 0, 1],
            ],
            dtype=np.int64,
        )
    )
    rhs = FractionArray.from_array(np.array([3, 2, 4, 2], dtype=np.int64))
    objective = FractionArray.from_array(
        np.array([5, 5, 3, 0, 0, 0, 0], dtype=np.int64)
    )

    problem = LPFormulation(LPFormType.Standard, table, rhs, objective)
    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    expected_value = Fraction(10)
    expected_variables = FractionArray(
        numerator=[32, 8, 30, 1, 0, 0, 0],
        denominator=[29, 29, 29, 29, 1, 1, 1],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_degeneracy_canonical():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[0, 0, 2], [2, -4, 6], [-1, 3, 4]]),
        rhs=FractionArray.from_array([1, 3, 2]),
        objective=FractionArray.from_array([2, -1, 8]),
    )

    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    expected_value = Fraction(27, 2)
    expected_variables = FractionArray(
        numerator=[
            17,
            7,
            0,
        ],
        denominator=[2, 2, 1],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_degeneracy_standard():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array(
            [[0, 0, 2, 1, 0, 0], [2, -4, 6, 0, 1, 0], [-1, 3, 4, 0, 0, 1]]
        ),
        rhs=FractionArray.from_array([1, 3, 2]),
        objective=FractionArray.from_array([2, -1, 8, 0, 0, 0]),
    )

    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    expected_value = Fraction(27, 2)
    expected_variables = FractionArray(
        numerator=[17, 7, 0, 1, 0, 0],
        denominator=[2, 2, 1, 1, 1, 1],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_cycling_canonical():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array(
            [
                [Fraction(1, 2), Fraction(-55, 10), -Fraction(25, 10), 9],
                [Fraction(1, 2), -Fraction(3, 2), -Fraction(1, 2), 1],
                [1, 0, 0, 0],
            ]
        ),
        rhs=FractionArray.from_array([0, 0, 1]),
        objective=FractionArray.from_array([10, -57, -9, -24]),
    )

    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    assert result["result"] == "cycle"


@pytest.mark.simplex
def test_problem_cycling_standard():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array(
            [
                [
                    Fraction(1, 2),
                    Fraction(-55, 10),
                    -Fraction(25, 10),
                    9,
                    1,
                    0,
                    0,
                ],
                [Fraction(1, 2), -Fraction(3, 2), -Fraction(1, 2), 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 1],
            ]
        ),
        rhs=FractionArray.from_array([0, 0, 1]),
        objective=FractionArray.from_array([10, -57, -9, -24, 0, 0, 0]),
    )

    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    assert result["result"] == "cycle"


@pytest.mark.simplex
def test_problem_two_phase():
    """ Test two phase solving """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[2, -1, 2], [2, -3, 1], [-1, 1, -2]]),
        rhs=FractionArray.from_array([4, -5, -1]),
        objective=FractionArray.from_array([1, -1, 1]),
    )

    simplex = Simplex.instantiate(problem)
    result = simplex.solve()

    expected_value = Fraction(3, 5)
    expected_variables = FractionArray(
        numerator=[
            0,
            14,
            17,
        ],
        denominator=[1, 5, 5],
    )

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] == expected_variables)


@pytest.mark.simplex
def test_problem_infeasible_canonical():
    """ Test two phase solving """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[1], [1]]),
        rhs=FractionArray.from_array([1, -1]),
        objective=FractionArray.from_array([1]),
    )

    result = Simplex.instantiate(problem).solve()

    assert result["result"] == "infeasible"


@pytest.mark.simplex
def test_problem_unbounded_canonical():
    """ Test two phase solving """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[2, 2, -1], [3, -2, 1], [1, -3, 1]]),
        rhs=FractionArray.from_array([10, 10, 10]),
        objective=FractionArray.from_array([1, 3, -1]),
    )

    result = Simplex.instantiate(problem).solve()

    assert result["result"] == "unbounded"
