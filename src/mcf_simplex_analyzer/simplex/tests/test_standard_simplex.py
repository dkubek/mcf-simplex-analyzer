# -*- coding: utf-8 -*-
""" Tests for Simplex. """

import pytest
from fractions import Fraction
import numpy as np

from mcf_simplex_analyzer.simplex import (
    StandardSimplex,
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

    result = StandardSimplex.instantiate(problem)

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

    result = StandardSimplex.instantiate(problem)

    assert result is not None


PROBLEMS_SUCCESS = [
    (
        LPFormulation(
            LPFormType.Canonical,
            table=FractionArray.from_array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]),
            rhs=FractionArray.from_array([5, 11, 8]),
            objective=FractionArray.from_array([5, 4, 3]),
        ),
        Fraction(13),
    ),
    (
        LPFormulation(
            LPFormType.Standard,
            table=FractionArray.from_array(
                [[2, 3, 1, 1, 0, 0], [4, 1, 2, 0, 1, 0], [3, 4, 2, 0, 0, 1]]
            ),
            rhs=FractionArray.from_array([5, 11, 8]),
            objective=FractionArray.from_array([5, 4, 3, 0, 0, 0]),
        ),
        Fraction(13),
    ),
    (
        LPFormulation(
            LPFormType.Canonical,
            table=FractionArray.from_array(
                np.array(
                    [
                        [1, 3, 1],
                        [-1, 0, 3],
                        [2, -1, 2],
                        [2, 3, -1],
                    ],
                    dtype=np.int64,
                )
            ),
            rhs=FractionArray.from_array(
                np.array([3, 2, 4, 2], dtype=np.int64)
            ),
            objective=FractionArray.from_array(
                np.array([5, 5, 3], dtype=np.int64)
            ),
        ),
        Fraction(10),
    ),
    (
        LPFormulation(
            LPFormType.Standard,
            table=FractionArray.from_array(
                np.array(
                    [
                        [1, 3, 1, 1, 0, 0, 0],
                        [-1, 0, 3, 0, 1, 0, 0],
                        [2, -1, 2, 0, 0, 1, 0],
                        [2, 3, -1, 0, 0, 0, 1],
                    ],
                    dtype=np.int64,
                )
            ),
            rhs=FractionArray.from_array(
                np.array([3, 2, 4, 2], dtype=np.int64)
            ),
            objective=FractionArray.from_array(
                np.array([5, 5, 3, 0, 0, 0, 0], dtype=np.int64)
            ),
        ),
        Fraction(10),
    ),
]


@pytest.mark.simplex
@pytest.mark.parametrize("problem,expected_value", PROBLEMS_SUCCESS)
@pytest.mark.parametrize("decision_rule", ["dantzig", "bland", "lex"])
def test_problems(problem, expected_value, decision_rule):
    """ Test problem from canonical formulation. """

    simplex = StandardSimplex.instantiate(problem)
    result = simplex.solve(decision_rule=decision_rule)

    assert result["result"] == "success"
    assert result["value"] == expected_value
    assert np.all(result["variables"] >= 0)


@pytest.mark.simplex
def test_problem_degeneracy_canonical():
    """ Test problem from canonical formulation. """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[0, 0, 2], [2, -4, 6], [-1, 3, 4]]),
        rhs=FractionArray.from_array([1, 3, 2]),
        objective=FractionArray.from_array([2, -1, 8]),
    )

    simplex = StandardSimplex.instantiate(problem)
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

    simplex = StandardSimplex.instantiate(problem)
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

    simplex = StandardSimplex.instantiate(problem)
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

    simplex = StandardSimplex.instantiate(problem)
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

    simplex = StandardSimplex.instantiate(problem)
    result = simplex.solve(decision_rule="dantzig")

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
    """ Test infeasible problem """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[1], [1]]),
        rhs=FractionArray.from_array([1, -1]),
        objective=FractionArray.from_array([1]),
    )

    result = StandardSimplex.instantiate(problem).solve(
        decision_rule="dantzig"
    )

    assert result["result"] == "infeasible"


@pytest.mark.simplex
def test_problem_unbounded_canonical():
    """ Test unbounded problem """

    problem = LPFormulation(
        LPFormType.Canonical,
        table=FractionArray.from_array([[2, 2, -1], [3, -2, 1], [1, -3, 1]]),
        rhs=FractionArray.from_array([10, 10, 10]),
        objective=FractionArray.from_array([1, 3, -1]),
    )

    result = StandardSimplex.instantiate(problem).solve()

    assert result["result"] == "unbounded"


@pytest.mark.simplex
def test_problem_cycling_bland():
    """ Solve problem that cycles using Dantzig with Bland. """

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

    simplex = StandardSimplex.instantiate(problem)
    result = simplex.solve(decision_rule="bland")

    assert result["result"] == "success"
    assert result["value"] == Fraction(1)
    assert np.all(result["variables"] >= 0)


@pytest.mark.simplex
def test_problem_cycling_lex():
    """ Solve problem that cycles using Dantzig with Lexicographic rule. """

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

    simplex = StandardSimplex.instantiate(problem)
    result = simplex.solve(decision_rule="lex")

    assert result["result"] == "success"
    assert result["value"] == Fraction(1)
    assert np.all(result["variables"] >= 0)
