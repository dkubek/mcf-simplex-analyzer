# -*- coding: utf-8 -*-
""" Tests for Simplex. """

import pytest
from fractions import Fraction

from mcf_simplex_analyzer.simplex import RevisedSimplex, LPModel


@pytest.mark.revised_simplex
def test_instantiate_empty():
    """ Create empty simplex. """

    model = LPModel()
    simplex = RevisedSimplex.instantiate(model)

    ans = simplex.solve()
    assert ans.status == "success"
    assert ans.value == 0


@pytest.mark.revised_simplex
def test_problem_one():
    model = LPModel(name="problem_one")
    x = [model.new_variable(name=f"x_{i}") for i in range(4)]

    model.objective_function = 19 * x[0] + 13 * x[1] + 12 * x[2] + 17 * x[3]
    model.constraints = [
        3 * x[0] + 2 * x[1] + x[2] + 2 * x[3] <= 225,
        x[0] + x[1] + x[2] + x[3] <= 117,
        4 * x[0] + 3 * x[1] + 3 * x[2] + 4 * x[3] <= 420,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == 1827


@pytest.mark.revised_simplex
def test_problem_two():
    model = LPModel(name="problem_two")
    x = [model.new_variable(name=f"x_{i}") for i in range(3)]

    model.objective_function = 5 * x[0] + 4 * x[1] + 3 * x[2]
    model.constraints = [
        2 * x[0] + 3 * x[1] + x[2] <= 5,
        4 * x[0] + x[1] + 2 * x[2] <= 11,
        3 * x[0] + 4 * x[1] + 2 * x[2] <= 8,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == Fraction(13)


@pytest.mark.revised_simplex
def test_problem_three():
    model = LPModel(name="problem_three")
    x = [model.new_variable(name=f"x_{i}") for i in range(3)]

    model.objective_function = 5 * x[0] + 5 * x[1] + 3 * x[2]
    model.constraints = [
        x[0] + 3 * x[1] + x[2] <= 3,
        -x[0] + 3 * x[2] <= 2,
        2 * x[0] - x[1] + 2 * x[2] <= 4,
        2 * x[0] + 3 * x[1] - x[2] <= 2,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == Fraction(10)


@pytest.mark.revised_simplex
def test_problem_degeneracy():
    """ Test degenerate problem. """

    model = LPModel(name="problem_degeneracy")
    x = [model.new_variable(name=f"x_{i}") for i in range(3)]

    model.objective_function = 2 * x[0] - x[1] + 8 * x[2]
    model.constraints = [
        x[2] <= 1,
        2 * x[0] - 4 * x[1] + 6 * x[2] <= 3,
        -x[0] + 3 * x[1] + 4 * x[2] <= 2,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == Fraction(27, 2)


@pytest.mark.revised_simplex
def test_problem_cycling():
    """ Test problem cycling problem. """

    model = LPModel(name="problem_cycling")
    x = [model.new_variable(name=f"x_{i}") for i in range(4)]

    model.constraints = [
        Fraction(1, 2) * x[0]
        + Fraction(-55, 10) * x[1]
        - Fraction(25, 10) * x[2]
        + 9 * x[3]
        <= 0,
        Fraction(1, 2) * x[0]
        - Fraction(3, 2) * x[1]
        - Fraction(1, 2) * x[2]
        + x[3]
        <= 0,
        x[0] <= 1,
    ]
    model.objective_function = 10 * x[0] - 57 * x[1] - 9 * x[2] - 24 * x[3]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == Fraction(1, 1)


@pytest.mark.revised_simplex
def test_problem_two_phase():
    """ Test two phase solving """

    model = LPModel(name="problem_two_phase")
    x = [model.new_variable(name=f"x_{i}") for i in range(3)]

    model.objective_function = x[0] - x[1] + x[2]
    model.constraints = [
        2 * x[0] - x[1] + 2 * x[2] <= 4,
        2 * x[0] - 3 * x[1] + x[2] <= -5,
        -x[0] + x[1] - 2 * x[2] <= -1,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "success"
    assert ans.value == Fraction(3, 5)


@pytest.mark.revised_simplex
def test_problem_unbounded():
    """ Test unbounded problem """

    model = LPModel(name="problem_unbounded")
    x = [model.new_variable(name=f"x_{i}") for i in range(3)]

    model.objective_function = x[0] + 3 * x[1] - x[2]
    model.constraints = [
        2 * x[0] + 2 * x[1] - x[2] <= 10,
        3 * x[0] - 2 * x[1] + x[2] <= 10,
        x[0] - 3 * x[1] + x[2] <= 10,
    ]

    simplex = RevisedSimplex.instantiate(model)
    ans = simplex.solve()

    assert ans.status == "unbounded"
    assert ans.value is None
