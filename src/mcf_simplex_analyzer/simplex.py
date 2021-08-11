# -*- coding: utf-8 -*-
"""
The simplex algorithm.

TODO:
    - Refactor output
    - Refactor instantiation
    - Two phase simplex
    - Add tests

"""

import logging
from enum import Enum

import attr
import numpy as np

from .fractionarray import FractionArray

DEFAULT_MAX_ITERATIONS = 1e5


class LPFormType(Enum):
    Canonical = 0
    Standard = 1


@attr.s
class LPFormulation:
    type: LPFormType = attr.ib()

    table: FractionArray = attr.ib()
    rhs: FractionArray = attr.ib()
    objective: FractionArray = attr.ib()

    meta: dict = attr.ib(kw_only=True, factory=dict)


def _validate_formulation(formulation):
    if len(np.shape(formulation.table)) != 2:
        raise ValueError(
            "Invalid table shape: {}".format(np.shape(formulation.table))
        )

    if len(np.shape(formulation.rhs)) != 1:
        raise ValueError(
            "Invalid right hand side shape: {}".format(
                np.shape(formulation.rhs)
            )
        )

    if len(np.shape(formulation.objective)) != 1:
        raise ValueError(
            "Invalid objective function shape: {}".format(
                np.shape(formulation.objective)
            )
        )


@attr.s(kw_only=True)
class Simplex:
    table = attr.ib()
    _objective_row = attr.ib()

    base = attr.ib()
    _row_to_var_index = attr.ib()
    _var_index_to_row = attr.ib()

    @classmethod
    def _from_canonical(cls, form: LPFormulation):
        m, n = form.table.shape

        if np.all(form.rhs >= 0):
            slack = np.eye(m, dtype=form.table.numerator.dtype)
            new_table = FractionArray(
                np.hstack(
                    [
                        form.table.numerator,
                        slack,
                        form.rhs.numerator[..., np.newaxis],
                    ]
                ),
                np.hstack(
                    [
                        form.table.denominator,
                        np.ones_like(slack),
                        form.rhs.denominator[..., np.newaxis],
                    ]
                ),
            )

            new_objective = FractionArray(
                np.hstack(
                    [
                        form.objective.numerator,
                        np.zeros(m, dtype=form.objective.numerator.dtype),
                        [0],
                    ]
                ),
                np.hstack(
                    [
                        form.objective.denominator,
                        np.ones(m, dtype=form.objective.denominator.dtype),
                        [1],
                    ]
                ),
            )

            base = np.hstack([np.zeros(n, dtype=bool), np.ones(m, dtype=bool)])

            row_to_var_index = np.arange(m, dtype=int) + n
            var_index_to_row = np.hstack(
                [np.zeros(n, dtype=int), np.arange(m, dtype=int)]
            )

            return cls(
                table=new_table,
                objective_row=new_objective,
                base=base,
                row_to_var_index=row_to_var_index,
                var_index_to_row=var_index_to_row,
            )
        else:
            raise NotImplementedError("Two phase simplex not implemented.")

    @classmethod
    def instantiate(cls, formulation: LPFormulation):
        """
        Instantiate the simplex algorithm for the given problem formulation.

        Args:
            formulation (LPFormulation): The LP formulation

        Raises:
            ValueError: If the input formulation is invalid.

        Returns:
            (Simplex): An instance of the simplex algorithm for the given
            problem.

        """

        _validate_formulation(formulation)

        if formulation.type == LPFormType.Canonical:
            return cls._from_canonical(formulation)
        else:
            raise NotImplementedError

    def state(self):
        """ Return the value of the objective function and variable values. """

        variables = FractionArray.from_array(
            np.zeros(self.table.shape[1], dtype=int)
        )
        variables[self._row_to_var_index] = self.table[
            self._var_index_to_row[self._row_to_var_index], -1
        ]

        return {"value": self.objective_val, "variables": variables}

    def solve(
        self,
        decision_rule="dantzig",
        degenerate_max_iter=DEFAULT_MAX_ITERATIONS,
    ):
        """
        Start the simplex algorithm.

        Args:
            decision_rule:
            degenerate_max_iter:

        Returns:

        """

        logger = logging.getLogger(__name__)
        logger.info("Starting the simplex algorithm.")
        logger.debug(
            "decision_rule=%s, degenerate_max_iter",
            decision_rule,
            degenerate_max_iter,
        )

        if decision_rule not in DECISION_RULES:
            raise ValueError(
                "Decision rule {} is invalid or "
                "unsupported".format(decision_rule)
            )

        decision_function = DECISION_RULES[decision_rule]

        last_objective = self.objective_val
        iterations_from_last_increase = 0

        iteration_count = 0
        while True:
            logger.debug("Iteration count %d", iteration_count)

            # Check whether the optimal value has been reached
            if self._is_end():
                logger.info("Success")
                return {"result": "success", **self.state()}

            # Determine the entering variable
            entering = decision_function(self)
            logger.debug("Entering:", entering)

            if self._is_unbounded(entering):
                logger.info("The problem is unbounded.")
                return {"result": "unbounded"}

            leaving_row = self._determine_leaving_row(entering)
            logger.debug("Leaving row: %d", leaving_row)

            # Update the table
            update_row = self._get_update_row(entering, leaving_row)

            self._update_table(update_row, entering, leaving_row)
            self._update_objective(update_row, entering)
            self._update_base(entering, leaving_row)

            if self.objective_val > last_objective:
                last_objective = self.objective_val
                iterations_from_last_increase = 0
                continue

            iterations_from_last_increase += 1
            if iterations_from_last_increase >= degenerate_max_iter:
                logger.info("Max iterations reached, probably cycling.")
                return {"result": "cycle"}

            iteration_count += 1

    def _determine_leaving_row(self, entering):
        positive = np.where(self.table[..., entering] > 0)[0]

        bounds = (
            self.table[..., -1][positive] / self.table[..., entering][positive]
        )
        valid = np.where(bounds >= 0)[0]
        leaving_row = positive[valid[bounds[valid].argmin()]]

        return leaving_row

    def _get_update_row(self, entering, leaving_row):
        return self.table[leaving_row] / self.table[leaving_row, entering]

    def _update_objective(self, update_row, entering):
        update = update_row * self._objective_row[entering]
        update[:-1] *= -1

        self._objective_row += update

    def _update_base(self, entering, leaving_row):
        leaving = self._row_to_var_index[leaving_row]
        self._row_to_var_index[leaving_row] = entering

        self._var_index_to_row[entering] = self._var_index_to_row[leaving]

        self.base[entering] = True
        self.base[leaving] = False

    def _update_table(self, update_row, entering, leaving_row):
        self.table[leaving_row] = update_row

        for row in range(self.table.shape[0]):
            if row == leaving_row:
                continue

            self.table[row] -= self.table[row, entering] * update_row

    def _is_end(self):
        return np.all(self.objective_fun[~self.base] < 0)

    def _is_unbounded(self, entering):
        return np.all(self.table[..., entering] < 0)

    def __getattr__(self, attribute):
        if attribute == "objective_val":
            return self._objective_row[-1]
        elif attribute == "objective_fun":
            return self._objective_row[:-1]
        else:
            raise AttributeError(attribute + " not found")


def dantzig(simplex: Simplex):
    nonbasic = np.where(~simplex.base)[0]
    positive = np.where(simplex.objective_fun[nonbasic] > 0)[0]
    entering = nonbasic[
        positive[simplex.objective_fun[nonbasic][positive].argmax()]
    ]

    return entering

def bland(simplex: Simplex):
    pass


DECISION_RULES = {"dantzig": dantzig}
