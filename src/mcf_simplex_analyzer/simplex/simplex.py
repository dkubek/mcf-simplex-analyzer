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

import mcf_simplex_analyzer.fractionarray as fa
from mcf_simplex_analyzer.fractionarray import FractionArray
import mcf_simplex_analyzer.simplex as s

DEFAULT_MAX_ITERATIONS = 1e3


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

    _non_slack_variables = attr.ib(default=None)

    is_feasible = attr.ib(default=True)

    @classmethod
    def _from_canonical(cls, formulation: LPFormulation):
        """
        Instantiate the simplex algorithm from canonical formulation.

        Args:
            formulation (LPFormulation): The formulation of the problem

        Returns:
            Simplex: Instance of the simplex algorithm ready for solving.

        """

        is_two_phase = np.any(formulation.rhs < 0)

        if is_two_phase:
            return cls._formulate_two_phase(formulation)

        return cls._formulate_direct(formulation)

    @classmethod
    def _formulate_direct(cls, formulation):
        """
        Instantiate the simplex algorithm deriving initial solution
        directly

        """

        m, n = formulation.table.shape

        slack = FractionArray.from_array(
            np.eye(m, dtype=formulation.table.numerator.dtype)
        )
        new_table = fa.hstack(
            (formulation.table, slack, formulation.rhs[..., fa.newaxis])
        )

        slack_objective = FractionArray.from_array(
            np.zeros(m, dtype=formulation.objective.numerator.dtype)
        )
        objective_value = FractionArray.from_array([0])
        new_objective = fa.hstack(
            [formulation.objective, slack_objective, objective_value]
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
            non_slack_variables=n,
        )

    @classmethod
    def _formulate_two_phase(cls, formulation):
        """ Instantiate the simplex algorithm using two phase approach. """

        m, n = formulation.table.shape

        # Create the table for the first phase
        slack = FractionArray.from_array(
            np.eye(m, dtype=formulation.table.numerator.dtype)
        )
        aux_variable = FractionArray.from_array(
            -np.ones(m, dtype=formulation.table.numerator.dtype).reshape(m, 1)
        )
        table = fa.hstack(
            (
                aux_variable,
                formulation.table,
                slack,
                formulation.rhs[..., fa.newaxis],
            )
        )

        # Create the objective for the first phase
        slack_objective = FractionArray.from_array(
            np.zeros(m, dtype=formulation.objective.numerator.dtype)
        )
        objective_value = FractionArray.from_array([0])
        aux_objective = FractionArray.from_array(
            np.zeros(n + 1, dtype=formulation.table.numerator.dtype),
        )
        aux_objective[0] = -1
        objective = fa.hstack(
            (
                aux_objective,
                slack_objective,
                objective_value,
            )
        )

        base = np.hstack([np.zeros(n + 1, dtype=bool), np.ones(m, dtype=bool)])

        row_to_var_index = np.arange(m, dtype=int) + (n + 1)
        var_index_to_row = np.hstack(
            [np.zeros(n + 1, dtype=int), np.arange(m, dtype=int)]
        )

        phase_one = cls(
            table=table,
            objective_row=objective,
            base=base,
            row_to_var_index=row_to_var_index,
            var_index_to_row=var_index_to_row,
            non_slack_variables=n + 1,
        )

        # Try to find a feasible solution
        leaving_row = phase_one.table[..., -1].argmin()
        phase_one._pivot(0, leaving_row)
        phase_one.solve(decision_rule="_aux")

        # If the auxiliary variable is basic, the problem is infeasible
        if phase_one.base[0]:
            phase_one.is_feasible = False
            return phase_one

        # Formulate phase two
        phase_two_table = phase_one.table[..., 1:]
        phase_two_row_to_var_index = phase_one._row_to_var_index - 1
        phase_two_var_index_to_row = phase_one._var_index_to_row[1:]
        phase_two_base = base[1:]

        phase_two_objective = FractionArray.from_array(
            np.zeros(m + n + 1, dtype=formulation.objective.numerator.dtype)
        )
        for var_index, coefficient in enumerate(formulation.objective):
            if coefficient == 0:
                continue

            if phase_two_base[var_index]:
                row_index = phase_two_var_index_to_row[var_index]

                assert phase_two_table[row_index, var_index] == 1

                phase_two_objective[:var_index] -= (
                    coefficient * phase_two_table[row_index, :var_index]
                )
                phase_two_objective[(var_index + 1) :] -= (
                    coefficient * phase_two_table[row_index, (var_index + 1) :]
                )
                continue

            phase_two_objective[var_index] += coefficient

        phase_two_objective[-1] *= -1

        return cls(
            table=phase_two_table,
            objective_row=phase_two_objective,
            base=phase_two_base,
            row_to_var_index=phase_two_row_to_var_index,
            var_index_to_row=phase_two_var_index_to_row,
            non_slack_variables=n,
        )

    @classmethod
    def _from_standard(cls, formulation: LPFormulation):
        """
        Instantiate the simplex algorithm from standard formulation.

        Args:
            formulation (LPFormulation): The formulation of the problem

        Returns:
            Simplex: Instance of the simplex algorithm ready for solving.

        """

        new_table = fa.vstack((formulation.table, -formulation.table))
        new_rhs = fa.hstack((formulation.rhs, -formulation.rhs))

        return cls._from_canonical(
            LPFormulation(
                type=LPFormType.Canonical,
                table=new_table,
                rhs=new_rhs,
                objective=formulation.objective,
            )
        )

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
        elif formulation.type == LPFormType.Standard:
            return cls._from_standard(formulation)
        else:
            raise NotImplementedError

    def state(self, slack=False):
        """ Return the value of the objective function and variable values. """

        variables = FractionArray.from_array(
            np.zeros(self.table.shape[1], dtype=int)
        )
        variables[self._row_to_var_index] = self.table[
            self._var_index_to_row[self._row_to_var_index], -1
        ]

        if not slack and self._non_slack_variables is not None:
            variables = variables[: self._non_slack_variables]

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

        if not self.is_feasible:
            logger.info("Problem is infeasible")
            return {"result": "infeasible"}

        if decision_rule not in s.DECISION_RULES:
            raise ValueError(
                "Decision rule {} is invalid or "
                "unsupported".format(decision_rule)
            )

        decision_rule = s.DECISION_RULES[decision_rule]

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
            entering = decision_rule.entering(self)
            if entering is None:
                logger.info("Success, no entering variable")
                return {"result": "success", **self.state()}

            logger.debug("Entering: %d", entering)

            if self._is_unbounded(entering):
                logger.info("The problem is unbounded.")
                return {"result": "unbounded"}

            # Determine leaving row
            leaving_row = decision_rule.leaving_row(entering, self)
            logger.debug(
                "Leaving row: %d, only_positive: %s",
                leaving_row,
            )

            self._pivot(entering, leaving_row)

            iteration_count += 1

            if self.objective_val > last_objective:
                last_objective = self.objective_val
                iterations_from_last_increase = 0
                continue

            iterations_from_last_increase += 1
            if iterations_from_last_increase >= degenerate_max_iter:
                logger.info("Max iterations reached, probably cycling.")
                return {"result": "cycle"}

    def _pivot(self, entering, leaving_row):
        """
        Pivot the table through the given entering variable and leaving row.

        """

        update_row = self._get_update_row(entering, leaving_row)
        self._update_table(update_row, entering, leaving_row)
        self._update_objective(update_row, entering)
        self._update_base(entering, leaving_row)

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
        return np.all(self.table[..., entering] <= 0)

    def __getattr__(self, attribute):
        if attribute == "objective_val":
            return self._objective_row[-1]
        elif attribute == "objective_fun":
            return self._objective_row[:-1]
        else:
            raise AttributeError(attribute + " not found")
