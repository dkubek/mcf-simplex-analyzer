# cython: linetrace=True
# -*- coding: utf-8 -*-
""" The revised simplex method """

import time
import logging
import fractions

from typing import Union

import attr
import numpy as np
import cython

from gmpy2 import mpq

from ._model import LPModel, InequalityType
import mcf_simplex_analyzer.fractionarray as fa
import mcf_simplex_analyzer.fractionarray.sparse as fas
from mcf_simplex_analyzer.fractionarray import FractionArray


def lu_factor(matrix: FractionArray, inplace=False):
    m, n = matrix.shape

    assert m == n

    if inplace:
        out = matrix
    else:
        out = matrix.copy()

    perm = np.arange(n, dtype=np.int64)
    for i in range(n):
        # Determine pivot
        k = i
        while i <= k < n:
            if out[k, i] != 0:
                break

            k += 1

        if k == n:
            raise ValueError("Matrix is singular.")

        if k != i:
            perm[[i, k]] = perm[[k, i]]

            tmp = out[i].copy()
            out[i] = out[k]
            out[k] = tmp

        nonzero = np.where(out[i:, i] != 0)[0] + i
        for j in nonzero[1:]:
            factor = out[j, i] / out[i, i]
            out[j, i:] -= factor * out[i, i:]
            out[j, i] = factor

    return perm, out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _mpq_dot(object[:] fa, object[:] fb):
    """ Dot product of two arrays. """

    cdef:
        size_t i, n

    assert len(fa) == len(fb)

    n = len(fa)

    ans = mpq()
    for i in range(n):
        ans += fa[i] * fb[i]

    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
def _lu_solve(object[:, :] lu, object[:] x):
    cdef:
        size_t n, i

    n = lu.shape[0]

    ans = np.empty_like(x)

    # Ly = b
    for i in range(n):
        ans[i] = x[i] - _mpq_dot(lu[i, :i], ans[:i])

    # Ux = y
    for i in range(n - 1, -1, -1):
        ans[i] = ans[i] - _mpq_dot(lu[i, (i + 1) :], ans[(i + 1) :])
        ans[i] = ans[i] / lu[i, i]

    return ans


@cython.boundscheck(False)
@cython.wraparound(False)
def _lu_solve_transposed(object[:, :] lu, object[:] x):
    cdef:
        size_t n, i

    n = lu.shape[0]

    ans = np.empty_like(x)

    # U^T y = b
    for i in range(n):
        ans[i] = x[i] - _mpq_dot(lu[i, :i], ans[:i])
        ans[i] = ans[i] / lu[i, i]

    # L^t x = y
    for i in range(n - 1, -1, -1):
        ans[i] = ans[i] - _mpq_dot(lu[i, (i + 1) :], ans[(i + 1) :])

    return ans


def _inverse_permutation(perm: np.ndarray):
    perm_inverse = np.empty_like(perm)

    perm_inverse[perm] = perm

    return perm_inverse


@attr.s
class Solver:
    def ftran(self, b: FractionArray):
        raise NotImplementedError(
            "ftran needs to be implemented in a subclass"
        )

    def btran(self, b: FractionArray):
        raise NotImplementedError(
            "btran needs to be implemented in a subclass"
        )


@attr.s
class PLUSolver(Solver):
    lu: FractionArray = attr.ib()
    perm: np.ndarray = attr.ib()
    _perm_trans: np.ndarray = attr.ib(default=None)

    @lu.validator
    def check_square(self, attribute, value):
        m, n = self.lu.shape

        if m != n:
            raise ValueError("matrix must be square")

    def ftran(self, b: FractionArray):
        return FractionArray(_lu_solve(self.lu.data, b.data[self.perm]))

    def btran(self, b: FractionArray):
        if self._perm_trans is None:
            self._perm_trans = _inverse_permutation(self.perm)

        return FractionArray(
            _lu_solve_transposed(self.lu.T.data, b.data[self._perm_trans])
        )


@attr.s
class EtaSolver(Solver):
    index: np.int64 = attr.ib()
    column: FractionArray = attr.ib()

    def ftran(self, b: FractionArray):
        factor = b[self.index] / self.column[self.index]
        b -= factor * self.column
        b[self.index] = factor

        return b

    def btran(self, b: FractionArray):
        b[self.index] -= fa.dot(
            self.column[: self.index], b[: self.index]
        ) + fa.dot(self.column[self.index + 1 :], b[self.index + 1 :])
        b[self.index] /= self.column[self.index]

        return b


@attr.s
class EtaFile:
    file = attr.ib(factory=list)

    def extend(self, solver: Solver):
        self.file.append(solver)

    def ftran(self, b: FractionArray):
        tic = time.perf_counter()

        tmp = b.copy()
        for solver in self.file:
            tmp = solver.ftran(tmp)

        toc = time.perf_counter()

        return tmp

    def btran(self, b: FractionArray):
        tic = time.perf_counter()

        tmp = b.copy()
        for solver in reversed(self.file):
            tmp = solver.btran(tmp)

        toc = time.perf_counter()

        return tmp


@attr.s
class LPResult:
    status: str = attr.ib()
    value: Union[fractions.Fraction, None] = attr.ib(default=None)


def _is_unbounded(d):
    return np.all(d <= 0)


def _determine_leaving(p, d, all_possible=False):
    positive = np.where(d > 0)[0]

    bounds = p[positive] / d[positive]
    valid = np.where(bounds >= 0)[0]
    min_bound = bounds[valid].min()

    arg_min_bound = valid[np.where(bounds == min_bound)[0]]
    leaving_index = positive[arg_min_bound]
    if all_possible:
        return leaving_index, min_bound

    return leaving_index[0], min_bound


def _determine_entering(obj):
    valid_entering = np.where(obj > 0)[0]

    if valid_entering.size == 0:
        return None

    entering_index = valid_entering[obj[valid_entering].argmax()]

    return entering_index


def _construct_lp_formulation_data(model, need_artificial, base_indices):

    right_hand_side, table_coo = _collect_sparse_table_from_constraints(
        model, base_indices, need_artificial
    )
    table = fas.coo_to_csc(table_coo)

    objective = fa.zeros(table.shape[1])
    for fv in model.objective_function:
        objective[fv.var.index] = fv.factor

    return table, objective, right_hand_side


def _collect_sparse_table_from_constraints(
        model,
        base_indices,
        need_artificial_variables
):

    cols, rows, data = [], [], []
    right_hand_side = fa.zeros(len(model.constraints))
    for row, constraint in enumerate(model.constraints):
        if constraint.type == InequalityType.GE:
            constraint = -constraint

        if constraint.type == InequalityType.LE:
            slack = model.new_variable(name=f"slack_{row}")
            constraint = (
                    constraint.combination + slack
                    == constraint.right_hand_side
            )

            if constraint.right_hand_side < 0:
                need_artificial_variables.append(row)
                constraint = -constraint
            else:
                base_indices[row] = slack.index
        elif constraint.type == InequalityType.EQ:
            if constraint.right_hand_side < 0:
                constraint = -constraint

            need_artificial_variables.append(row)

        right_hand_side[row] = constraint.right_hand_side
        for fv in constraint.combination:
            cols.append(fv.var.index)
            rows.append(row)
            data.append(fv.factor)

    shape = (
        len(model.constraints),
        model.variable_no,
    )
    table_coo = fas.FractionCOOMatrix(
        shape=shape,
        cols=np.array(cols, dtype=np.int64),
        rows=np.array(rows, dtype=np.int64),
        data=fa.FractionArray.from_array(data),
    )

    return right_hand_side, table_coo


def _formulate_phase_one(
        model,
        table,
        right_hand_side,
        base_indices,
        need_artificial_variables
):

    artificial_no = len(need_artificial_variables)

    artificial_table = _construct_artificial_table(
        model, table, need_artificial_variables
    )

    artificial_base = _construct_artificial_base(
        model, base_indices, need_artificial_variables
    )

    artificial_objective = _construct_artificial_objective(model, artificial_no)

    phase_one = RevisedSimplex(
        table=artificial_table,
        right_hand_side=right_hand_side,
        objective_function=artificial_objective,
        base=artificial_base,
    )

    return phase_one


def _construct_artificial_objective(model, artificial_no):

    artificial_objective = fa.zeros(model.variable_no + artificial_no)
    artificial_objective[-artificial_no:] = -1

    return artificial_objective


def _construct_artificial_base(
        model,
        base_indices,
        need_artificial_variables
):
    artificial_no = len(need_artificial_variables)

    base_indices[need_artificial_variables] = np.arange(
        model.variable_no,
        model.variable_no + artificial_no,
        dtype=np.int64,
    )
    artificial_base = np.zeros(
        model.variable_no + artificial_no, dtype=bool
    )
    artificial_base[base_indices] = True

    return artificial_base


def _construct_artificial_table(
        model,
        table,
        need_artificial_variables,
):

    artificial_no = len(need_artificial_variables)

    artificial_table = fas.FractionCOOMatrix(
        shape=(len(model.constraints), artificial_no),
        rows=np.array(
            need_artificial_variables, dtype=np.int64
        ),
        cols=np.arange(
            artificial_no,
            dtype=np.int64,
        ),
        data=fa.ones(artificial_no),
    )
    artificial_table = fas.coo_to_csc(artificial_table, sorted=True)
    artificial_table = fas.csc_hstack((table, artificial_table))

    return artificial_table


@attr.s(kw_only=True)
class RevisedSimplex:
    table: fas.FractionCSCMatrix = attr.ib()
    objective_function: fa.FractionArray = attr.ib()
    right_hand_side: fa.FractionArray = attr.ib()
    base: np.ndarray = attr.ib()

    is_feasible = attr.ib(default=True)
    eta_file : EtaFile = attr.ib(default=None)

    _base_indices: np.ndarray = attr.ib(default=None)
    _nonbase_indices: np.ndarray = attr.ib(default=None)
    _base_values: fa.FractionArray = attr.ib(default=None)

    @classmethod
    def instantiate(cls, model: LPModel):

        base_indices = np.empty(len(model.constraints), dtype=np.int64)
        need_artificial_variables = []

        table, objective, right_hand_side = _construct_lp_formulation_data(
            model,
            need_artificial_variables,
            base_indices
        )
        base = np.zeros(model.variable_no, dtype=bool)

        artificial_no = len(need_artificial_variables)
        if artificial_no > 0:
            phase_one = _formulate_phase_one(
                model,
                table,
                right_hand_side,
                base_indices,
                need_artificial_variables
            )

            ans = phase_one.solve()

            if ans.status != "success" or ans.value < 0:
                phase_one.is_feasible = False
                return phase_one

            real_variable_no = phase_one.table.shape[1] - artificial_no
            if np.any(phase_one.base[-artificial_no:]):
                print("DRIVING OUT OF THE BASIS")

                # Initialize base table
                m = phase_one.table.shape[0]
                base_table = fa.empty((m, m))
                eta_file = phase_one._refactorize_base(base_table)

                S = np.where(phase_one._base_indices >= real_variable_no)[0]
                print(S)

                rows_to_remove = []
                for leaving_index in S:
                    leaving = phase_one._base_indices[leaving_index]
                    print("leaving_index", leaving_index, "leaving", leaving)

                    e = fa.zeros(m)
                    e[leaving_index] = 1
                    print(e)

                    r = eta_file.btran(e)
                    print(r)

                    n = len(phase_one._nonbase_indices)
                    entering_index, entering = None, None
                    for i in range(n):
                        col = phase_one._nonbase_indices[i]

                        start = phase_one.table.indptr[col]
                        end = phase_one.table.indptr[col + 1]

                        ans = mpq()
                        for j in range(start, end):
                            row = phase_one.table.indices[j]
                            ans += r[row] * phase_one.table.data[j]

                        if ans != 0:
                            entering_index = i
                            entering = col
                            break

                    if entering_index is None:
                        rows_to_remove.append(leaving_index)
                        continue

                    phase_one._update_basis(
                        entering, entering_index, leaving, leaving_index
                    )

                    # Update eta file
                    fas.csc_to_dense(
                        fas.csc_select_columns(
                            phase_one.table, phase_one._base_indices
                        ),
                        out=base_table,
                    )
                    # Factorize the basis
                    perm, lu = lu_factor(base_table, inplace=True)
                    #print(lu)

                    # Initialize the eta file
                    eta_file = EtaFile()
                    eta_file.extend(PLUSolver(lu=lu, perm=perm))

                print(rows_to_remove)

                # Remove rows
                table = fas.csc_remove_rows(table, rows_to_remove)
                right_hand_side = np.delete(right_hand_side, rows_to_remove)

            base = phase_one.base[:-artificial_no]
        else:
            base[base_indices] = True

        second_phase = RevisedSimplex(
            table=table,
            right_hand_side=right_hand_side,
            objective_function=objective,
            base=base,
        )

        return second_phase

    def solve(self, refactorization_period=25):

        logger = logging.getLogger(__name__)
        logger.info("Starting the revised simplex algorithm.")
        logger.debug("refactorization_period=%d", refactorization_period)

        if not self.is_feasible:
            logger.info("Problem is infeasible")
            return LPResult(status="infeasible")

        m = self.table.shape[0]
        base_table = fa.empty((m, m))

        self.eta_file = self._refactorize_base(base_table)

        iteration_count = 0
        while True:
            print(iteration_count)
            if iteration_count % 100 == 0:
                logger.info("\nIteration: %d", iteration_count)
            else:
                logger.debug("\nIteration: %d", iteration_count)

            if len(self.eta_file.file) > refactorization_period:
                self.eta_file = self._refactorize_base(base_table)

            if np.any(self._base_values < 0):
                raise ValueError("Variable negative")

            # Compute objective function
            y = self.eta_file.btran(self.objective_function[self._base_indices])
            logger.debug("y= %s", y)

            nonbasic_objective = self._compute_objective(y)
            logger.debug("nonbasic_objective= %s", nonbasic_objective)

            entering_index, leaving_index, min_bound = self._greatest_increase(
                nonbasic_objective
            )
            logger.debug("entering_index= %s", entering_index)

            if entering_index is None:
                logger.info("Optimal solution found.")

                optimum = fa.dot(
                    self._base_values,
                    self.objective_function[self._base_indices],
                )

                logger.info("base values: %s", self._base_values)
                logger.info("optimum: %s", optimum)

                return LPResult(status="success", value=optimum)

            entering = self._nonbase_indices[entering_index]
            logger.debug("Entering= %d", entering)

            # Determine entering column
            entering_column = fas.csc_to_dense(
                fas.csc_select_columns(self.table, [entering])
            )[:, 0]
            logger.debug("Entering column= %s", entering_column)

            d = self.eta_file.ftran(entering_column)
            logger.debug("d= %s", d)

            if _is_unbounded(d):
                logger.info("Problem is unbounded")
                return LPResult(status="unbounded")

            # Determine leaving
            leaving = self._base_indices[leaving_index]
            logger.debug("leaving_index= %d", leaving_index)
            logger.debug("leaving= %d", leaving)
            logger.debug("min_bound= %s", min_bound)

            self._update_basis(
                entering, entering_index, leaving, leaving_index
            )

            # Update the eta file
            eta = EtaSolver(index=leaving_index, column=d)
            self.eta_file.extend(eta)

            # Update optimal solution
            self._base_values -= min_bound * d
            self._base_values[leaving_index] = min_bound
            #self._base_values = self.eta_file.ftran(self.right_hand_side)
            optimum = fa.dot(
                self._base_values,
                self.objective_function[self._base_indices],
            )
            print(optimum)
            print(leaving, entering)

            logger.info("base values: %s", self._base_values)
            logger.info("optimum: %s", optimum)
            logger.debug(
                "Solution for the current basis: %s ", self._base_values
            )

            iteration_count += 1

    def _update_basis(self, entering, entering_index, leaving, leaving_index):

        logger = logging.getLogger(__name__)
        logger.debug("Updating base")

        self.base[entering] = True
        self.base[leaving] = False
        logger.debug("base= %s", self.base)

        self._base_indices[leaving_index] = entering
        self._nonbase_indices[entering_index] = leaving
        logger.debug("base_indices= %s", self._base_indices)
        logger.debug("nonbase_indices= %s", self._nonbase_indices)

    def _refactorize_base(self, base_table):

        logger = logging.getLogger(__name__)
        logger.info("Refactorizing base...")
        tic = time.perf_counter()

        self._base_indices = np.where(self.base)[0]
        self._nonbase_indices = np.where(~self.base)[0]

        fas.csc_to_dense(
            fas.csc_select_columns(self.table, self._base_indices),
            out=base_table,
        )

        # Factorize the basis
        perm, lu = lu_factor(base_table, inplace=True)

        # Initialize the eta file
        eta_file = EtaFile()
        eta_file.extend(PLUSolver(lu=lu, perm=perm))

        toc = time.perf_counter()
        logger.info("Base refactorized. Time: %s", toc - tic)

        # Compute the basic feasible solution
        self._base_values = eta_file.ftran(self.right_hand_side)

        return eta_file

    def _compute_objective(self, y):
        cdef:
            size_t i, j, n, start, end

        n = len(self._nonbase_indices)
        tmp = fa.empty(n)

        for i in range(n):
            col = self._nonbase_indices[i]
            start, end = self.table.indptr[col], self.table.indptr[col + 1]

            ans = mpq()
            for j in range(start, end):
                row = self.table.indices[j]
                ans += y[row] * self.table.data[j]
            tmp[i] = ans

        return self.objective_function[self._nonbase_indices] - tmp

    def _greatest_increase(self, obj):
        entering_index, leaving_index, min_bound = None, None, None
        for i, value in enumerate(obj.data):
            if value <= 0:
                continue

            entering = self._nonbase_indices[i]

            # Determine entering column
            entering_column = fas.csc_to_dense(
                fas.csc_select_columns(self.table, [entering])
            )[:, 0]

            d = self.eta_file.ftran(entering_column)

            if _is_unbounded(d):
                print(entering_column)
                print(entering, value, d)
                return i, leaving_index, min_bound

            leaving_indices, min_bound = _determine_leaving(
                self._base_values, d, all_possible=True
            )

            best = None
            new_base_values = self._base_values - min_bound * d
            for j in leaving_indices:
                tmp = new_base_values.copy()
                tmp[j] = min_bound

                new_base_indices = self._base_indices.copy()
                new_base_indices[j] = entering

                optimum = fa.dot(
                    tmp,
                    self.objective_function[new_base_indices],
                )

                if best is None or optimum > best:
                    best = optimum
                    leaving_index = j
                    entering_index = i

        return entering_index, leaving_index, min_bound
