# -*- coding: utf-8 -*-
""" The revised simplex method """

import fractions

from typing import Union

import attr
import numpy as np

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


def _lu_solve(lu: FractionArray, x: FractionArray):
    n = lu.shape[0]

    ans = fa.zeros_like(x)

    # Ly = b
    for i in range(n):
        ans[i] = x[i] - fa.dot(lu[i, :i], ans[:i])

    # Ux = y
    for i in range(n - 1, -1, -1):
        ans[i] -= fa.dot(lu[i, (i + 1) :], ans[(i + 1) :])
        ans[i] /= lu[i, i]

    return ans


def _lu_solve_transposed(lu: FractionArray, x: FractionArray):
    n = lu.shape[0]

    ans = fa.zeros_like(x)

    # U^T y = b
    for i in range(n):
        ans[i] = x[i] - fa.dot(lu[i, :i], ans[:i])
        ans[i] /= lu[i, i]

    # L^t x = y
    for i in range(n - 1, -1, -1):
        ans[i] -= fa.dot(lu[i, (i + 1) :], ans[(i + 1) :])

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
        if self._perm_trans is None:
            self._perm_trans = _inverse_permutation(self.perm)

        return _lu_solve(self.lu, b)[self._perm_trans]

    def btran(self, b: FractionArray):
        return _lu_solve_transposed(self.lu.T, b[self.perm])


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
        tmp = b.copy()
        for solver in self.file:
            tmp = solver.ftran(tmp)

        return tmp

    def btran(self, b: FractionArray):
        tmp = b.copy()
        for solver in reversed(self.file):
            tmp = solver.btran(tmp)

        return tmp


@attr.s
class LPResult:
    status: str = attr.ib()
    value: Union[fractions.Fraction, None] = attr.ib(default=None)


@attr.s(kw_only=True)
class RevisedSimplex:
    table: fas.FractionCSCMatrix = attr.ib()
    objective_function: fa.FractionArray = attr.ib()
    right_hand_side: fa.FractionArray = attr.ib()
    base: np.ndarray = attr.ib()
    is_feasible = attr.ib(default=True)

    _base_indices: np.ndarray = attr.ib(default=None)
    _nonbase_indices: np.ndarray = attr.ib(default=None)
    _base_values: fa.FractionArray = attr.ib(default=None)

    @classmethod
    def _formulate_two_phase(cls, formulation):
        """ Instantiate the simplex algorithm using two phase approach. """

        raise NotImplementedError("two phase not implemented")

    @classmethod
    def instantiate(cls, model: LPModel):

        base_indices = np.empty(len(model.constraints), dtype=np.int64)
        constraints_need_artificial_variables = []

        cols, rows, data = [], [], []
        b = fa.zeros(len(model.constraints))
        for row, constraint in enumerate(model.constraints):
            if constraint.type == InequalityType.GE:
                constraint = -constraint

            if constraint.type == InequalityType.LE:
                slack = model.new_variable(name=f"slack_{row}")
                constraint = (
                    constraint.combination + slack
                ) == constraint.right_hand_side

                if constraint.right_hand_side < 0:
                    constraints_need_artificial_variables.append(row)
                    constraint = -constraint
                else:
                    base_indices[row] = slack.index

            elif constraint.type == InequalityType.EQ:
                constraints_need_artificial_variables.append(row)

            b[row] = constraint.right_hand_side
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
        table = fas.coo_to_csc(table_coo)

        base = np.zeros(model.variable_no, dtype=bool)

        print("base_indices", base_indices)
        artificial_no = len(constraints_need_artificial_variables)
        if artificial_no > 0:
            artificial_table = fas.FractionCOOMatrix(
                shape=(len(model.constraints), artificial_no),
                rows=np.array(
                    constraints_need_artificial_variables, dtype=np.int64
                ),
                cols=np.arange(
                    artificial_no,
                    dtype=np.int64,
                ),
                data=fa.ones(artificial_no),
            )
            artificial_table = fas.coo_to_csc(artificial_table, sorted=True)
            artificial_table = fas.csc_hstack((table, artificial_table))

            base_indices[constraints_need_artificial_variables] = np.arange(
                model.variable_no,
                model.variable_no + artificial_no,
                dtype=np.int64,
            )

            print("base_indices final", base_indices)
            print()

            artificial_base = np.zeros(
                model.variable_no + artificial_no, dtype=bool
            )
            artificial_base[base_indices] = True

            artificial_objective = fa.zeros(model.variable_no + artificial_no)
            artificial_objective[-artificial_no:] = -1

            phase_one = RevisedSimplex(
                table=artificial_table,
                right_hand_side=b,
                objective_function=artificial_objective,
                base=artificial_base,
            )

            ans = phase_one.solve()

            if ans.status != "success" or ans.value > 0:
                phase_one.is_feasible = False
                return phase_one

            if np.any(phase_one.base[-artificial_no:]):
                raise NotImplementedError(
                    "Driving out of the base in two step not implemented"
                )

            base = phase_one.base[:-artificial_no]
        else:
            base[base_indices] = True

        c = fa.zeros(shape[1])
        for fv in model.objective_function:
            c[fv.var.index] = fv.factor

        return RevisedSimplex(
            table=table, right_hand_side=b, objective_function=c, base=base
        )

    def solve(
        self,
        refactorization_period=25,
    ):

        if not self.is_feasible:
            return LPResult(status="infeasible")

        m = self.table.shape[0]
        base_table = fa.empty((m, m))

        eta_file = self._refactorize_base(base_table)

        # Compute the basic feasible solution
        self._base_values = eta_file.ftran(self.right_hand_side)
        print("p: ", self._base_values)

        iteration_count = 0
        while True:
            print(iteration_count)
            if len(eta_file.file) > refactorization_period:
                eta_file = self._refactorize_base(base_table)

                # Compute the basic feasible solution
                self._base_values = eta_file.ftran(self.right_hand_side)
                print("p: ", self._base_values)

            y = eta_file.btran(self.objective_function[self._base_indices])
            print("y:", y)

            # Compute objective function
            obj = self._compute_objective(y)
            print("obj:", obj)

            # Determine entering
            entering_index = self._determine_entering(obj)

            if entering_index is None:
                print(self._base_values)
                return LPResult(
                    status="success",
                    value=fa.dot(
                        self._base_values,
                        self.objective_function[self._base_indices],
                    ),
                )

            entering = self._nonbase_indices[entering_index]
            print("entering_index", entering_index, "entering", entering)

            # Determine entering column
            # TODO: Return vector when selecting one column
            a = fas.csc_to_dense(
                fas.csc_select_columns(self.table, [entering])
            )[:, 0]
            print("a", a)

            print(eta_file)
            d = eta_file.ftran(a)
            print("d", d)

            if np.all(d <= 0):
                print("Unbounded")
                return LPResult(status="unbounded")

            # Determine leaving
            leaving_index, min_bound = self._determine_leaving(d)

            leaving = self._base_indices[leaving_index]
            print("leaving_index", leaving_index, "leaving", leaving)

            # Change basis
            self._update_basis(
                entering, entering_index, leaving, leaving_index
            )

            self._base_values -= min_bound * d
            self._base_values[leaving_index] = min_bound
            # self._base_values = eta_file.ftran(self.right_hand_side)
            print("p", self._base_values)

            # Update eta file
            eta = EtaSolver(index=leaving_index, column=d)
            eta_file.extend(eta)

            iteration_count += 1

            print()

    def _update_basis(self, entering, entering_index, leaving, leaving_index):
        self.base[entering] = True
        self.base[leaving] = False
        print("base:", self.base)

        self._base_indices[leaving_index] = entering
        self._nonbase_indices[entering_index] = leaving
        print("base_ind", self._base_indices)
        print("nonbase_ind", self._nonbase_indices)

    def _determine_leaving(self, d):
        positive = np.where(d > 0)[0]

        bounds = self._base_values[positive] / d[positive]
        valid = np.where(bounds >= 0)[0]
        arg_min_bound = valid[bounds[valid].argmin()]

        min_bound = bounds[arg_min_bound]
        leaving_index = positive[arg_min_bound]

        return leaving_index, min_bound

    def _determine_entering(self, obj):
        valid_entering = np.where(obj > 0)[0]

        if valid_entering.size == 0:
            return None

        entering_index = valid_entering[obj[valid_entering].argmax()]

        return entering_index

    def _refactorize_base(self, base_table):

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

        return eta_file

    def _compute_objective(self, y):

        # TODO: empty?
        tmp = fa.zeros(len(self._nonbase_indices))
        for i, col in enumerate(self._nonbase_indices):
            start, end = self.table.indptr[col], self.table.indptr[col + 1]
            tmp[i] = sum(
                factor * y[row]
                for row, factor in zip(
                    self.table.indices[start:end], self.table.data[start:end]
                )
            )

        return self.objective_function[self._nonbase_indices] - tmp
