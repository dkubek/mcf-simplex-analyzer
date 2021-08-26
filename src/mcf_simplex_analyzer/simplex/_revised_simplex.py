# -*- coding: utf-8 -*-
""" The revised simplex method """

import attr
import numpy as np

import mcf_simplex_analyzer.fractionarray as fa
from mcf_simplex_analyzer.fractionarray import FractionArray


def _compute_lu_inplace(matrix: FractionArray):
    m, n = matrix.shape

    assert m == n

    perm = np.arange(n, dtype=np.int64)

    for i in range(n):
        # Determine pivot
        k = i
        while i <= k < n:
            if matrix[k, i] != 0:
                break

            k += 1

        if k == n:
            raise ValueError("Matrix is singular.")

        if k != i:
            perm[[i, k]] = perm[[k, i]]

            tmp = matrix[i].copy()
            matrix[i] = matrix[k]
            matrix[k] = tmp

        nonzero = np.where(matrix[i:, i] != 0)[0] + i
        for j in nonzero[1:]:
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= factor * matrix[i, i:]
            matrix[j, i] = factor

    return perm, matrix


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
class PLUSolver:
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
        return _lu_solve_transposed(self.lu.T, b)[self.perm]
