# -*- coding: utf-8 -*-
""" Sparse FractionArrays """

import attr
import numpy as np

import mcf_simplex_analyzer.fractionarray as fa
from mcf_simplex_analyzer.fractionarray import FractionArray


@attr.s
class FractionSparseArray:
    """ Sparse one-dimensional array of fractions. """

    size = attr.ib()
    indices = attr.ib()
    data: FractionArray = attr.ib()


@attr.s
class FractionCOOMatrix:
    """ Sparse two-dimensional array of fractions. """

    shape = attr.ib()

    row = attr.ib()
    col = attr.ib()

    data: FractionArray = attr.ib()

    def nnz(self):
        return len(self.data)


@attr.s
class FractionCSCMatrix:
    """ Sparse two-dimensional array of fractions. """

    shape = attr.ib()

    indptr = attr.ib()
    indices = attr.ib()

    data: FractionArray = attr.ib()

    def nnz(self):
        return len(self.data)


def coo_to_csc(coo: FractionCOOMatrix, sorted=False):
    m, n = coo.shape

    row_sorted = coo.row
    col_sorted = coo.col
    data = None
    if not sorted:
        asort = np.lexsort((coo.row, coo.col))

        row_sorted = coo.row[asort]
        col_sorted = coo.col[asort]

        data = coo.data[asort]
    else:
        data = coo.data.copy()

    indptr = np.empty(n + 1, dtype=np.int64)
    indices = np.empty(coo.nnz(), dtype=np.int64)

    index = 0
    for col in range(n):
        indptr[col] = index
        while index < len(col_sorted) and col_sorted[index] == col:
            indices[index] = row_sorted[index]
            index += 1
    indptr[n] = index

    return FractionCSCMatrix(
        shape=coo.shape, indptr=indptr, indices=indices, data=data
    )


def csc_to_coo(csc: FractionCSCMatrix):
    raise NotImplementedError("csc_to_coo not implemented")


def csc_to_dense(csc: FractionCSCMatrix):
    dense = fa.zeros(csc.shape)

    for j in range(len(csc.indptr) - 1):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        if end - start == 0:
            continue

        for k in range(start, end):
            i = csc.indices[k]
            dense[i, j] = csc.data[k]

    return dense
