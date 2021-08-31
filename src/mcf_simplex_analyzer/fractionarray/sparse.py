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

    rows = attr.ib()
    cols = attr.ib()

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

    row_sorted = coo.rows
    col_sorted = coo.cols
    data = None
    if not sorted:
        asort = np.lexsort((coo.rows, coo.cols))

        row_sorted = coo.rows[asort]
        col_sorted = coo.cols[asort]

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


def csc_to_dense(csc: FractionCSCMatrix, out: FractionArray = None):

    if out is not None:
        if out.shape != csc.shape:
            raise ValueError(
                "The given output array does not "
                "fit the output shape {}".format(out.shape)
            )

        out.numerator.fill(0)
        out.denominator.fill(1)

    if out is None:
        out = fa.zeros(csc.shape)

    for j in range(len(csc.indptr) - 1):
        start, end = csc.indptr[j], csc.indptr[j + 1]
        if end - start == 0:
            continue

        for k in range(start, end):
            i = csc.indices[k]
            out[i, j] = csc.data[k]

    return out


def dense_to_csc(dense: FractionArray):
    m, n = dense.shape

    indptr = np.empty(n + 1, dtype=np.int64)
    indices = []
    data = []

    index = 0
    for j in range(n):
        indptr[j] = index
        nonzero = np.where(dense[:, j] != 0)[0]
        for i in nonzero:
            indices.append(i)
            data.append(dense[i, j])
            index += 1

    indptr[n] = index

    return FractionCSCMatrix(
        shape=dense.shape,
        indptr=indptr,
        indices=np.array(indices, dtype=np.int64),
        data=FractionArray.from_array(data),
    )


def csc_hstack(seq):
    if len(seq) == 0:
        raise ValueError("Need at least one array to concatenate")

    rows = seq[0].shape[0]
    if not all(map(lambda arr: arr.shape[0] == rows, seq)):
        raise ValueError("Not all arrays have the same number of rows")
    cols = sum(map(lambda arr: arr.shape[1], seq))

    nnzs = list(map(lambda arr: arr.nnz(), seq))
    cumsum = np.cumsum(nnzs)

    indptr = [seq[0].indptr[:-1]]
    indptr.extend(
        list(
            map(
                lambda pair: pair[0][:-1] + pair[1],
                zip(map(lambda arr: arr.indptr, seq[1:]), cumsum),
            )
        )
    )
    indptr.append([cumsum[-1]])
    indptr = np.hstack(indptr)

    indices = np.hstack(list(map(lambda arr: arr.indices, seq)))
    data = fa.hstack(list(map(lambda arr: arr.data, seq)))

    return FractionCSCMatrix(
        shape=(rows, cols), indptr=indptr, indices=indices, data=data
    )


def csc_select_columns(csc: FractionCSCMatrix, columns):
    m, _ = csc.shape
    n = len(columns)

    indptr = np.empty(n + 1, dtype=np.int64)
    index = 0
    for i, col in enumerate(columns):
        indptr[i] = index
        index += csc.indptr[col + 1] - csc.indptr[col]
    indptr[n] = index

    print(indptr)

    indices = np.empty(index, dtype=np.int64)
    data = fa.empty(index)

    index = 0
    for i, col in enumerate(columns):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        diff = end - start
        indices[index : index + diff] = csc.indices[start:end]
        data[index : index + diff] = csc.data[start:end]
        index += diff

    print(indices)
    print(data)

    return FractionCSCMatrix(
        shape=(m, n), indptr=indptr, indices=indices, data=data
    )
