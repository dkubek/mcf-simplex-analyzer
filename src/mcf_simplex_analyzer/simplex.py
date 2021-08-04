from enum import Enum

import attr
import numpy as np

from .fractionarray import FractionArray


def dantzig_entering(z, base):
    positive = np.where(~base)[0]
    entering = positive[z[positive].argmax()]

    return entering


def dantzig_leaving(T, entering):
    positive = np.where(T[..., entering].numerator >= 0)[0]

    bounds = T[..., -1][positive] / T[..., entering][positive]
    print("Bounds:", bounds)
    valid = np.where(bounds.numerator > 0)[0]
    leaving_row = positive[valid[bounds[valid].argmin()]]

    return leaving_row


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


@attr.s(kw_only=True)
class Simplex:
    table = attr.ib()
    objective = attr.ib()

    _base = attr.ib()
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
                objective=new_objective,
                base=base,
                row_to_var_index=row_to_var_index,
                var_index_to_row=var_index_to_row,
            )
        else:
            raise NotImplementedError("Two phase simplex not implemented.")

    @classmethod
    def from_formulation(cls, formulation: LPFormulation):
        if formulation.type == LPFormType.Canonical:
            return cls._from_canonical(formulation)
        else:
            raise NotImplemented

    def _get_update_row(self, entering, leaving_row):
        return self.table[leaving_row] / self.table[leaving_row, entering]

    def _update_objective(self, update_row, entering, leaving_row):
        update = update_row * self.objective[entering]
        update[:-1] *= -1

        self.objective += update

    def _update_base(self, entering, leaving_row):
        leaving = self._row_to_var_index[leaving_row]
        self._row_to_var_index[leaving_row] = entering

        self._var_index_to_row[entering] = self._var_index_to_row[leaving]

        self._base[entering] = True

    def _update_table(self, update_row, entering, leaving_row):
        self.table[leaving_row] = update_row

        for row in range(self.table.shape[0]):
            if row == leaving_row:
                continue

            self.table[row] -= self.table[row, entering] * update_row

    def _is_end(self):
        return np.all(self.objective[:-1][~self._base] < 0)

    def solve(self):
        while True:
            print(self.objective)
            print("Objective function:", self.objective)
            for var in self._row_to_var_index:
                print(f"x_{var}:", self.table[self._var_index_to_row[var], -1])

            # Check end
            if self._is_end():
                print("End")
                break

            # Determine entering
            entering = dantzig_entering(self.objective[:-1], self._base)
            print("Entering:", entering)

            # Check unbounded
            if np.all(self.table[..., entering] < 0):
                print("Unbounded")
                break

            # Determine leaving
            leaving_row = dantzig_leaving(self.table, entering)
            print("Leaving row:", leaving_row)

            # Update
            update_row = self._get_update_row(entering, leaving_row)

            self._update_table(update_row, entering, leaving_row)
            self._update_objective(update_row, entering, leaving_row)
            self._update_base(entering, leaving_row)
