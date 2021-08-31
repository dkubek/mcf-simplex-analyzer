# -*- coding: utf-8 -*-
""" Tests for Simplex. """

import numbers
from typing import List
from enum import Enum

import attr


@attr.s(cmp=False)
class Variable:
    index: int = attr.ib()
    name: str = attr.ib(default="")

    def __add__(self, other):
        """ Returns self + other. """

        return other.add_variable(self)

    def __radd__(self, other):
        """ Returns other + self. """

        return self.__add__(other)

    def __neg__(self):
        """ Returns -self. """

        return LinearCombination([FactorVar(-1, self)])

    def __sub__(self, other):
        """ Returns self - other. """

        return (-other).add_variable(self)

    def __rsub__(self, other):
        """ Returns other - self. """

        return other.sub_variable(self)

    def __mul__(self, other):
        """ Returns self * other. """

        if not isinstance(other, numbers.Number):
            raise ValueError(
                "Other needs to be a number, however it's a {}".format(
                    type(other)
                )
            )

        return LinearCombination([FactorVar(other, self)])

    def __rmul__(self, other):
        """ Returns other * self. """

        return self.__mul__(other)

    def add_variable(self, var: "Variable"):
        return LinearCombination([FactorVar(1, self), FactorVar(1, var)])

    def sub_variable(self, var: "Variable"):
        return LinearCombination([FactorVar(1, self), FactorVar(-1, var)])

    def add_linear_combination(self, comb: "LinearCombination"):
        return comb.add_variable(self)

    def sub_linear_combination(self, comb: "LinearCombination"):
        return (-comb).add_variable(self)

    def __eq__(self, other) -> "Constraint":
        """ Returns self == other. """

        return LinearCombination([FactorVar(1, self)]) == other

    def __le__(self, other) -> "Constraint":
        """ Returns self <= other. """

        return LinearCombination([FactorVar(1, self)]) <= other

    def __ge__(self, other) -> "Constraint":
        """ Returns self >= other. """

        return LinearCombination([FactorVar(1, self)]) <= other

    def __ne__(self, other):
        """ Returns self != other. """

        raise NotImplementedError("Not equal is not a valid constraint.")

    def __lt__(self, other):
        """ Returns self < other. """

        raise NotImplementedError("Less than is not a valid constraint.")

    def __gt__(self, other):
        """ Returns self > other. """

        raise NotImplementedError("Greater than is not a valid constraint.")


@attr.s
class FactorVar:
    factor = attr.ib()
    var: Variable = attr.ib()


@attr.s(cmp=False)
class LinearCombination:
    combination: List[FactorVar] = attr.ib(factory=list)

    def __add__(self, other):
        """ Returns self + other. """

        return other.add_linear_combination(self)

    def __radd__(self, other):
        """ Returns other + self. """

        return self.__add__(other)

    def __neg__(self):
        """ Returns -self. """

        return LinearCombination(
            list(
                map(lambda fv: FactorVar(-fv.factor, fv.var), self.combination)
            )
        )

    def __sub__(self, other):
        """ Returns self - other. """

        return (-other).add_linear_combination(self)

    def __rsub__(self, other):
        """ Returns other - self. """

        return other.sub_linear_combination(self)

    def __mul__(self, other):
        """ Returns self * other. """

        if not isinstance(other, numbers.Number):
            raise ValueError(
                "Other needs to be a number, however it's a {}".format(
                    type(other)
                )
            )

        return LinearCombination(
            list(
                map(
                    lambda fv: FactorVar(other * fv.factor, fv.var),
                    self.combination,
                )
            )
        )

    def __rmul__(self, other):
        """ Returns other * self. """

        return self.__mul__(other)

    def add_variable(self, other: "Variable", factor=1):
        new_combination = self.combination.copy()

        for fv in new_combination:
            if fv.var.index == other.index:
                fv.factor += factor
                return LinearCombination(new_combination)

        new_combination.append(FactorVar(factor, other))
        return LinearCombination(new_combination)

    def sub_variable(self, other: "Variable"):
        return self.add_variable(other, factor=-1)

    def add_linear_combination(self, other: "LinearCombination"):
        self.combination.sort(key=lambda fv: fv.var.index)
        other.combination.sort(key=lambda fv: fv.var.index)

        new_combination = []
        i = 0
        j = 0
        while i < len(self.combination) and j < len(other.combination):
            if self.combination[i].var.index == other.combination[j].var.index:
                new_combination.append(
                    FactorVar(
                        self.combination[i].factor
                        + other.combination[i].factor,
                        self.combination[i].var,
                    )
                )
                continue

            if self.combination[i].var.index < other.combination[j].var.index:
                new_combination.append(self.combination[i])
                i += 1
                continue

            new_combination.append(other.combination[j])
            j += 1

        while i < len(self.combination):
            new_combination.append(self.combination[i])
            i += 1

        while j < len(other.combination):
            new_combination.append(other.combination[j])
            j += 1

        return LinearCombination(new_combination)

    def sub_linear_combination(self, other: "LinearCombination"):
        self.add_linear_combination(-other)

    def _make_constraint(self, inequality_type, rhs):

        if not isinstance(rhs, numbers.Number):
            raise ValueError(
                "Right hand side needs to be a number,"
                "however it is a {}".format(type(rhs))
            )

        return Constraint(
            combination=self,
            type=inequality_type,
            right_hand_side=rhs,
        )

    def __eq__(self, other) -> "Constraint":
        """ Returns self == other. """

        return self._make_constraint(InequalityType.EQ, other)

    def __le__(self, other) -> "Constraint":
        """ Returns self <= other. """

        return self._make_constraint(InequalityType.LE, other)

    def __ge__(self, other) -> "Constraint":
        """ Returns self >= other. """

        return self._make_constraint(InequalityType.GE, other)

    def __ne__(self, other):
        """ Returns self != other. """

        raise NotImplementedError("Not equal is not a valid constraint.")

    def __lt__(self, other):
        """ Returns self < other. """

        raise NotImplementedError("Less than is not a valid constraint.")

    def __gt__(self, other):
        """ Returns self > other. """

        raise NotImplementedError("Greater than is not a valid constraint.")

    def __iter__(self):
        return self.combination.__iter__()


class InequalityType(Enum):
    LE = 0
    GE = 1
    EQ = 2


def negate_inequality_type(inequality_type: InequalityType):
    if inequality_type == InequalityType.EQ:
        return InequalityType.EQ

    if inequality_type == InequalityType.GE:
        return InequalityType.LE

    if inequality_type == InequalityType.LE:
        return InequalityType.GE

    raise NotImplementedError("Inequality type not supported.")


@attr.s
class Constraint:
    combination: LinearCombination = attr.ib()
    right_hand_side = attr.ib()
    type: InequalityType = attr.ib()

    def __neg__(self):
        """ Negate the constraint. """

        return Constraint(
            combination=-self.combination,
            right_hand_side=-self.right_hand_side,
            type=negate_inequality_type(self.type),
        )


@attr.s(kw_only=True)
class LPModel:
    name: str = attr.ib(default="lp_model")
    constraints: List[Constraint] = attr.ib(factory=list)
    objective_function: LinearCombination = attr.ib(factory=LinearCombination)

    variable_no = attr.ib(default=0)

    def new_variable(self, name=""):
        var = Variable(self.variable_no, name=name)
        self.variable_no += 1

        return var

    def export_standard(self):
        raise NotImplemented

    def export_canonical(self):
        raise NotImplemented
