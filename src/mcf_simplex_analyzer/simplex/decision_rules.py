import numpy as np
from mcf_simplex_analyzer.simplex._standard_simplex import StandardSimplex


class SimplexDecisionRule:
    @classmethod
    def entering(cls, simplex: StandardSimplex):
        raise NotImplementedError(
            "SimplexDecision entering needs to be "
            "implemented in a subclass."
        )

    @classmethod
    def leaving_row(cls, entering, simplex: StandardSimplex):
        raise NotImplementedError(
            "SimplexDecision leaving_row needs to be "
            "implemented in a subclass."
        )


def dantzig(simplex: StandardSimplex):
    nonbasic = np.where(~simplex.base)[0]
    positive = np.where(simplex.objective_fun[nonbasic] > 0)[0]

    if positive.size == 0:
        return None

    entering = nonbasic[
        positive[simplex.objective_fun[nonbasic][positive].argmax()]
    ]

    return entering


class Dantzig(SimplexDecisionRule):
    @classmethod
    def entering(cls, simplex):
        return dantzig(simplex)

    @classmethod
    def leaving_row(cls, entering, simplex):
        positive = np.where(simplex.table[..., entering] > 0)[0]

        bounds = (
            simplex.table[..., -1][positive]
            / simplex.table[..., entering][positive]
        )
        valid = np.where(bounds >= 0)[0]

        return positive[valid[bounds[valid].argmin()]]


class DantzigAux(SimplexDecisionRule):
    @classmethod
    def entering(cls, simplex):
        return dantzig(simplex)

    @classmethod
    def leaving_row(cls, entering, simplex: StandardSimplex):
        positive = np.where(simplex.table[..., entering] > 0)[0]

        bounds = (
            simplex.table[..., -1][positive]
            / simplex.table[..., entering][positive]
        )
        valid = np.where(bounds >= 0)[0]
        arg_min_bound = valid[bounds[valid].argmin()]
        leaving_row = positive[arg_min_bound]

        # Choose the auxiliary variable with higher priority
        if simplex.base[0]:
            aux_variable_row = simplex._var_index_to_row[0]
            if aux_variable_row in positive:
                candidates = np.where(positive == aux_variable_row)[0]
                if candidates.size > 0:
                    positive_aux_index = candidates[0]
                    if bounds[positive_aux_index] == bounds[arg_min_bound]:
                        leaving_row = simplex._var_index_to_row[0]

        return leaving_row


class Bland(SimplexDecisionRule):
    @classmethod
    def entering(cls, simplex):
        nonbasic = np.where(~simplex.base)[0]
        positive = np.where(simplex.objective_fun[nonbasic] > 0)[0]

        if positive.size == 0:
            return None

        return nonbasic[positive[0]]

    @classmethod
    def leaving_row(cls, entering, simplex):
        positive = np.where(simplex.table[..., entering] > 0)[0]

        bounds = (
            simplex.table[..., -1][positive]
            / simplex.table[..., entering][positive]
        )
        valid = np.where(bounds >= 0)[0]

        return positive[valid[bounds[valid].argmin()]]


class Lexicographic(SimplexDecisionRule):
    @classmethod
    def entering(cls, simplex):
        return dantzig(simplex)

    @classmethod
    def leaving_row(cls, entering, simplex):
        positive = np.where(simplex.table[..., entering] > 0)[0]

        bounds = (
            simplex.table[..., -1][positive]
            / simplex.table[..., entering][positive]
        )
        valid = np.where(bounds >= 0)[0]
        min_val = bounds[valid].min()
        candidates = positive[np.where(bounds == min_val)[0]]

        nonbasic = ~simplex.base
        leaving_row = None
        best_candidate = None
        for candidate_row in candidates:
            divide_coefficient = simplex.table[candidate_row, entering]
            if divide_coefficient == 0:
                continue

            next_candidate = (
                simplex.table[candidate_row, :-1][nonbasic]
                / divide_coefficient
            )
            if leaving_row is None or _is_lexicographically_greater(
                next_candidate, best_candidate
            ):
                leaving_row = candidate_row
                best_candidate = next_candidate

        return leaving_row


def _is_lexicographically_greater(fa, fb):
    unequal = np.where(fa != fb)[0]
    if unequal.size > 0:
        index_unequal = unequal[0]
        if fa[index_unequal] > fb[index_unequal]:
            return True

    return False
