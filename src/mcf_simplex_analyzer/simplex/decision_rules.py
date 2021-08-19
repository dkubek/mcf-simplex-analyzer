import numpy as np
from mcf_simplex_analyzer.simplex.simplex import Simplex


class SimplexDecisionRule:
    @classmethod
    def entering(cls, simplex: Simplex):
        raise NotImplementedError(
            "SimplexDecision entering needs to be "
            "implemented in a subclass."
        )

    @classmethod
    def leaving_row(cls, entering, simplex: Simplex):
        raise NotImplementedError(
            "SimplexDecision leaving_row needs to be "
            "implemented in a subclass."
        )


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
    def leaving_row(cls, entering, simplex: Simplex):
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


def dantzig(simplex: Simplex):
    nonbasic = np.where(~simplex.base)[0]
    positive = np.where(simplex.objective_fun[nonbasic] > 0)[0]

    if positive.size == 0:
        return None

    entering = nonbasic[
        positive[simplex.objective_fun[nonbasic][positive].argmax()]
    ]

    return entering


def bland(simplex: Simplex):
    pass
