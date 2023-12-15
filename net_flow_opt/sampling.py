import numpy as np

from pymoo.core.sampling import Sampling


class GroupingStructureSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.zeros((n_samples, problem.original_plan.activities))

        return val

    