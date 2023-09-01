import unittest
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from net_flow_opt.system import System
from net_flow_opt.scheduler import  Plan, Activity
from net_flow_opt.callbacks import TrackPerformance
from net_flow_opt.continuous_model import ContinuousModel
from net_flow_opt.utils import components, structure, activities_duration


class TestContinuousAlgorithm(unittest.TestCase):

    def test_continuous_algorithm(self):
        pop_size = 150
        termination = ('n_gen', 20)
        seed = 1124
        resources = 3
        eps = 1e-3

        system = System(structure, resources, components)

        dates = np.array([c.x_star for c in components])

        original_activities = [
            Activity(component, date, duration)
            for component, date, duration in zip(system.components, dates, activities_duration)
        ]

        original_plan = Plan(
            system=system,
            activities=original_activities
        )

        problem = ContinuousModel(
            system=system,
            original_plan=original_plan,
            resources=resources,
        )

        T = np.max(dates)

        d_t = np.vstack([- dates, T - dates])

        i = np.argmax(np.abs(d_t), axis=0)

        d_t = d_t[i, np.arange(d_t.shape[1])]

        ref_point = np.array([
            system.regular_flow * T,
            np.sum([a.h(d_t[i] + eps) for i, a in enumerate(original_activities)])
        ])

        algorithm = NSGA2(
            pop_size=pop_size,
            eliminate_duplicates=True,
            callback=TrackPerformance(ref_point=ref_point)
        )

        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            seed=seed,
            verbose=True
        )


if __name__ == '__main__':
    unittest.main()
