import unittest
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from net_flow_opt.utils import components, structure, activities_duration
from net_flow_opt.system import System
from net_flow_opt.scheduler import  Plan, Activity
from net_flow_opt.discrete_model import DiscreteModel
from net_flow_opt.sampling import GroupingStructureSampling


class TestDiscreteAlgorithm(unittest.TestCase):

    def test_discrete_algorithm(self):
        pop_size = 150
        termination = ('n_gen', 5)
        seed = 1124
        resources = 3

        system = System(structure, resources, components)

        dates = [c.x_star for c in components]

        original_activities = [
            Activity(component, date, duration)
            for component, date, duration in zip(system.components, dates, activities_duration)
        ]

        original_plan = Plan(
            system=system,
            activities=original_activities
        )

        problem = DiscreteModel(
            system=system,
            original_plan=original_plan,
            resources=resources,
        )

        algorithm = NSGA2(
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            pop_size=pop_size,
            eliminate_duplicates=True,
        )

        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            seed=seed,
            verbose=True
        )

    def test_discrete_sampling(self):
        s = GroupingStructureSampling()

        resources = 3

        system = System(structure, resources, components)

        dates = [c.x_star for c in components]

        original_activities = [
            Activity(component, date, duration)
            for component, date, duration in zip(system.components, dates, activities_duration)
        ]

        original_plan = Plan(
            system=system,
            activities=original_activities
        )

        problem = DiscreteModel(
            system=system,
            original_plan=original_plan,
            resources=resources,
        )

        X = s.do(problem, 3)


if __name__ == '__main__':
    unittest.main()
