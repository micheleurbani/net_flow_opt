import unittest
import numpy as np
from net_flow_opt.utils import structure, components
from net_flow_opt.network import Component, System, Activity, Plan, Group
from net_flow_opt.model import ContinuousModel, DiscreteModel


class TestComponents(unittest.TestCase):

    def setUp(self) -> None:
        self.system = System(structure, components)

    def test_component(self):
        c = Component(40, 15, 35, 2.1, 20, 2.5)

    def test_system(self):
        s = System(structure, components)

    def test_activity(self):
        i = 0

        a = Activity(self.system.components[i])

        with self.assertRaises(AssertionError):
            a.expectedCost(- a.component.x_star - 1)

    def test_group(self):
        activities = [Activity(c) for c in components]
        g = Group([activities[0], activities[3], activities[7]])
        tG = g.minimize()
        for a in g.activities:
            self.assertGreaterEqual(tG, - a.component.x_star)

    def test_plan(self):
        activities = [Activity(c) for c in components]
        plan = Plan(activities, self.system)
        # sample a random grouping structure
        rnd = np.random.randint(1, len(activities), len(activities))
        # rnd = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 3])
        gStr = np.zeros((len(activities), len(activities)))
        gStr[np.arange(len(activities), dtype=int), rnd] = 1
        # calculate dates
        t = plan.compute_dates(gStr)
        for i, a in enumerate(plan.activities):
            self.assertGreaterEqual(t[i], - a.component.x_star)
        # test calculation of IC

        # test calculation of LF
        history = plan.generate_structure_history(t)
        history = plan.generate_flow_history(t)
        LF = plan.evaluate_LF(t)
        self.assertGreaterEqual(LF, 0)
        IC = plan.evaluate_IC(t)
        self.assertGreaterEqual(IC, 0)


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        self.system = System(structure, components)

    def test_continuous(self):
        model = ContinuousModel(system=self.system, resources=4)
        # sample random dates
        t = np.zeros(len(model.plan.activities))
        for i in range(len(model.plan.activities)):
            a = model.plan.activities[i]
            t[i] = np.random.rand() * (a.component.x_star * 2)
        out = {}
        model._evaluate(t, out)

        model = DiscreteModel(system=self.system, resources=4)
        # sample a random grouping structure
        N = self.system.N
        rnd = np.random.randint(1, N, N)
        # rnd = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 3])
        gStr = np.zeros((N, N))
        gStr[np.arange(N, dtype=int), rnd] = 1
        # calculate dates
        out = {}
        model._evaluate(gStr, out)
        print(out)
