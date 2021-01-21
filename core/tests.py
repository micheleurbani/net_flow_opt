import copy
import random
import plotly
import unittest
import numpy as np
import networkx as nx


from system import Component, System
from scheduler import Activity, Group, Plan
from utils import components, structure


random.seed(123)


class ComponentTestCase(unittest.TestCase):

    def test_components_declaration(self):
        for c in components:
            self.assertIsInstance(c, Component)


class SystemTestCase(unittest.TestCase):

    def setUp(self):
        self.system = System(
            structure=structure,
            resources=3,
            components=components,
        )

    def test_system_declaration(self):
        self.assertIsInstance(self.system, System)

    def test_pointing_to_component(self):
        cid = random.randint(0, self.system.N - 1)
        c = self.system.components[cid]
        new_date = 14.0
        c.t = new_date
        self.assertEqual(c.t, self.system.components[cid].t)

    def test_edged_graph(self):
        e_capacited_graph = self.system.from_node_to_edge_capacity(
            structure=self.system.structure
        )
        self.assertIsInstance(e_capacited_graph, nx.DiGraph)


class ActivityTestCase(unittest.TestCase):

    def setUp(self):
        self.cid = random.randint(0, len(components) - 1)
        self.activity = Activity(
            component=components[self.cid],
            date=14.0,
            duration=3.0,
        )

    def test_pointing_to_component(self):
        self.activity.component.x_star = 19.0
        self.assertEqual(
            components[self.cid].x_star,
            self.activity.component.x_star
        )


class GroupTestCase(unittest.TestCase):

    def setUp(self):
        # Create one maintenance activity per component
        self.activities = [
            Activity(
                component=c,
                date=c.x_star,
                duration=random.random() * 3 + 1
            ) for c in components
        ]
        self.selected_activities = random.sample(self.activities, 3)
        self.group = Group(
            activities=self.selected_activities,
        )

    def test_group_feasibility(self):
        # Test an infeasible group
        # TODO: find an infeasible group and use it here
        group = Group(
            activities=[
                self.activities[1],
                self.activities[2],
                self.activities[3],
            ]
        )
        self.assertTrue(group.is_feasible())

    def test_find_execution_date(self):
        optimized_group = copy.deepcopy(self.group)
        optimized_group.minimize()
        for i in range(self.group.size):
            # print(
            #     "t_opt: {:.3f}".format(optimized_group.activities[i].t),
            #     "\tt: {:.3f}".format(self.group.activities[i].t)
            # )
            self.assertNotEqual(
                self.group.activities[i].t,
                optimized_group.activities[i].t
            )


class PlanTestCase(unittest.TestCase):

    def setUp(self):
        # Create one maintenance activity per component
        self.activities = [
            Activity(
                component=c,
                date=c.x_star,
                duration=random.random() * 3 + 1
            ) for c in components
        ]
        self.system = System(
            structure=structure,
            resources=3,
            components=components
        )
        self.plan = Plan(
            activities=self.activities,
            system=self.system,
        )

    def test_gantt_chart(self):
        self.assertIsInstance(
            self.plan.gantt_chart(),
            plotly.graph_objs._figure.Figure
        )

    def test_group_assignment(self):
        # Encode a random grouping structure in a numpy array
        sgm = []
        for i in range(self.system.N):
            x = np.zeros(self.system.N)
            x[np.random.randint(self.system.N)] = 1
            sgm.append(x)
        sgm = np.stack(sgm)
        # print(sgm)
        # Create a copy of the plan to be modified according to the sgm
        plan_opt = Plan(
            system=copy.deepcopy(self.system),
            activities=copy.deepcopy(self.activities),
            grouping_structure=sgm,
        )
        # print(self.plan, "\n")
        # print(plan_opt)
        # Compare the original plan with the new plan
        for j in range(sgm.shape[1]):
            # Check each column of the sgm: if there are more than one
            # activity, compare the new dates with the old ones; these should
            # differ.
            if np.sum(sgm[:, j]) > 1:
                for i in range(len(sgm[:, j])):
                    if sgm[i, j] == 1:
                        self.assertNotEqual(
                            self.plan.activities[i].t,
                            plan_opt.activities[i].t,
                        )

    def test_structure_history_generation(self):
        history = self.plan.generate_structure_history()
        # Verify that the number of nodes changes before and after each date
        for i in range(1, len(history)):
            self.assertNotEqual(
                len(history[i]["structure"].nodes),
                len(history[i + 1]["structure"].nodes),
            )
            self.assertIsInstance(history[i], dict)
            self.assertIsInstance(
                history[i]["structure"],
                nx.DiGraph
            )


if __name__ == "__main__":
    unittest.main()
