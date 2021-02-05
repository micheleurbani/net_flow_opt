import copy
import random
import plotly
import unittest
import numpy as np
import networkx as nx


from .system import Component, System
from .scheduler import Activity, Group, Plan
from .moga import Individual, MOGA, MOGAResults
from .utils import components, structure


# Set seed values
random.seed(123)
np.random.seed(12)


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
        self.assertIsNotNone(optimized_group.IC)
        if len(self.activities) > 1:
            self.assertGreater(optimized_group.IC, 0.0)


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

    def add_random_grouping(self):
        # Encode a random grouping structure in a numpy array
        sgm = self.plan.generate_random_assignment_matrix()
        self.plan = Plan(
            activities=self.activities,
            system=self.system,
            grouping_structure=sgm,
            original_plan=self.plan,
        )

    def test_separate_activities(self):
        sgm = np.zeros((self.plan.N, self.plan.N), dtype=int)
        sgm = np.fill_diagonal(sgm, 1)
        plan = Plan(
            activities=self.activities,
            system=self.system,
            grouping_structure=sgm,
            original_plan=self.plan,
        )

    def test_gantt_chart(self):
        self.assertIsInstance(
            self.plan.plot_gantt_chart(),
            plotly.graph_objs._figure.Figure
        )

    def test_set_dates(self):
        # Encode a random grouping structure in a numpy array
        sgm = self.plan.generate_random_assignment_matrix()
        # Create a copy of the plan to be modified according to the sgm
        plan_opt = Plan(
            system=copy.deepcopy(self.system),
            activities=copy.deepcopy(self.activities),
            grouping_structure=sgm,
            original_plan=self.plan,
        )
        # Initialize varible to check existence of at leat one group
        at_least_one_group = False
        # Compare the original plan with the new plan
        for j in range(sgm.shape[1]):
            # Check each column of the sgm: if there are more than one
            # activity, compare the new dates with the old ones; these should
            # differ.
            if np.sum(sgm[:, j]) > 1:
                # The is at least one group of size > 1
                at_least_one_group = True
                for i in range(len(sgm[:, j])):
                    if sgm[i, j] == 1:
                        self.assertNotEqual(
                            self.plan.activities[i].t,
                            plan_opt.activities[i].t,
                        )
        # If there is at least one group of size > 1, the IC must be > 0
        if at_least_one_group:
            self.assertGreater(plan_opt.IC, 0.0)

    def test_generate_structure_history(self):
        # Change the plan by adding a random grouping structure
        self.add_random_grouping()
        history = self.plan.generate_structure_history()
        # Verify that the number of nodes changes before and after each date
        for i in range(1, len(history)):
            self.assertIsInstance(history[i], dict)
            self.assertIsInstance(
                history[i]["structure"],
                nx.DiGraph
            )

    def test_generate_flow_history(self):
        # Change the plan by adding a random grouping structure
        self.add_random_grouping()
        history = self.plan.generate_flow_history()
        for config in history:
            self.assertGreaterEqual(config["flow"], 0)

    def test_evaluate_flow_reduction(self):
        # Change the plan by adding a random grouping structure
        self.add_random_grouping()
        lf = self.plan.evaluate_flow_reduction()
        self.assertGreater(lf, 0)


class IndividualTestCase(unittest.TestCase):

    def setUp(self):
        pass


class MOGATestCase(unittest.TestCase):

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
        self.moga = MOGA(
            init_pop_size=5,
            p_mutation=0.1,
            n_generations=5,
            maintenance_plan=self.plan,
            parallel=False,
        )

    def test_generate_individual(self):
        individual = self.moga.generate_individual()
        for i in range(individual.shape[0]):
            self.assertEqual(sum(individual[i, :]), 1)
        for j in range(individual.shape[1]):
            self.assertGreaterEqual(
                self.system.resources,
                sum(individual[:, j])
            )
        # TODO: remember to test also with parallel=True

    def test_generate_initial_population(self):
        population = self.moga.generate_initial_population()
        self.assertIsNotNone(population)
        self.assertIsInstance(population, list)
        for i in population:
            self.assertIsInstance(i, Individual)

    def test_adapt_initial_population(self):
        self.moga.run()
        population = self.moga.population_history[-1]
        resources_old = self.moga.plan.system.resources
        self.moga.plan.system.resources += 1
        self.moga.run(initial_population=population)
        for ind in self.moga.population_history[-1]:
            self.assertIsInstance(ind, Individual)
            self.assertGreater(ind.plan.system.resources, resources_old)

    def test_selection(self):
        pop = self.moga.generate_initial_population()
        sel = self.moga.selection(pop)
        self.assertEqual(len(pop), len(sel))

    def test_mutation(self):
        population = []
        for i in range(20):
            sgm = self.moga.generate_individual()
            individual = Individual(
                plan=Plan(
                    activities=copy.deepcopy(self.activities),
                    system=self.system,
                    grouping_structure=sgm,
                    original_plan=self.plan,
                )
            )
            population.append(individual)
        # Change p_mutation to be sure the mutation to occure
        self.moga.p_mutation = 1
        mutated = self.moga.mutation(population)
        for i in mutated:
            self.assertFalse(np.all(i == i.plan.grouping_structure))
        self.moga.p_mutation = 0.2

    def test_fast_non_dominated_sort(self):
        population = self.moga.generate_initial_population()
        fronts = self.moga.fast_non_dominated_sort(population)
        # for i, f in enumerate(fronts):
        #     print('Front {}'.format(i))
        #     for i in f:
        #         print(i)
        self.assertGreaterEqual(len(fronts[0]), 1)

    def test_crowding_distance(self):
        population = self.moga.generate_initial_population()
        fronts = self.moga.fast_non_dominated_sort(population)
        for front in fronts:
            self.moga.crowding_distance(front)
            self.assertEqual(
                front[0].score[0],
                max((i.score[0] for i in front))
            )
            self.assertEqual(
                front[-1].score[1],
                max((i.score[1] for i in front))
            )
            for individual in front:
                self.assertIsNotNone(individual.crowding_distance)

    def test_run(self):
        self.moga.run()
        # Verify the number of generations
        self.assertEqual(
            len(self.moga.population_history),
            self.moga.n_generations + 1,
        )
        # Check the number of individuals in each generation
        for gen in self.moga.population_history:
            self.assertEqual(len(gen), self.moga.init_pop_size)


class MOGAResultsTestCase(unittest.TestCase):

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
        self.moga = MOGA(
            init_pop_size=5,
            p_mutation=0.1,
            n_generations=5,
            maintenance_plan=self.plan,
            parallel=False,
        )
        # Run the algorithm
        self.moga.run()

    def test_hypervolume_indicator(self):
        results = MOGAResults(self.moga)
        # Retrieve bounds for calculation
        lf_max, lf_min, ic_max, ic_min = 0.0, 1e4, 0.0, 1e4
        for generation in results.moga.population_history:
            for ind in generation:
                if ind.plan.LF > lf_max:
                    lf_max = ind.plan.LF
                elif ind.plan.LF < lf_min:
                    lf_min = ind.plan.LF
                if ind.plan.IC > ic_max:
                    ic_max = ind.plan.IC
                elif ind.plan.IC < ic_min:
                    ic_min = ind.plan.IC
        hv_values = results.hypervolume_indicator(1000, lf_max, lf_min,
                                                  ic_max, ic_min)


if __name__ == "__main__":
    unittest.main()
