import numpy as np
from tqdm import tqdm
from pickle import dump
from copy import deepcopy
from pandas import DataFrame
from itertools import repeat
from os import listdir, mkdir
from plotly.express import scatter
from multiprocessing import Pool, cpu_count


from .scheduler import Group, Plan


class Individual(object):
    """
    The object contains useful information for the MOGA about the individual.

    :param network_system: a :class:`core.system.System` object containing
    system information

    """

    def __init__(self, plan):
        self.plan = plan
        self.dominated_solutions = []
        self.dominator_counter = 0
        self.rank = 0
        self.score = np.array([self.plan.LF, self.plan.IC])
        self.crowding_distance = 0

    def __str__(self):
        if self.score is not None:
            return "Individual, score: LF = {:.3f}, IC = {:.3f}, rank: {}, cd:\
                {}.".format(self.score[0], self.score[1], self.rank,
                            self.crowding_distance)
        else:
            return "Individual, score: {}, rank: {}, cd: {}."\
                .format(self.score, self.rank, self.crowding_distance)

    @staticmethod
    def mutate(individual, p_mutation, original_plan):
        """
        The method is used to parallelize the mutation process of a population.
        It returns a newly generated individual.
        """
        x = deepcopy(individual.plan.grouping_structure)
        for i in range(x.shape[0]):               # Loop among components
            if np.random.rand() < p_mutation:     # Sample mutation probability
                g = []                            # List of candidate groups
                g_id = np.flatnonzero(np.sum(x[i, :, :], axis=0))[0]
                for j in range(x.shape[1]):
                    # Avoid to check the component against itself
                    # and screen the possible groups using the number of
                    # resources
                    if j != g_id and np.sum(x[:, j, :]) < \
                            individual.plan.system.resources:
                        g.append(j)
                # Further screen the groups on feasibility
                f = []
                for j in g:
                    activities = [
                            individual.plan.activities[c] for c in
                            np.flatnonzero(np.sum(x[:, j, :], axis=1))
                        ] + [individual.plan.activities[i]]
                    group = Group(activities=activities)
                    if group.is_feasible():
                        # Group is feasible update f
                        f.append(j)
                allele = np.zeros((individual.plan.N,
                                  individual.plan.system.resources), dtype=int)
                allele[np.random.choice(f),
                       np.random.randint(individual.plan.system.resources)] = 1
                x[i, :, :] = allele
        individual = Individual(
            plan=Plan(
                activities=deepcopy(individual.plan.activities),
                system=individual.plan.system,
                grouping_structure=x,
                original_plan=original_plan,
            )
        )
        return individual


class MOGA(object):
    """
    Implementation of the NSGA-II algorithm by Deb.

    :param int init_pop_size: the number of individuals in the initial
    population.
    :param float p_mutation: the probability of a mutation to occur.
    :param int n_generations: the number of generations after which to stop.
    :param object maintenance_plan: a maintenance plan object
    :class:`core.scheduler.Plan`.
    :param bool parallel: whether to run the parallelized version of the
    algorithm or not.
    """

    def __init__(
        self,
        init_pop_size,
        p_mutation,
        n_generations,
        maintenance_plan,
        parallel=False
    ):
        self.init_pop_size = init_pop_size
        self.parallel = parallel
        self.p_mutation = p_mutation
        self.n_generations = n_generations
        self.plan = maintenance_plan
        self.population_history = []

    def run(self, initial_population=None):
        """
        The method runs the algorithm until the ``stopping_criteria`` is not
        met and it returns the last generation of individuals.
        """
        # Generate the initial population
        if initial_population is None:
            P = self.generate_initial_population()
        else:
            P = self.adapt_initial_population(initial_population)
        # Save the actual population for statistics
        self.population_history.append(P)
        for i in tqdm(range(self.n_generations),
                      desc="MOGA execution", ncols=100):
            # Generate the offspring population Q by mutation
            Q = self.mutation(P)
            # Perform fast non-dominated sort
            fronts = self.fast_non_dominated_sort(P+Q)
            # Create the new generation
            P = []
            for f in fronts:
                f = self.crowding_distance(f)
                if len(P) + len(f) < self.init_pop_size:
                    P += f
                else:
                    P += f[:self.init_pop_size-len(P)]
                    break
            assert len(P) == self.init_pop_size
            # Save the actual populaiton for statistics
            self.population_history.append(P)

    def generate_individual(self, i=None):
        """
        The method generates a feasible grouping structure. The following two
        constraints are satisfied:

        .. math::

            |G| \\le R \\qquad \\forall G \\in grouping\\_structure

            \\min_{c \\in G} \\{t_c + d_c\\} > \\max_{c \\in G} \\{t_c\\}
            \\qquad \\forall G \\in grouping\\_structure

        where :math:`G` identifies a set of components also called group, which
        ensure that the optimality of nestde intervals holds.

        :return: a numpy array encoding the grouping structure.
        :rtype: numpy array

        """
        # List of components to be assigned to a group
        R = np.arange(self.plan.N)
        # Empty array to store the grouping structure
        S = np.zeros(shape=(self.plan.N, self.plan.N), dtype=int)
        for i in range(self.plan.N):
            # Create a local copy of R, which will be used for assignment of
            # the component to one of the remaining groups
            Q = np.copy(R)
            while True:
                # Create a local copy of S
                S1 = np.copy(S)
                # Sample a group from Q
                j = np.random.randint(len(Q))
                # Assign the component to the group
                S1[i, Q[j]] = 1
                # Check if the date ranges of components in the j-th group are
                # compatible
                components = [c for k, c in enumerate(self.plan.activities)
                              if S1[k, Q[j]] == 1]
                max_end_date = np.max(np.array([c.t + c.d for c in
                                                components]))
                min_begin_date = np.min(np.array([c.t for c in components]))
                # If the intersection of the date ranges is not an empty set...
                if max_end_date >= min_begin_date:
                    # the group is feasible.
                    # Check if the number of components in the group is lower
                    # than the number of available resources.
                    if np.sum(S1[:, Q[j]]) >= self.plan.system.resources:
                        # The group is full, remove it from R.
                        for k, c in enumerate(R):
                            if c == Q[j]:
                                R = np.delete(R, k)
                    # Update S and
                    S = np.copy(S1)
                    # step to the next component.
                    break
                else:
                    # clear group from the local list of candidate groups
                    Q = np.delete(Q, j)
        return S

    def generate_individual_with_resources(self, i=None):
        S = self.generate_individual()
        assert type(S) is np.ndarray
        x = []
        for i in range(S.shape[0]):
            c = np.zeros((self.plan.N, self.plan.system.resources))
            c[:, np.random.randint(self.plan.system.resources)] = S[i, :]
            x.append(c)
        S = np.stack(x)
        return S

    def generate_initial_population(self):
        """
        Generate an initial population of :class:`core.moga.Individual`
        objects.

        :return: a list of :class:`core.moga.Individual` objects.

        """
        if self.parallel:
            with Pool(processes=cpu_count()) as pool:
                population = pool.map(
                    self.generate_individual_with_resources,
                    range(self.init_pop_size - 1)
                )
        else:
            population = [self.generate_individual_with_resources() for _ in
                          range(self.init_pop_size - 1)]
        population = [
            Individual(
                plan=Plan(
                    system=self.plan.system,
                    activities=deepcopy(self.plan.activities),
                    grouping_structure=s,
                    original_plan=self.plan,
                )
            ) for s in population
        ]
        # Remember to inject artificially the solution with all the activities
        # executed separately.
        S = np.zeros(shape=(self.plan.N, self.plan.N,
                            self.plan.system.resources), dtype=int)
        for i in range(self.plan.N):
            S[i, i, np.random.randint(self.plan.system.resources)] = 1
        population.append(
            Individual(
                plan=Plan(
                    system=self.plan.system,
                    activities=deepcopy(self.plan.activities),
                    grouping_structure=S,
                    original_plan=self.plan,
                )
            )
        )
        return population

    def adapt_initial_population(self, initial_population):
        """
        The method (un-)squeeze the assignment matrix along the dimension of
        the resources.

        :param list initial_population: a list of :class:`core.moga.Individual`
        objects from which the assignment matrices are extracted.

        :return: a list of :class:`core.moga.Individual` objects with the new
        assignment matrices.
        """
        population = [ind.plan.grouping_structure for ind
                      in initial_population]
        assert population[0].shape[2] <= self.plan.system.resources
        dim_to_add = self.plan.system.resources - population[0].shape[2]
        if dim_to_add > 0:
            population = [
                np.dstack(
                    (ind, np.zeros((self.plan.N, self.plan.N, dim_to_add)))
                ) for ind in population
            ]
            population = [
                Individual(
                    plan=Plan(
                        activities=deepcopy(self.plan.activities),
                        system=self.plan.system,
                        grouping_structure=sgm,
                        original_plan=self.plan,
                    )
                ) for sgm in population
            ]
            return population
        elif dim_to_add == 0:
            return initial_population
        else:
            raise ValueError("The number of resources assigned to the system\
                is lower than the number of resources in the initial_\
                population")

    def mutation(self, parents):
        """
        The method returns a population of mutated individuals.
        The procedure parallel processes the individuals in parents.

        :param list parents: a list of individuals that are used to generate
        the off spring.
        :return: a list of mutated individuals.

        """
        if self.parallel:
            with Pool(processes=cpu_count()) as pool:
                offspring = pool.starmap(
                    Individual.mutate,
                    zip(parents, repeat(self.p_mutation, len(parents)),
                        repeat(self.plan, len(parents)))
                )
        else:
            offspring = [Individual.mutate(individual, self.p_mutation,
                         self.plan) for individual in parents]
        return offspring

    def fast_non_dominated_sort(self, population):
        """
        Apply the fast non-dominated sort algorithm as in Deb et al. [1]_.

        :param list population: a list of :class:`core.moga.Individual`
        objects.
        :return: a list of lists; each list represents a front, and fronts are
        returned in ascending order (from the best to the worst).

        """

        # Initialize the list of individuals
        frontiers = [[]]
        # Clean the population
        for i in population:
            i.rank = 0
            i.dominatorCounter = 0
            i.dominatedSolutions = list()
        # Start the algorithm
        for p in population:
            for q in population:
                if p.score[0] < q.score[0] and p.score[1] < q.score[1]:
                    # p dominates q, add q to the set of solutions dominated
                    # by p
                    p.dominatedSolutions.append(q)
                elif p.score[0] > q.score[0] and p.score[1] > q.score[1]:
                    # p is dominated by q, increment the dominator counter of p
                    p.dominatorCounter += 1
            if p.dominatorCounter == 0:
                p.rank = 1
                frontiers[0].append(p)
        i = 0
        while frontiers[i]:
            Q = list()
            for p in frontiers[i]:
                for q in p.dominatedSolutions:
                    q.dominatorCounter -= 1
                    if q.dominatorCounter == 0:
                        q.rank = i + 2
                        Q.append(q)
            i += 1
            frontiers.append(Q)
        if not frontiers[-1]:
            del frontiers[-1]
        return frontiers

    def crowding_distance(self, front):
        """
        Given a front - i.e., a set of individuals with the same rank - for
        each individual, the method calculates the crowding distance, which is
        appended to the individual's attribute.

        :param list population: a list of individuals.
        :return: the list of individuals in ascending order of crowding
        distance indicator.
        :rtype: list

        """
        # Initialize distance
        for ind in front:
            ind.crowding_distance = 0

        for m in range(2):
            # Sort using each objective value
            front.sort(key=lambda i: i.score[m])
            # Assign an infinite distance value to the extreme points
            front[0].crowding_distance = np.inf
            front[-1].crowding_distance = np.inf
            # Find the minimum and maximum value of the m-th objective
            m_values = [ind.score[m] for ind in front]
            m_min, m_max = min(m_values), max(m_values)
            # Crowding distance update
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].score[m] -
                                               front[i-1].score[m]) / \
                                                   (m_max - m_min)

        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)

    def save(self, fname):
        """
        Pickle the object and save it in the folder `results` with name
        `fname`.
        """
        if "results" not in listdir():
            mkdir("results")
        # Check if the name ends with .pkl extension, else add it.
        x = fname.find(".")
        if x == -1:
            fname += ".pkl"
        elif fname[x:] != ".pkl":
            raise ValueError("The name of the file is incorrect, please \
                                check its extension.")
        # Save object
        with open("results/{}".format(fname), "wb") as f:
            dump(self, f)


class MOGAResults(object):

    def __init__(self, moga):
        self.moga = moga

    def __str__(self):
        return "MOGA: {} generations with {} individuals and {} resources"\
            .format(
                len(self.moga.population_history),
                self.moga.init_pop_size,
                self.moga.plan.system.resources
            )

    def to_dataframe(self):
        df = []
        for generation, population in enumerate(self.moga.population_history):
            for idx, i in enumerate(population):
                df.append(
                    {
                        "LF": i.plan.LF,
                        "IC": i.plan.IC,
                        "rank": i.rank,
                        "generation": generation,
                        "ID": idx,
                    }
                )
        df = DataFrame(df)
        return df

    def pareto_front(self):
        """Returns a scatter plot representig the Pareto front of the last
        generation."""
        df = self.to_dataframe()
        df = df[df.generation == df.generation.max()]
        fig = scatter(
            data_frame=df,
            x="LF",
            y="IC",
            hover_name="ID",
        )
        fig.update_layout(
            title="Pareto front after {} ".format(self.moga.n_generations) +
                  "generations with {} ".format(self.moga.init_pop_size) +
                  "individuals, {} ".format(self.moga.plan.system.resources) +
                  "resources, and p_mutation={}.".format(self.moga.p_mutation)
        )
        return fig

    def pareto_evolution(self):
        """Returns an animation showing the evolution of the population."""
        df = self.to_dataframe()
        df["is_pareto"] = df["rank"] == 1
        fig = scatter(
            data_frame=df,
            x="LF",
            y="IC",
            color="is_pareto",
            animation_frame="generation",
            range_x=(df.LF.min() * 0.95, df.LF.max() * 1.05),
            range_y=(df.IC.min() - 1, df.IC.max() + 2),
        )
        return fig

    def pareto_to_csv(self):
        """Export the coordinates of the points on the Pareto front in a CSV
        file, and saves it in the folder `static`."""
        df = self.to_dataframe()
        df = df[df.generation == df.generation.max()]
        df = df.filter(["IC", "LF"])
        df = df.drop_duplicates()
        fname = "export.csv"
        df.to_csv("flask_app/static/" + fname, index=False)
        return fname
