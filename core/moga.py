import numpy as np
from tqdm import tqdm
from pickle import dump
from random import sample
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
        for i in range(individual.plan.N):        # Loop among components
            if np.random.rand() < p_mutation:     # Sample mutation probability
                g = []                            # List of candidate groups
                g_id = np.flatnonzero(x[i, :])[0]
                for j in range(individual.plan.N):
                    # Avoid to check the component against itself
                    # and screen the possible groups using the number of
                    # resources
                    if j != g_id and np.sum(x[:, j]) < \
                            individual.plan.system.resources:
                        g.append(j)
                # Further screen the groups on feasibility
                f, p = [], []
                for j in g:
                    activities = [
                            individual.plan.activities[c] for c in
                            np.flatnonzero(x[:, j])
                        ] + [individual.plan.activities[i]]
                    group = Group(activities=activities)
                    if group.is_feasible():
                        # Group is feasible update f
                        f.append(j)
                        p.append(max([1, group.size]))
                allele = np.zeros(individual.plan.N, dtype=int)
                assert len(f) < individual.plan.N
                p = np.array(p) / sum(p)
                allele[np.random.choice(f, p=p)] = 1
                x[i, :] = allele
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
            # Selection of individuals
            P = self.selection(P)
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

    def generate_initial_population(self):
        """
        Generate an initial population of :class:`core.moga.Individual`
        objects.

        :return: a list of :class:`core.moga.Individual` objects.

        """
        if self.parallel:
            with Pool(processes=cpu_count()) as pool:
                population = pool.map(
                    self.generate_individual,
                    range(self.init_pop_size - 1)
                )
        else:
            population = [self.generate_individual() for _ in
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
        S = np.zeros((self.plan.N, self.plan.N), dtype=int)
        for i in range(self.plan.N):
            S[i, i] = 1
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
        The method initialize a new population with the new system.

        :param list initial_population: a list of :class:`core.moga.Individual`
        objects from which the assignment matrices are extracted.

        :return: a list of :class:`core.moga.Individual` objects with the new
        assignment matrices.
        """
        population = [ind.plan.grouping_structure for ind
                      in initial_population]
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

    def selection(self, population):
        """
        The method removes the duplicates in the population.

        :return: a list of :class:`core.moga.Individual` objects.
        """
        n = len(population)
        # Remove duplicates
        selected, scores = [], []
        for ind in population:
            score = (ind.plan.IC, ind.plan.LF)
            if score not in scores:
                scores.append(score)
                selected.append(ind)
        # Enrich population by binary tournament selection
        population = []
        while len(population) < n - len(selected):
            mates = sample(selected, 2)
            if mates[0].rank == mates[1].rank:
                # Look at the rank
                if mates[0].crowding_distance >= mates[1].crowding_distance:
                    idx = 0
                else:
                    idx = 1
            elif mates[0].rank > mates[1].rank:
                idx = 1
            else:
                idx = 0
            population.append(
                Individual(
                    plan=Plan(
                        activities=deepcopy(mates[idx].plan.activities),
                        system=self.plan.system,
                        grouping_structure=mates[idx].plan.grouping_structure,
                        original_plan=self.plan,
                    )
                )
            )
        return population + selected

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
        with open("{}".format(fname), "wb") as f:
            dump(self, f)


class MOGAResults(object):

    def __init__(self, moga):
        # Sanity check
        assert type(moga) is MOGA
        # Define attributes
        self.moga = moga

    def __str__(self):
        return "MOGA: {} generations with {} individuals and {} resources"\
            .format(
                len(self.moga.population_history),
                self.moga.init_pop_size,
                self.moga.plan.system.resources
            )

    def to_dataframe(self):
        """
        Return a :class:`pandas.DataFrame` containing per individual
        information. The values include :math:`LF`, :math:`IC`, rank,
        generation, and ID of each solution.

        :return: a :class:`pandas.DataFrame` object.
        """
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
        # Filter only the last generation
        df = df[df.generation == df.generation.max()]
        # Filter individuals with rank 1
        df = df[df["rank"] == 1]
        fig = scatter(
            data_frame=df,
            x="LF",
            y="IC",
            hover_name="ID",
            range_x=(df.LF.min() * 0.95, df.LF.max() * 1.05),
            range_y=(df.IC.min() - 1, df.IC.max() + 2),
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
        fig = scatter(
            data_frame=df,
            x="LF",
            y="IC",
            animation_frame="generation",
            range_x=(df.LF.min() * 0.95, df.LF.max() * 1.05),
            range_y=(df.IC.min() - 1, df.IC.max() + 2),
        )
        return fig

    def pareto_to_csv(self):
        """Export the coordinates of the points on the Pareto front in a CSV
        file, and saves it in the folder `static`."""
        df = self.to_dataframe()
        df = df[df.generation == df.generation.max()]  # Keep only the last gen
        df = df[df["rank"] == 1]  # Keep only point on the Pareto front
        df = df.filter(["IC", "LF"])  # Keep only the score
        df = df.drop_duplicates()
        df = df.sort_values(by="IC", ascending=False)
        fname = "export.csv"
        df.to_csv("flask_app/static/" + fname, index=False)
        return fname

    @staticmethod
    def generation_HV(n_sample, list_of_scores, bounds):
        lf_max, lf_min, ic_max, ic_min = bounds
        dominated = 0
        for _ in range(n_sample):
            # Generate a random point
            x = (
                np.random.random() * (lf_max - lf_min) + lf_min,  # LF
                np.random.random() * (ic_max - ic_min) + ic_min   # IC
            )
            for score in list_of_scores:
                if score[0] < x[0] and score[1] < x[1]:
                    dominated += 1
                    break
        return dominated / n_sample

    def hypervolume_indicator(self, n_sample, lf_max, lf_min, ic_max, ic_min):
        """
        The method calculates the hyper-volume (HV) indicator of each
        generation and returns the series of points required to plot it.
        It is required to provide :math:`LF_{max}`, :math:`LF_{min}`,
        :math:`IC_{max}`, and :math:`IC_{min}` so that several HV curves can be
        compared.

        :params int n_sample: the number of sample points used to evaluate the
        HV indicator.
        :params float lf_min: the lower bound for :math:`LF`.
        :params float lf_max: the upper bound for :math:`LF`.
        :params float ic_min: the lower bound for :math:`IC`.
        :params float ic_max: the upper bound for :math:`IC`.

        :return: the HV indicator value for each generation.
        :rtype: list of float.

        """
        # Define a function for calculation of the HV of a single generation

        # Retrieve the list of scores
        scores = [
            [(ind.plan.LF, ind.plan.IC) for ind in generation if ind.rank == 1]
            for generation in self.moga.population_history
        ]
        # Wrap bounds in a tuple
        bounds = (lf_max, lf_min, ic_max, ic_min)

        # Parallelize calculation
        with Pool(processes=cpu_count()) as pool:
            hv = pool.starmap(
                MOGAResults.generation_HV,
                zip(
                    repeat(n_sample, len(self.moga.population_history)),
                    scores,
                    repeat(bounds, len(self.moga.population_history))
                )
            )
        return hv
