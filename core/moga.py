import numpy as np
from tqdm import tqdm


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
        self.score = None
        self.crowding_distance = 0

    def __str__(self):
        if self.score is not None:
            return "Individual, score: LF = {:.3f}, IC = {:.3f}, rank: {}, cd:\
                {}.".format(self.score[0], self.score[1], self.rank,
                            self.crowding_distance)
        else:
            return "Individual, score: {}, rank: {}, cd: {}."\
                .format(self.score, self.rank, self.crowding_distance)


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

    def run(self):
        """
        The method runs the algorithm until the ``stopping_criteria`` is not
        met and it returns the last generation of individuals.
        """
        self.n_iterations = n_iterations
        # Generate the initial population
        P = self._initial_population()
        # Save the actual population for statistics
        self.population_history.append(P)
        for i in tqdm(range(self.n_generations),
                      desc="MOGA execution", ncols=100):
            # Generate the offspring population Q by mutation
            Q = self._mutation(P)
            population = self._score(population=P+Q)
            # Perform fast non-dominated sort
            fronts = self._fast_non_dominated_sort(population)
            # Create the new generation
            P = []
            for f in fronts:
                f = self._crowding_distance(f)
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

            |G| \le R \qquad \\forall G \in grouping\_structure

            \min_{c \in G} \{t_c + d_c\} > \max_{c \in G} \{t_c\} \qquad
            \\forall G \in grouping\_structure

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
                S1[i,Q[j]] = 1
                # Check if the date ranges of components in the j-th group are
                # compatible
                components = [c for k, c in enumerate(self.plan.activities) \
                    if S1[k,Q[j]] == 1]
                max_end_date = np.max(np.array([c.t + c.d for c in
                                                components]))
                min_begin_date = np.min(np.array([c.t for c in components]))
                # If the intersection of the date ranges is not an empty set...
                if max_end_date >= min_begin_date:
                    # the group is feasible.
                    # Check if the number of components in the group is lower
                    # than the number of available resources.
                    if np.sum(S1[:,Q[j]]) >= self.plan.system.resources:
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