
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count


from .moga import Individual
from .scheduler import Group, Plan


class State(object):
    """
    A specific state (a.k.a. solution) in the context of the AMOSA algorithm.
    """

    def __init__(self, plan):
        self.plan = plan
        self.score = np.array([self.plan.LF, self.plan.IC])

    def __repr__(self):
        return '<State LF:{LF:.2f} IC:{IC:.2f}>'.format(LF=self.score[0],
                                                        IC=self.score[1])

    def __le__(self, a):
        """
        Define the dominance relationship. Returns true if the current solution
        dominates `a`.
        """
        if self.score[0] < a.score[0] and self.score[1] < a.score[1]:
            return True
        else:
            return False

    def nondominated(self, a):
        """
        Returns true if the actual solution is a nondomination relationship
        with the solution `a`.
        """
        if self.score[0] > a.score[0] and self.score[1] > a.score[1]:
            return False
        else:
            return True

    def energy(self, temperature):
        raise NotImplementedError

    def perturbation(self, p_mutation, original_plan):
        """
        The method perturbates a solution by applying the same mutation that
        is used in `core.moga.Individual.muation`.
        """
        x = deepcopy(self.plan.grouping_structure)
        for i in range(self.plan.N):        # Loop among components
            if np.random.rand() < p_mutation:     # Sample mutation probability
                g = []                            # List of candidate groups
                g_id = np.flatnonzero(x[i, :])[0]
                for j in range(self.plan.N):
                    # Avoid to check the component against itself
                    # and screen the possible groups using the number of
                    # resources
                    if j != g_id and np.sum(x[:, j]) < \
                            self.plan.system.resources:
                        g.append(j)
                # Further screen the groups on feasibility
                f, p = [], []
                for j in g:
                    activities = [
                            self.plan.activities[c] for c in
                            np.flatnonzero(x[:, j])
                        ] + [self.plan.activities[i]]
                    group = Group(activities=activities)
                    if group.is_feasible():
                        # Group is feasible update f
                        f.append(j)
                        p.append(max([1, group.size]))
                allele = np.zeros(self.plan.N, dtype=int)
                assert len(f) < self.plan.N
                p = np.array(p) / sum(p)
                allele[np.random.choice(f, p=p)] = 1
                x[i, :] = allele
        state = State(
            plan=Plan(
                activities=deepcopy(self.plan.activities),
                system=self.plan.system,
                grouping_structure=x,
                original_plan=original_plan,
            )
        )
        return state


class AMOSA(object):
    """
    An implementation of the Archived MultiObjective Simulated Annealing by
    Bandyopadhyay et al. [_BAN2008].

    Parameters
    ----------
    HL : int
        The maximum size of the archive on termination. This is set equal to
        the maximum number of nondominated solutions required by the user.
    SL : int
        The maximum size to which the archive may be filled before clustering
        is used to reduce its size to `HL`.
    Tmax : int
        Maximum (initial) temperature.
    Tmin : int
        Minimum (initial) temperature.
    iterations : int
        Number of iterations at each temperature.
    alpha : int
        The cooling rate in SA.
    maintenance_plan : a `core.scheduler.Plan` object
        Store the information about the preventive maintenance plan to
        optimize.
    parallel : boolean
        Specify if the algorithm exploits parallel processing. Default `False`.

    Return
    ------

    """

    def __init__(self, HL, SL, Tmax, Tmin, iterations, alpha,
                 maintenance_plan, parallel=False):
        self.HL = HL
        self.SL = SL
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.T = Tmax
        self.iter = iterations
        self.alpha = alpha
        self.archive = None
        self.plan = maintenance_plan
        self.parallel = parallel

    def selection_probability(self, s, q):
        """
        The selection probability is computer according to the follwing
        equation:

            p_{s, q} = \\frac{1}{1 + \\exp \\frac{-E(q, T) - E(s, T)}{T}}

        `s` and `q` are the current and the previous state respectively.
        """
        return 1 / (1 + np.exp((-q.energy - s.energy) / self.T))

    @staticmethod
    def domination_amount(a, b):
        """
        The amount of domination of a solution `a` with respect to a solution
        `b`.
        """
        return np.prod([np.abs(a.score[i] - b.score[i]) / R_i
                        for i in range(len(a.score))])

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
        Generate an initial population of :class:`core.amosa.State` objects.

        :return: a list of :class:`core.amosa.State` objects.
        """
        if self.parallel:
            with Pool(processes=cpu_count()) as pool:
                population = pool.map(
                    self.generate_individual,
                    range(self.SL - 1)
                )
        else:
            population = [self.generate_individual() for _ in
                          range(self.SL - 1)]
        population = [
            State(
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
            State(
                plan=Plan(
                    system=self.plan.system,
                    activities=deepcopy(self.plan.activities),
                    grouping_structure=S,
                    original_plan=self.plan,
                )
            )
        )
        return population

    def run(self):
        """
        The main procedure of the algorithm.
        """
        if self.archive is None:
            self.archive = self.generate_initial_population()
        # TODO: perform clustering
        print(self.archive)

        # chose a random solution from the archive
        current_pt = np.random.choice(self.archive)
        # generate a new solution by perturbation
        new_pt = current_pt.perturbation(
            p_mutation=0.25,
            original_plan=self.plan
        )

        # k_set is the set of solutions that dominate new_pt
        ks = lambda new_pt: np.sum([i for i in self.archive if i <= new_pt])

        while self.T >= self.Tmin:
            for i in range(self.iter):
                # current_pt dominates the new_pt and k points from the archive
                # dominate new_pt
                if current_pt <= new_pt:
                    k_set = ks(new_pt=new_pt)
                    ave_dom = (
                            np.sum(
                                [self.domination_amount(i, new_pt)
                                    for i in k_set]
                            ) + self.domination_amount(current_pt, new_pt)) / \
                            (k + 1)
                    # the new_pt is selected as the current_pr with probability
                    p = 1 / (1 + np.exp(ave_dom * self.T))
                    if np.random.rand() <= p:
                        # set new_pt as current_pt
                        current_pt = deepcopy(new_pt)

                # current_pt and new_pt are nondominated to each other
                elif current_pt.nondominated(new_pt):
                    # check the domination status of new_pt and points in the
                    # archive
                    pass

                break
                # TODO: check if len(self.archive) > SL
            self.T = self.Tmin - 1


if __name__ == '__main__':
    pass
