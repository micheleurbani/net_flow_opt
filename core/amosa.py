
import random
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count


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
        Returns true if the actual solution is in a nondomination relationship
        with the solution `a`.
        """
        if (self.score[0] > a.score[0] and self.score[1] > a.score[1]) or \
            (self.score[0] < a.score[0] and self.score[1] < a.score[1]):
            return False
        else:
            return True

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
        assert alpha < 1
        self.alpha = alpha
        self.archive = None
        self.plan = maintenance_plan
        self.parallel = parallel
        self.solution_history = []

    def range(self, i):
        """
        Since the range of an objective i is not known a priori, it is computed
        using the actual archive of solutions.
        """
        a = min([j.score[i] for j in self.archive])
        b = max([j.score[i] for j in self.archive])
        return abs(b - a)

    def domination_amount(self, a, b):
        """
        The amount of domination of a solution `a` with respect to a solution
        `b`.
        """
        return np.prod([np.abs(a.score[i] - b.score[i]) / self.range(i)
                        for i in range(len(a.score))])

    def p(self, x):
        """
        Given the dominance amount, it computes the selection probability of a
        solution.
        """
        return 1 / (1 + np.exp(x * self.T))

    def clustering(self):
        """
        The single linkage algorithm is used to perform clustering.
        """
        # filter out duplicates
        self.remove_duplicates()
        # define the distance matrix
        d = np.zeros((len(self.archive), len(self.archive)))
        for i in range(len(self.archive)):
            for j in range(i+1, len(self.archive)):
                d[i, j] = np.sqrt(
                    np.sum(
                        [(self.archive[i].score[k] +
                            self.archive[j].score[k])**2
                                for k in range(len(self.archive[i].score))]
                    )
                )
        # filter out zero from thresholds
        thresholds = [i for i in np.sort(np.unique(d)) if i != 0]
        # allocate empty matrix for conn_old
        conn_old = np.zeros_like(d)
        # create the first cluster structure
        clusters = [(i,) for i in self.archive]
        for t in thresholds:
            conn = np.zeros_like(d)
            # the connections with dissimilarity t
            conn[(d > 0) & (d <= t)] = 1
            # the new connection to add
            new_conn = conn - conn_old
            # update old_conn to the actual structure of connections
            conn_old = conn
            # new elements to be connected
            i, j = new_conn.nonzero()
            temp_clusters = []
            for node in [i, j]:
                if temp_clusters and \
                    self.archive[int(node)] in temp_clusters[0]:
                    break
                k = 0
                while not self.archive[int(node)] in clusters[k]:
                    k += 1
                temp_clusters.append(clusters.pop(k))
            if len(temp_clusters) == 1:
                clusters.append(temp_clusters[0])
            else:
                clusters.append(temp_clusters[0] + temp_clusters[1])
            # when ther's only one cluster, then stop
            if len(clusters) == 1: break
            # if there are self.HL clusters then group
            if len(clusters) == self.HL:
                for i in range(len(clusters)):
                    if len(clusters[i]) == 2:
                        del clusters[i][random.randint(0,
                            len(clusters[i]) - 1)]
                    # keep the solution with the minimum average distance
                    elif len(clusters[i]) > 2:
                        ave_dist = []
                        for j in clusters[i]:
                            ave_dist.append(
                                np.sum(
                                    [np.sqrt(
                                        np.sum(
                                            [(j.score[l] + k.score[l])**2
                                                for l in range(len(j.score))]
                                        )
                                    ) for k in clusters[i]]
                                ) / len(clusters[i])
                            )
                        clusters[i] = (clusters[i][np.argmin(ave_dist)],)
                break
        self.archive = [i[0] for i in clusters]

    def remove_duplicates(self):
        x = []
        for i in self.archive:
            add = True
            for j in x:
                if i.score[0] == j.score[0] and i.score[1] == j.score[1]:
                    add = False
                    break
            if add:
                x.append(i)
        self.archive = x

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
        print('Initial population')
        print(self.archive)
        # TODO: perform clustering
        self.clustering()
        print('Perform clustering ...')
        print(self.archive)

        # chose a random solution from the archive
        current_pt = np.random.choice(self.archive)
        j = 0
        while self.T >= self.Tmin:
            print(f'================ Iteration {j} T: {self.T} ==============')
            for i in range(self.iter):
                # generate a new solution by perturbation
                new_pt = current_pt.perturbation(
                    p_mutation=0.25,
                    original_plan=self.plan
                )
                print('new_pt: ', new_pt)
                # k_set is the set of solutions that dominate new_pt
                k_set = [i for i in self.archive if i <= new_pt]
                # d_set is the set containing the solutions dominated by new_pt
                d_set = [i for i in self.archive if new_pt <= i]
                # current_pt dominates the new_pt and k points from the archive
                # dominate new_pt
                print('iteration ', i, current_pt)
                if current_pt <= new_pt:
                    print('current_pt <= new_pt')
                    ave_dom = (sum(
                                [self.domination_amount(i, new_pt)
                                    for i in k_set]
                            ) + self.domination_amount(current_pt, new_pt)) / \
                            (len(k_set) + 1)
                    print('ave_dom:', ave_dom, 'p:', self.p(ave_dom))
                    # the new_pt is selected as the current_pr with probability
                    if np.random.rand() <= self.p(ave_dom):
                        # set new_pt as current_pt
                        current_pt = new_pt
                    print('iteration:', i, current_pt)

                # current_pt and new_pt are nondominated to each other
                elif current_pt.nondominated(new_pt):
                    print('current_pt nondominated by new_pt')
                    # check the domination status of new_pt against the points
                    # in the archive
                    # case 2a: new_pt is dominated by k points
                    if len(k_set) >= 1:
                        print(f'new_pt is dominated by {len(k_set)} points')
                        ave_dom = sum([self.domination_amount(i, new_pt)
                                       for i in self.archive]) / len(k_set)
                        print('ave_dom:', ave_dom, 'p:', self.p(ave_dom))
                        if np.random.rand() <= self.p(ave_dom):
                            # set new_pt as current_pt
                            current_pt = new_pt
                    # case 2b: new_pt is nondominating with respect to the
                    # other points in the archive
                    elif len(k_set) == 0:
                        print('new_pt is nondominating with respect to the',
                            'other points in the archive')
                        self.archive.append(deepcopy(new_pt))
                        current_pt = new_pt
                        if len(self.archive) >= self.SL:
                            self.clustering()
                    # case 2c: new_pt dominates k points in the archive
                    elif len(d_set) >= 1:
                        print(f'new_pt dominates {len(d_set)} points in the archive')
                        current_pt = new_pt
                        self.archive.append(deepcopy(new_pt))
                        for i in d_set:
                            self.archive.remove(i)
                    else:
                        raise NotImplementedError
                elif new_pt <= current_pt:
                    print('new_tp <= current_pt')
                    # new_pt dominates current_pt but k points in the archive
                    # dominate new_pt
                    if len(k_set) >= 1:
                        domination_amounts = [self.domination_amount(i, new_pt)
                                            for i in self.archive]
                        if np.random.rand() <= \
                            self.p(-min(domination_amounts)):
                            current_pt = self.archive[
                                np.argmin(domination_amounts)]
                        else:
                            current_pt = new_pt
                        self.archive.append(current_pt)
                    # new_pt is nondominating with respect to the points in the
                    # archive except current_pt if it belongs to the archive
                    elif len(k_set) == 0 and len(d_set) == 0:
                        if current_pt in self.archive:
                            self.archive.remove(current_pt)
                        current_pt = deepcopy(new_pt)
                        self.archive.append(current_pt)
                    elif len(k_set) == 0 and len(d_set) >= 1:
                        # remove solutions dominated by new_pt
                        self.archive = [i for i in self.archive
                                        if not new_pt <= i]
                        current_pt = deepcopy(new_pt)
                        self.archive.append(current_pt)
                if len(self.archive) > self.SL:
                    self.clustering()
            # Lower the temperature
            self.T = self.alpha * self.T
            print(self.archive)


if __name__ == '__main__':
    from .system import Component, System
    from .utils import components, structure
    from .scheduler import Activity, Group, Plan
    from .amosa import State, AMOSA
    random.seed(123)
    np.random.seed(123)
    # Create one maintenance activity per component
    activities = [
        Activity(
            component=c,
            date=c.x_star,
            duration=random.random() * 3 + 1
        ) for c in components
    ]
    system = System(
        structure=structure,
        resources=3,
        components=components
    )
    plan = Plan(
        activities=activities,
        system=system,
    )
    amosa = AMOSA(
        HL=3,
        SL=10,
        Tmax=1500,
        Tmin=800,
        iterations=20,
        alpha=0.95,
        maintenance_plan=plan,
    )

    amosa.run()
