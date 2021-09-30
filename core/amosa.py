
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

    def energy(self, temperature):
        raise NotImplementedError


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
