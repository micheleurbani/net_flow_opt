import numpy as np

from pymoo.core.problem import ElementwiseProblem

from net_flow_opt.network import System, Plan, Activity


class BaseModel(ElementwiseProblem):

    def __init__(self, system: System, resources: int, xu, xl, vtype) -> None:

        self.system = system
        self.r = resources

        # define activities
        activities = [Activity(component=c) for c in self.system.components]

        # define maintenance plan
        self.plan = Plan(activities=activities, system=self.system)

        super().__init__(
            n_var=system.N,
            n_obj=2,
            n_ieq_constr=system.N,
            xl=xl,
            xu=xu,
            vtype=float,
        )

    def eval_G(self, x):
        return np.array([
            np.sum([
                t_i <= t_j <= t_i + self.plan.activities[i].d for i, t_i in enumerate(x)
            ]) - self.r for t_j in x
        ])


class ContinuousModel(BaseModel):

    def __init__(self, system: System, resources: int) -> None:

        super().__init__(
            system=system,
            resources=resources,
            xl=0,
            xu=max([c.x_star for c in system.components]) * 1.5,
            vtype=float
        )

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = np.array(self.plan.eval(x))
        out["G"] = self.eval_G(x)


class DiscreteModel(BaseModel):

    def __init__(self, system: System, resources: int) -> None:

        super().__init__(
            xl=1,
            xu=system.N,
            vtype=int,
            system=system,
            resources=resources
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # one-hot encoding of x
        grouping_structure = np.zeros((self.system.N, self.system.N))
        grouping_structure[np.arange(self.system.N, dtype=int), x.astype(int) - 1] = 1
        t = self.plan.compute_dates(grouping_structure)

        out["F"] = np.array(self.plan.eval(t))
        out["G"] = self.eval_G(t)


if __name__ == '__main__':
    pass
