
import numpy as np

from pymoo.core.problem import ElementwiseProblem

from net_flow_opt.system import System
from net_flow_opt.scheduler import Plan, Activity


class DiscreteModel(ElementwiseProblem):

    def __init__(self, system: System, original_plan: Plan, resources: int
    ) -> None:

        self.system = system
        self.original_plan = original_plan
        self.r = resources

        super().__init__(
            n_var=system.N,
            n_obj=2,
            n_ieq_constr=system.N,
            xl=1,
            xu=system.N,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # one-hot encoding of x
        grouping_structure = np.zeros((self.system.N, self.system.N))
        grouping_structure[np.arange(self.system.N), x - 1] = 1

        # retrieve activities durations
        durations = [a.d for a in self.original_plan.activities]

        # define activities
        activities = [
            Activity(
                component=component,
                date=t,
                duration=d,
            )
        for component, t, d in zip(self.system.components, x, durations)]

        # define maintenance plan
        plan = Plan(
            activities=activities,
            system=self.system,
            grouping_structure=grouping_structure,
            original_plan=self.original_plan,
        )

        out["F"] = np.array([plan.IC, plan.LF])
        out["G"] = np.array([
            np.sum([
                b.t < a.t <= b.t + b.d for b in plan.activities
            ]) - self.r for a in plan.activities
        ])


if __name__ == '__main__':
    pass
