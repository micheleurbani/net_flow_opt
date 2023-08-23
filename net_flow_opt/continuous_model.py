
import numpy as np

from pymoo.core.problem import ElementwiseProblem

from net_flow_opt.system import System
from net_flow_opt.scheduler import Plan, Activity


class ContinuousModel(ElementwiseProblem):

    def __init__(self, system: System, original_plan: Plan, resources: int
    ) -> None:

        self.system = system
        self.original_plan = original_plan
        self.r = resources

        super().__init__(
            n_var=system.N,
            n_obj=2,
            n_ieq_constr=system.N,
            xl=0,
            xu=max([c.x_star for c in self.system.components]) * 1.5,
            vtype=float,
        )

    def _evaluate(self, x, out, *args, **kwargs):

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
