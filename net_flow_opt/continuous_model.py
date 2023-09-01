
import numpy as np
from copy import deepcopy

from pymoo.core.problem import ElementwiseProblem

from net_flow_opt.system import System
from net_flow_opt.base_model import BaseModel
from net_flow_opt.scheduler import Plan, Activity


class ContinuousModel(BaseModel):

    def __init__(self, system: System, resources: int) -> None:

        super().__init__(
            system=system,
            resources=resources,
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
            activities=deepcopy(activities),
            system=deepcopy(self.system),
        )

        out["F"] = np.array([plan.IC, plan.LF])
        out["G"] = np.array([
            np.sum([
                b.t <= a.t <= b.t + b.d for b in plan.activities
            ]) - self.r for a in plan.activities
        ])


if __name__ == '__main__':
    pass
