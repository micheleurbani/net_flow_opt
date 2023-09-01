
import numpy as np
from copy import deepcopy

from net_flow_opt.system import System
from net_flow_opt.scheduler import Plan, Activity
from net_flow_opt.base_model import BaseModel


class DiscreteModel(BaseModel):

    def __init__(self, system: System, original_plan: Plan, resources: int
    ) -> None:

        self.original_plan = original_plan

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
            activities=deepcopy(activities),
            system=deepcopy(self.system),
            grouping_structure=grouping_structure,
            original_plan=deepcopy(self.original_plan),
        )

        out["F"] = np.array([plan.IC, plan.LF])
        out["G"] = np.array([
            np.sum([
                b.t <= a.t <= b.t + b.d for b in plan.activities
            ]) - self.r for a in plan.activities
        ])


if __name__ == '__main__':
    pass
