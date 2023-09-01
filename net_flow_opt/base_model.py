
import numpy as np

from pymoo.core.problem import ElementwiseProblem

from net_flow_opt.system import System
from net_flow_opt.scheduler import Plan, Activity


class BaseModel(ElementwiseProblem):

    def __init__(self, system: System, resources: int, xu, xl, vtype) -> None:

        self.system = system
        self.r = resources

        super().__init__(
            n_var=system.N,
            n_obj=2,
            n_ieq_constr=system.N,
            xl=xl,
            xu=xu,
            vtype=float,
        )
