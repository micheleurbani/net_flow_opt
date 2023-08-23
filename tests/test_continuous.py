
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from net_flow_opt.utils import components, structure, activities_duration
from net_flow_opt.system import System
from net_flow_opt.scheduler import  Plan, Activity
from net_flow_opt.continuous_model import ContinuousModel
pop_size = 150
termination = ('n_gen', 20)
seed = 1124
resources = 3

system = System(structure, resources, components)

dates = [c.x_star for c in components]

original_activities = [
    Activity(component, date, duration)
    for component, date, duration in zip(system.components, dates, activities_duration)
]

original_plan = Plan(
    system=system,
    activities=original_activities
)

problem = ContinuousModel(
    system=system,
    original_plan=original_plan,
    resources=resources,
)

algorithm = NSGA2(
    pop_size=pop_size,
    eliminate_duplicates=True,
)

res = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=termination,
    seed=seed,
    verbose=True
)