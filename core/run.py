
from copy import deepcopy
from random import random, seed

from core.moga import MOGA
from core.system import System
from core.utils import components, structure, activities_duration
from core.scheduler import Activity, Plan


seed(123)
resources = [2, 3, 4]
for r in resources:
    activities = [
        Activity(
            component=c,
            date=c.x_star,
            duration=activities_duration[i],
        ) for i, c in enumerate(components)
    ]

    system = System(
        structure=structure,
        resources=r,
        components=components,
    )

    plan = Plan(
        activities=activities,
        system=system,
    )

    moga = MOGA(
        init_pop_size=100,
        p_mutation=0.2,
        n_generations=150,
        maintenance_plan=plan,
        parallel=True,
    )
    if r == resources[0]:
        initial_population = None
    moga.run(initial_population=initial_population)
    moga.save("r{}_2701_200iter_100ind_20p".format(r))
    initial_population = deepcopy(moga.population_history[-1])
