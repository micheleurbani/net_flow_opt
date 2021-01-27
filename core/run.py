
from random import random, seed

from core.moga import MOGA
from core.system import System
from core.utils import components, structure
from core.scheduler import Activity, Plan


seed(123)

for r in [2, 3, 4]:
    activities = [
        Activity(
            component=c,
            date=c.x_star,
            duration=random() * 3 + 1
        ) for c in components
    ]

    system = System(
        structure=structure,
        resources=r,
        components=components
    )

    plan = Plan(
        activities=activities,
        system=system,
    )

    moga = MOGA(
        init_pop_size=100,
        p_mutation=0.2,
        n_generations=200,
        maintenance_plan=plan,
        parallel=True,
    )

    moga.run()
    moga.save("r{}_2701_200iter_100ind_20p".format(r))
