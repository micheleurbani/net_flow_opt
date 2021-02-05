
import pandas as pd
import networkx as nx
from pickle import load
from copy import deepcopy
from os import listdir, mkdir
from datetime import datetime
from random import seed, random

from core.system import Component, System
from core.moga import MOGA, MOGAResults
from core.scheduler import Activity, Plan
from core.utils import components, structure, activities_duration


seed(123)
resources = [2, 3, 4]


def simple_experiment():
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
            init_pop_size=5,
            p_mutation=0.3,
            n_generations=5,
            maintenance_plan=plan,
            parallel=True,
        )
        if r == resources[0]:
            initial_population = None
        moga.run(initial_population=initial_population)
        moga.save("r{}_".format(r))
        initial_population = deepcopy(moga.population_history[-1])


def randomized_experiment(experiment_index):
    # Create a folder where to save the experiment
    dir_name = f"results/exp_{experiment_index}_{datetime.now().day}" + \
                f"{datetime.now().month}_{datetime.now().hour}" + \
                f"{datetime.now().minute}"
    if dir_name not in ["results/" + p for p in listdir("results")]:
        mkdir(dir_name)
    # Change some properties of components
    random_components = []
    capacity = [20, 15, 9, 11, 7, 8, 15, 21, 10, 9]
    for i in range(10):
        c = Component(
            cc=random() * 50 + 50,
            cp=random() * 30 + 20,
            alpha=random() * 30 + 30,
            beta=random() * 2 + 1.1,
            capacity=capacity[i]
        )
        random_components.append(c)

    structure = nx.DiGraph()
    # Add source and sink nodes
    structure.add_nodes_from(["s", "t"])
    structure.add_nodes_from(random_components)
    structure.add_edges_from([
        ("s", random_components[0]),
        ("s", random_components[1]),
        (random_components[0], random_components[2]),
        (random_components[0], random_components[3]),
        (random_components[1], random_components[4]),
        (random_components[1], random_components[5]),
        (random_components[2], random_components[6]),
        (random_components[3], random_components[6]),
        (random_components[3], random_components[7]),
        (random_components[4], random_components[7]),
        (random_components[5], random_components[7]),
        (random_components[6], "t"),
        (random_components[7], random_components[8]),
        (random_components[7], random_components[9]),
        (random_components[8], "t"),
        (random_components[9], "t"),
    ])

    activities = [
        Activity(
            component=c,
            date=c.x_star,
            duration=activities_duration[i],
        ) for i, c in enumerate(random_components)
    ]

    for r in resources:
        system = System(
            structure=structure,
            resources=r,
            components=random_components,
        )

        plan = Plan(
            activities=activities,
            system=system,
        )

        moga = MOGA(
            init_pop_size=5,
            p_mutation=0.3,
            n_generations=5,
            maintenance_plan=plan,
            parallel=True,
        )
        if r == resources[0]:
            initial_population = None
        moga.run(initial_population=initial_population)
        moga.save(dir_name + "/r{}".format(r))
        results = MOGAResults(moga)
        df = results.to_dataframe()
        df = df[df.generation == df.generation.max()]  # Keep only the last gen
        df = df[df["rank"] == 1]  # Keep only point on the Pareto front
        df = df.filter(["IC", "LF"])  # Keep only the score
        df = df.drop_duplicates()
        df = df.sort_values(by="IC", ascending=False)
        fname = "/r{}.csv".format(r)
        df.to_csv(dir_name + fname, index=False)
        initial_population = deepcopy(moga.population_history[-1])


def repeat_random_experiment(n):
    """Repeat a random experiment n times."""
    for i in range(n):
        randomized_experiment(i)


def experiment_from_old_data():
    results = []
    for r in [2, 3, 4]:
        with open("results/exp_17_42_335/r{r}.pkl", "rb") as f:
            moga = load(f)

        # Clear old solutions
        moga.population_history = []

        # Run the experiment again, without initial population
        moga.run()
        # Save results
        if "paper" not in listdir("results"):
            mkdir("results/paper")
        moga.save("results/paper/r{}".format(r))
        result = MOGAResults(moga)
        df = results.to_dataframe()
        df = df[df.generation == df.generation.max()]  # Keep only the last gen
        df = df[df["rank"] == 1]  # Keep only point on the Pareto front
        df = df.filter(["IC", "LF"])  # Keep only the score
        df = df.drop_duplicates()
        df = df.sort_values(by="IC", ascending=False)
        fname = "results/paper/r{}.csv".format(r)
        df.to_csv(fname, index=False)
        results.append(result)
    return results

def hypervolume_multiple_experiments(results):
    for r in results:
        assert type(r) is MOGAResults
    # Find bounds for calculation of the hypervolume
    lf_max, lf_min, ic_max, ic_min = 0.0, 1e4, 0.0, 1e4
    for result in results:
        for generation in result.moga.population_history:
            for ind in generation:
                if ind.plan.LF > lf_max:
                    lf_max = ind.plan.LF
                elif ind.plan.LF < lf_min:
                    lf_min = ind.plan.LF
                if ind.plan.IC > ic_max:
                    ic_max = ind.plan.IC
                elif ind.plan.IC < ic_min:
                    ic_min = ind.plan.IC
    # Compute hypervolumes
    hv = []
    for r in result:
        hv.append(r.hypervolume_indicator())
    data = {i: hv for i, hv in enumerate(hv)}
    df = pd.DataFrame(data)
    df.to_csv("results/paper/hv_values.csv", index=False)


if __name__ == "__main__":
    results = experiment_from_old_data()
    hypervolume_multiple_experiments(results)
