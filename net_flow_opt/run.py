
import numpy as np
import pandas as pd
import networkx as nx
from pickle import load
from copy import deepcopy
from os import listdir, mkdir
from datetime import datetime
import random

from net_flow_opt.system import Component, System
from net_flow_opt.moga import MOGA, MOGAResults
from net_flow_opt.scheduler import Activity, Plan
from net_flow_opt.utils import components, structure, activities_duration


resources = [2, 3, 4, 5]


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


def randomized_experiment(experiment_index, seed):
    random.seed(seed)
    np.random.seed(seed)
    # Create a folder where to save the experiment
    dir_name = f"results/"

    # Change some properties of components
    random_components = []
    capacity = [20, 15, 9, 11, 7, 8, 15, 21, 10, 9]
    # for i in range(10):
    #     c = Component(
    #         cc=random() * 50 + 50,
    #         cp=random() * 30 + 20,
    #         alpha=random() * 30 + 30,
    #         beta=random() * 2 + 1.1,
    #         capacity=capacity[i]
    #     )
    #     random_components.append(c)
    random_components = components

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
            init_pop_size=200,
            p_mutation=0.3,
            n_generations=150,
            maintenance_plan=plan,
            parallel=True,
        )
        if r == resources[0]:
            initial_population = None
        moga.run(initial_population=initial_population)
        moga.save(dir_name + f"/r{r}_{seed}")
        results = MOGAResults(moga)
        df = results.to_dataframe()
        df = df[df.generation == df.generation.max()]  # Keep only the last gen
        df = df[df["rank"] == 1]  # Keep only point on the Pareto front
        df = df.filter(["IC", "LF"])  # Keep only the score
        df = df.drop_duplicates()
        df = df.sort_values(by="IC", ascending=False)
        fname = f"/r{r}_{seed}.csv"
        df.to_csv(dir_name + fname, index=False)
        initial_population = deepcopy(moga.population_history[-1])


def repeat_random_experiment(n):
    """Repeat a random experiment n times."""
    for i in range(n):
        randomized_experiment(i, i)


def experiment_from_old_data():
    seed(12345)
    np.random.seed(12345)

    for r in [2, 3, 4]:
        with open("results/plan.pkl", "rb") as f:
            plan = load(f)

        moga = MOGA(
            init_pop_size=200,
            p_mutation=0.3,
            n_generations=150,
            maintenance_plan=plan,
            parallel=False,
        )

        # Change the number of available resources before to run the algorithm
        moga.plan.system.resources = r
        # Run the experiment again, without initial population
        moga.run()
        # Save results
        if "paper" not in listdir("results"):
            mkdir("results/paper")
        moga.save("results/paper/r{}".format(r))
        result = MOGAResults(moga)
        df = result.to_dataframe()
        df = df[df.generation == df.generation.max()]  # Keep only the last gen
        df = df[df["rank"] == 1]  # Keep only point on the Pareto front
        df = df.filter(["IC", "LF"])  # Keep only the score
        df = df.drop_duplicates()
        df = df.sort_values(by="IC", ascending=False)
        fname = "results/paper/r{}.csv".format(r)
        df.to_csv(fname, index=False)


def hypervolume_multiple_experiments(results, n_samples):
    """
    Wrapper to run the hypervolume test on multiple instances of the experiment
    with common bounds for generation of samples.

    :param list results: a list of :class:`core.moga.MOGAResults`.
    :param int n_samples: the number of samples to use to run the HV test.
    """
    np.random.seed(12345)
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
    for r in results:
        hv.append(r.hypervolume_indicator(
            n_samples, lf_max, lf_min, ic_max, ic_min
        ))
    data = {i: hv for i, hv in enumerate(hv)}
    df = pd.DataFrame(data)
    df.to_csv("results/more_resources/hv_values.csv")


if __name__ == "__main__":
    # experiment_from_old_data()
    # results = []
    # for r in [3]:
    #     with open(f"results/r{r}.pkl", "rb") as f:
    #         moga = load(f)
    #     results.append(MOGAResults(moga))
    # hypervolume_multiple_experiments(results, 1000000)
    repeat_random_experiment(30)
