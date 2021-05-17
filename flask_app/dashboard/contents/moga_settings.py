
import pandas as pd
import networkx as nx
from random import random
from numpy import product
from ast import literal_eval
from datetime import datetime
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


from .components import load_systems
from .solution_analysis import load_models
from core.moga import MOGA
from core.system import Component, System
from core.scheduler import Activity, Plan


system_selection = dbc.FormGroup(
    [
        dbc.Label("System model", html_for="system-model-dropdown", width=2),
        dbc.Col(
            dcc.Dropdown(
                id="system-model-dropdown",
                options=load_systems()
            ),
            width=10
        ),
    ],
    row=True
)

experiment_name = dbc.FormGroup(
    [
        dbc.Label("Experiment name", html_for="experiment-name-input",
                  width=2),
        dbc.Col(
            dbc.Input(
                id="experiment-name-input",
                type="text",
                placeholder="Optional. Provide a name for the experiment.",
            ),
            width=10
        ),
    ],
    row=True
)

p_mutation = dbc.FormGroup(
    [
        dbc.Label(
            "Mutation probability",
            html_for="p_mutation-input"
        ),
        dbc.Col(
            dbc.Input(id="p_mutation-input",
                      type="number", value="20"),
            width=10
        ),
        dbc.FormText("Provide the mutation probability as a percentage."),
        dbc.FormFeedback("Mutation probability set successfully.", valid=True),
        dbc.FormFeedback(
            "Provide a floating point number between 0 and 1.",
            valid=False,
        ),
    ],
    row=True
)

initial_population = dbc.FormGroup(
    [
        dbc.Label("Population size", html_for="population-size-input",
                  width=2),
        dbc.Col(
            dbc.Input(id="population-size-input", type="number",
                     placeholder="The population size", value="40", min=1),
            width=10,
        ),
        dbc.FormFeedback(
            "Population size set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ],
    row=True
)

seed = dbc.FormGroup(
    [
        dbc.Label("Seed value", html_for="seed-input", width=2),
        dbc.Col(
            dbc.Input(id="seed-input", type="number", value="123", min=0),
            width=10
        ),
        dbc.FormFeedback(
            "Seed set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ],
    row=True
)

p_mutation = dbc.FormGroup(
    [
        dbc.Label("Mutation probability", html_for="p_mutation-input",
                  width=2),
        dbc.Col(
            dbc.Input(id="p_mutation-input", type="number", value="20"),
            width=10
        ),
        dbc.FormFeedback(
            "Mutation probability set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a floating point number between 0 and 1.",
            valid=False,
        ),
    ],
    row=True
)

resources = dbc.FormGroup(
    [
        dbc.Label("Available resources", html_for="resources-input", width=2),
        dbc.Col(
            dbc.Input(id="resources-input", type="number", value="2"),
            width=10
        ),
        dbc.FormFeedback(
            "Number of crews set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ],
    row=True
)

generations = dbc.FormGroup(
    [
        dbc.Label("Number of generations", html_for="generations-input",
                  width=2),
        dbc.Col(
            dbc.Input(id="generations-input", type="number", value="10"),
            width=10
        ),
        dbc.FormFeedback(
            "Number of generations set successfully.", valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ],
    row=True
)

graph_loader = dcc.Loading(
    id="loading-1",
    children=[
        html.Div(id="hidden-div2", style={"display": "none"})
        ],
    type="default")

button_row = html.Div(
    id='button-row',
    children=[
        dbc.Button("Run the algorithm", id='start-algorithm', color="primary"),
        graph_loader,
    ]
)

form = dbc.Form([
        html.Br(),
        system_selection,
        html.Br(),
        experiment_name,
        html.Br(),
        initial_population,
        html.Br(),
        seed,
        html.Br(),
        p_mutation,
        html.Br(),
        resources,
        html.Br(),
        generations,
        html.Br(),
        button_row,
    ]
)

moga_settings_contents = html.Div(form)


def moga_settings_callbacks(app):
    @app.callback(
        Output('hidden-div2', 'children'),
        Input('start-algorithm', 'n_clicks'),
        [State('experiment-name-input', 'value'),
         State('population-size-input', 'value'),
         State('seed-input', 'value'),
         State('p_mutation-input', 'value'),
         State('resources-input', 'value'),
         State('generations-input', 'value'),
         State('system-model-dropdown', 'value')]
    )
    def run_algorithm(
        n_clicks,
        experiment_name,
        pop_size,
        seed,
        p_mutation,
        resources,
        n_generations,
        system_fname
    ):
        discriminants = [pop_size, seed, p_mutation, system_fname,
                         resources, n_generations, n_clicks]
        discriminants = [0 if x is None else 1 for x in discriminants]

        if not product(discriminants):
            raise PreventUpdate
        # Cast variables
        pop_size = int(pop_size)
        seed = int(seed)
        p_mutation = int(p_mutation)/100
        resources = int(resources)
        n_generations = int(n_generations)

        # Load system data
        sys_data = pd.read_csv("data/systems_data/{}.csv".format(system_fname))
        # Declare components
        components = [
            Component(
                cc=sys_data.loc[i].C_c,
                cp=sys_data.loc[i].C_p,
                alpha=sys_data.loc[i].alpha,
                beta=sys_data.loc[i].beta,
                capacity=sys_data.loc[i].capacity
            )
            for i in range(1, len(sys_data) - 1)
        ]
        # Declare system structure
        structure = nx.DiGraph()
        structure.add_nodes_from(["s", "t"])
        structure.add_nodes_from(components)
        for i in range(len(sys_data) - 1):
            for j in literal_eval(sys_data.loc[i].successors):
                if i == 0:
                    structure.add_edge(
                        "s", components[int(j) - 1]
                    )
                else:
                    if j == "t":
                        structure.add_edge(
                            components[int(sys_data.loc[i].label) - 1], "t"
                        )
                    else:
                        structure.add_edge(
                            components[int(sys_data.loc[i].label) - 1],
                            components[int(j) - 1]
                        )
        # Declare the system
        network_system = System(
            structure=structure,
            resources=resources,
            components=components,
        )
        network_system.plot_system_structure()
        # Declare a plan
        plan = Plan(
            activities=[Activity(c, c.x_star, random() * 3 + 1) for c
                        in network_system.components],
            system=network_system,
        )
        # Run the algorithm anf find a Pareto front
        ga = MOGA(
            init_pop_size=pop_size,
            p_mutation=p_mutation,
            n_generations=n_generations,
            maintenance_plan=plan,
            parallel=True,
        )
        ga.run()

        if not experiment_name:
            experiment_name = "experiment_" + \
                datetime.now().strftime("%Y%m%d_%H%M%S") + ".pkl"
        ga.save(fname=experiment_name)

        return ['waiting']
