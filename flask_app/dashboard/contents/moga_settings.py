from random import random
from numpy import product
from datetime import datetime
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


from core.moga import MOGA
from core.system import System
from core.scheduler import Activity, Plan
from core.utils import components, structure


experiment_name = dbc.FormGroup(
    [
        dbc.Label("Experiment name"),
        dbc.Input(
            id="experiment-name-input",
            type="text",
            placeholder="Optional. Provide a name for the experiment.",
        ),
        dbc.FormText("The name of the experiment. Optional."),
    ]
)

initial_population = dbc.FormGroup(
    [
        dbc.Label("Population size"),
        dbc.Input(id="population-size-input",
                  type="number", value="40", min=1),
        dbc.FormText("The size of the population. It requires a positive \
            integer number."),
        dbc.FormFeedback(
            "Population size set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ]
)

seed = dbc.FormGroup(
    [
        dbc.Label("Seed value"),
        dbc.Input(id="seed-input", type="number", value="123", min=0),
        dbc.FormText("Setting a seed value allows to replicate the\
                     experiment."),
        dbc.FormFeedback(
            "Seed set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ]
)

p_mutation = dbc.FormGroup(
    [
        dbc.Label("Mutation probability"),
        dbc.Input(id="p_mutation-input", type="number", value="20"),
        dbc.FormText("Provide the mutation probability as a percentage."),
        dbc.FormFeedback(
            "Mutation probability set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a floating point number between 0 and 1.",
            valid=False,
        ),
    ]
)

resources = dbc.FormGroup(
    [
        dbc.Label("Available resources"),
        dbc.Input(id="resources-input", type="number", value="2"),
        dbc.FormText("The number of available repairmen crews."),
        dbc.FormFeedback(
            "Number of crews set successfully.",
            valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ]
)

generations = dbc.FormGroup(
    [
        dbc.Label("Number of generations"),
        dbc.Input(id="generations-input", type="number", value="10"),
        dbc.FormText("The number of generations after which to stop."),
        dbc.FormFeedback(
            "Number of generations set successfully.", valid=True
        ),
        dbc.FormFeedback(
            "Provide a positive integer number.",
            valid=False,
        ),
    ]
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
        experiment_name,
        initial_population,
        seed,
        p_mutation,
        resources,
        generations,
        html.Br(),
        button_row,
    ]
)

moga_settings_contents = html.Div(form)


def moga_settings_callbacks(app):
    @app.callback(
        [Output('hidden-div2', 'children')],
        Input('start-algorithm', 'n_clicks'),
        [State('experiment-name-input', 'value'),
         State('population-size-input', 'value'),
         State('seed-input', 'value'),
         State('p_mutation-input', 'value'),
         State('resources-input', 'value'),
         State('generations-input', 'value')]
    )
    def run_algorithm(
        n_clicks,
        experiment_name,
        pop_size,
        seed,
        p_mutation,
        resources,
        n_generations,
    ):
        discriminants = [pop_size, seed, p_mutation,
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

        # Declare the system
        network_system = System(
            structure=structure,
            resources=resources,
            components=components,
        )

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
