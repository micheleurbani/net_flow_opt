from os import listdir
from pickle import load


import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output


from core.moga import MOGAResults


def load_models():
    model_names = listdir('./results/')
    model_names = [{'label': exp[:-4], 'value': exp} for exp
                   in model_names if exp[-4:] == '.pkl']
    return model_names


form_model = dbc.FormGroup(
    [
        dbc.Label("Select the experiment", html_for="models-dropdown"),
        dcc.Dropdown(
            id="models-dropdown",
            multi=False,
            options=load_models()
        )
    ]
)

form_solution = dbc.FormGroup(
    [
        dbc.Label("Select a plan using the ID", html_for="solution-dropdown"),
        dcc.Dropdown(
            id="solution-dropdown",
            multi=False,
        )
    ]
)

form_pareto_type = dbc.FormGroup(
    [
        dbc.Label("Select plot type", html_for="pareto-dropdown"),
        dcc.Dropdown(
            id="pareto-dropdown",
            multi=False,
            options=[
                {"label": "Pareto front", "value": "pfront"},
                {"label": "Pareto evolution", "value": "pevo"},
            ],
        )
    ]
)

form_solution_type = dbc.FormGroup(
    [
        dbc.Label(
            "Select PM plan or optimized plan",
            html_for="solution-type"
        ),
        dcc.Dropdown(
            id="solution-type",
            multi=False,
            options=[
                {"label": "PM schedule", "value": "PM"},
                {"label": "Optimized schedule", "value": "optim"},
            ],
        )
    ]
)

load_model = html.Div(
    id="load-model-input",
    className="container",
    children=[
        html.Br(),
        html.Div(
            [
                html.Div(
                    [
                        form_model,
                    ],
                    className="col"
                ),
                html.Div(
                    [
                        form_pareto_type,
                    ],
                    className="col"
                )
            ],
            className="row"
        ),
    ]
)

solution_layout = [
    html.Br(),
    html.Div(
        [
            html.Div(
                [
                    form_solution,
                ],
                className="col"
            ),
            html.Div(
                [
                    form_solution_type,
                ],
                className="col"
            )
        ],
        className="row",
    )
]

load_solution = html.Div(
    id="load-solution-input",
    className="container",
)

pareto_front = dcc.Loading(
    id="loading-pareto",
    type="default",
    children=html.Div(
        id="div-pareto",
        className="container",
    )
)

gantt_chart = html.Div(
    id="gantt-chart-div",
    className="container",
)

flow_plot = html.Div(
    id="flow-plot-div",
    className="container",
)

solution_analysis_contents = html.Div(
    id="solution_analysis_contents",
    children=[
        load_model,
        pareto_front,
        load_solution,
        gantt_chart,
        flow_plot,
    ],
    className="container",
)


def solution_analysis_callbacks(app, cache):

    @cache.memoize()
    def load_data(model_name):
        with open('results/' + model_name, 'rb') as f:
            ga = MOGAResults(load(f))
        return ga

    @cache.memoize()
    def load_last_generation(model_name):
        with open('results/' + model_name, 'rb') as f:
            ga = MOGAResults(load(f))
        last_generation = ga.moga.population_history[-1]
        original_plan = ga.moga.plan
        return original_plan, last_generation

    @app.callback(
        [Output("div-pareto", "children"),
         Output("load-solution-input", "children")],
        [Input("models-dropdown", "value"),
         Input("pareto-dropdown", "value")]
    )
    def load_pareto(model_name, plot_type):
        # Load the MOGA object from the file
        if model_name is None or plot_type is None:
            raise PreventUpdate
        ga = load_data(model_name)

        if plot_type == "pfront":
            fname = ga.pareto_to_csv()
            return (
                [
                    dcc.Graph(id="pareto-plot", figure=ga.pareto_front()),
                    html.A("Export CSV", href="download/" + fname)
                ],
                solution_layout
            )
        elif plot_type == "pevo":
            return ([dcc.Graph(id="pareto-plot",
                     figure=ga.pareto_evolution())], [])
        return [], []

    @app.callback(
        Output("solution-dropdown", "options"),
        [Input("pareto-dropdown", "value"),
         Input("pareto-plot", "figure")]
    )
    def load_solutions_dropdown(plot_type, figure):
        if plot_type is None or plot_type == "pevo":
            raise PreventUpdate
        return [{'label': "Plan ID: {}".format(figure['data'][0]
                ['hovertext'][i]), 'value': figure['data'][0]['hovertext'][i]}
                for i, y in enumerate(figure['data'][0]['y']) if y is not None]

    @app.callback(
        [Output("gantt-chart-div", "children"),
         Output("flow-plot-div", "children")],
        [Input("solution-dropdown", "value"),
         Input("solution-type", "value"),
         Input("models-dropdown", "value"),
         Input("pareto-dropdown", "value")]
    )
    def load_solution(solution_id, solution_type, model_name, pareto_type):
        if solution_id is None or solution_type is None:
            raise PreventUpdate
        plan, population = load_last_generation(model_name)
        if pareto_type == "pevo":
            return [], []
        else:
            if solution_type == "PM":
                gantt = [
                    dcc.Graph(
                        id="gantt-chart",
                        figure=plan.plot_gantt_chart()
                    )
                ]
                flow = [
                    dcc.Graph(
                        id="flow-plot",
                        figure=plan.plot_flow_history()
                    )
                ]
                return gantt, flow
            elif solution_type == "optim":
                gantt = [
                    dcc.Graph(
                        id="gantt-chart",
                        figure=population[solution_id].plan.plot_gantt_chart()
                    )
                ]
                flow = [
                    dcc.Graph(
                        id="flow-plot",
                        figure=population[solution_id].plan.plot_flow_history()
                    )
                ]
                return gantt, flow
            else:
                return [], []
