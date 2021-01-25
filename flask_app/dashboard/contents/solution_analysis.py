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
    return [{'label': exp[:-4], 'value': exp} for exp
            in model_names if exp[-4:] == '.pkl']


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

pareto_front = html.Div(
    id="div-pareto",
    className="container",
)

solution_analysis_contents = html.Div(
    id="solution_analysis_contents",
    children=[
        load_model,
        pareto_front,
        load_solution,
    ],
    className="container",
)


def solution_analysis_callbacks(app):
    @app.callback(
        Output("div-pareto", "children"),
        [Input("models-dropdown", "value"),
         Input("pareto-dropdown", "value")]
    )
    def load_pareto(model_name, plot_type):
        # Load the MOGA object from the file
        if not model_name or not plot_type:
            raise PreventUpdate
        with open('results/' + model_name, 'rb') as f:
            ga = MOGAResults(load(f))

        if plot_type == "pfront":
            return [dcc.Graph(id="pareto-plot", figure=ga.pareto_front())]
        elif plot_type == "pevo":
            return [dcc.Graph(id="pareto-plot", figure=ga.pareto_evolution())]
        return []

    @app.callback(
        Output("load-solution-input", "children"),
        Input("pareto-plot", "figure")
    )
    def load_solution_layout(figure):
        if figure is None:
            return []
        else:
            return solution_layout
