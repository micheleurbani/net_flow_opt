from os import listdir
from pickle import load


from dash import callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


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

pareto_front = html.Div(
    [
        dcc.Graph(id="pareto-plot"),
    ],
    className="container"
)

# loading_pareto = dcc.Loading(
#     id="loading-pareto",
#     type="default",
#     children=[pareto_front, ],
# )

solution_analysis_contents = html.Div(
    id="solution_analysis_contents",
    children=[
        load_model,
        pareto_front,
    ],
    className="container",
)

def solution_analysis_callbacks(app):

    @app.callback(
        Output("pareto-plot", "figure"),
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
            return ga.pareto_front()
        elif plot_type == "pevo":
            return ga.pareto_evolution()
