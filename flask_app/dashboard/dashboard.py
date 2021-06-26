from dash import Dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from .layout import html_layout
from .contents.overview import overview
from .contents.components import (components_contents,
                                  components_contents_callbacks)
from .contents.solution_analysis import (solution_analysis_contents,
                                         solution_analysis_callbacks)
from .contents.moga_settings import (moga_settings_contents,
                                     moga_settings_callbacks)


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/' +
            'dist/css/bootstrap.min.css',
        ]
    )

    dash_app.index_string = html_layout
    # Create Dash Layout

    card = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Overview", tab_id="tab-overview"),
                        dbc.Tab(label="Design system", tab_id="tab-design"),
                        dbc.Tab(
                            label="MOGA settings",
                            tab_id="tab-moga-settings"),
                        dbc.Tab(
                            label="Solution analysis",
                            tab_id="tab-solution-analysis")
                    ],
                    id="card-tabs",
                    card=True,
                    active_tab="tab-design"
                )
            ),
            dbc.CardBody(html.Div(id="card-content"))
        ]
    )

    dash_app.layout = html.Div(
        [
            html.Br(),
            card,
            html.Br(),
        ],
    )

    # Initialize callbacks after our app is loaded
    # Pass dash_app as a parameter
    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(app):
    moga_settings_callbacks(app)
    solution_analysis_callbacks(app)
    components_contents_callbacks(app)

    # callbacks for the landing page
    @app.callback(
        Output("card-content", "children"),
        [Input("card-tabs", "active_tab")]
    )
    def tab_content(active_tab):
        if active_tab == "tab-overview":
            return overview
        elif active_tab == "tab-design":
            return components_contents
        elif active_tab == "tab-moga-settings":
            return moga_settings_contents
        elif active_tab == "tab-solution-analysis":
            return solution_analysis_contents
        return None
