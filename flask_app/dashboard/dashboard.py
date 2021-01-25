from dash import Dash
import dash_html_components as html
import dash_bootstrap_components as dbc

from .layout import html_layout
from .contents.overview import overview
from .contents.solution_analysis import (solution_analysis_contents,
                                         solution_analysis_callbacks)
from .contents.moga_settings import (moga_settings_contents,
                                     moga_settings_callbacks)


def init_dashboard(server, cache):
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
    card_tabs = dbc.Tabs(
        id="homepage-tabs",
        children=[
            dbc.Tab(overview, label="Overview"),
            # dbc.Tab(components_contents, label="Components"),
            dbc.Tab(moga_settings_contents, label="MOGA settings"),
            dbc.Tab(
                solution_analysis_contents,
                id="tab-solution-analysis",
                label="Solution analysis")
        ]
    )

    dash_app.layout = html.Div(
        [
            card_tabs,
        ],
    )

    # Initialize callbacks after our app is loaded
    # Pass dash_app as a parameter
    init_callbacks(dash_app, cache)

    return dash_app.server


def init_callbacks(app, cache):
    moga_settings_callbacks(app)
    solution_analysis_callbacks(app, cache)
