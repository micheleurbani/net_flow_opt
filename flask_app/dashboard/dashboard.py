from dash import Dash
import dash_html_components as html
import dash_bootstrap_components as dbc

from .layout import html_layout
from .contents.overview import overview
from .contents.components import components_contents


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/\
                dist/css/bootstrap.min.css',
        ]
    )

    dash_app.index_string = html_layout
    # Create Dash Layout
    card_tabs = dbc.Tabs(
        id="homepage-tabs",
        children=[
            dbc.Tab(overview, label="Overview"),
            dbc.Tab(components_contents, label="Components"),
            dbc.Tab(label="MOGA settings"),
            dbc.Tab(label="Solution analysis")
        ]
    )

    dash_app.layout = html.Div(
        [
            card_tabs,
        ],
    )

    return dash_app.server
