
import os
import ast
import numpy as np
import pandas as pd

import dash_table
import dash_cytoscape as cyto
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def load_systems():
    fnames = [{"label": i[:-4], "value": i[:-4]} for i in
              os.listdir("data/systems_data")]
    if fnames:
        return fnames
    else:
        return []

system_model = pd.DataFrame(
    data={
        "label" : ["s", "t"],
        "alpha" : [0.0, 0.0],
        "beta" : [0.0, 0.0],
        "C_p" : [0.0, 0.0],
        "C_c" : [0.0, 0.0],
        "capacity": [0.0, 0.0],
        "duration": [0.0, 0.0],
        "successors":['0', '0']
    }
)

load_components = html.Div(id='load-components-list', children=[
    html.Br(),
    html.H3("Select a system model or create a new one"),
    html.P("Each item corresponds to a saved system model with components \
           and a system structure."),
    dcc.Dropdown(
        id='systems-dropdown',
        placeholder='Select a list of components...',
        options=load_systems()
    )
])

components_table = html.Div(children=[
    html.H3("Component Parameters"),
    html.P("The paramtes of components can be declared using the table below.\
        The meaning of each paramter is explained in the following."),
    dbc.Button("Add Row", id='editing-rows-button',
               color="primary", n_clicks=0),
    html.Br(),
    html.Br(),
    html.Div(id="div-table", children=[
    dash_table.DataTable(
        id='table-component-parameters',
        editable=True,
        row_deletable=True,
        columns=[{"name": i, "id": i} for i in system_model.columns],
        data=system_model.to_dict('records'),
    )]),
    html.Br(),
    dbc.InputGroup(
        [
            dbc.Input(
                id="system-fname",
                placeholder="Type system name",
                debounce=True,
                minLength=3),
            dbc.InputGroupAddon(
                dbc.Button("Save Components", id='save-button',
                           color="primary", n_clicks=0),
            )
        ]
    ),
])

components_plots = html.Div(id='components-features', children=[
    html.Br(),
    html.Center(
        dbc.ButtonGroup([
            dbc.Button('PDF', id='btn-components-pdf', n_clicks=0,
                       outline=True, color='primary'),
            dbc.Button('CDF', id='btn-components-cdf', n_clicks=0,
                       outline=True, color='primary'),
            dbc.Button('Failure rate', id='btn-components-failure-rate',
                       n_clicks=0, outline=True, color='primary'),
        ])
    ),
    dcc.Graph(id='components-plot')
])

components_contents = html.Div(children=[
    load_components,
    html.Br(),
    components_table,
    html.Br(),
    html.Div(id='div-graph'),
    html.Div(id="hidden-div")
])

def components_contents_callbacks(app):

    @app.callback(
        [Output('div-table', 'children'),
        Output('system-fname', 'value')],
        Input('systems-dropdown', 'value')
    )
    def load_system(fname):
        if not fname:
            raise PreventUpdate
        df = pd.read_csv("data/systems_data/{}.csv".format(fname))
        table = dash_table.DataTable(
            id='table-component-parameters',
            editable=True,
            row_deletable=True,
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records')
        )
        return table, fname

    @app.callback(
        Output('table-component-parameters', 'data'),
        Input('editing-rows-button', 'n_clicks'),
        [State('table-component-parameters', 'data'),
        State('table-component-parameters', 'columns')]
    )
    def add_row(n_clicks, rows, columns):
        if n_clicks > 0:
            rows.insert(-1, {c['id']: '' for c in columns})
        return rows

    @app.callback(
        Output('hidden-div', 'children'),
        [Input('save-button', 'n_clicks'),
        Input('system-fname', 'value')],
        [State('table-component-parameters', 'data'),
        State('table-component-parameters', 'columns')]
    )
    def save_system(n_clicks, fname, rows, columns):
        if not fname:
            raise PreventUpdate
        else:
            df = pd.DataFrame(data=rows)
            df.to_csv("data/systems_data/{}.csv".format(fname), index=False)
        raise PreventUpdate

    @app.callback(
        Output('systems-dropdown', 'options'),
        Input('save-button', 'n_clicks')
    )
    def update_dropdown_systems(n_clicks):
        if n_clicks > 0:
            return load_systems()

    @app.callback(
        Output('div-graph', 'children'),
        Input('table-component-parameters', 'data')
    )
    def define_graph(data):
        # Add nodes
        elements = [
            {'data': {'id': node['label'], 'label': node['label']}}
            for node in data
        ]
        # Add edges
        edges = []
        for node in data:
            if node['successors'] != '0':
                for target in ast.literal_eval(node['successors']):
                    edges.append(
                        {'data': {
                            'id': '#' + node['label'] + target,
                            'source': node['label'],
                            'target': target}
                        }
                    )
        # Add arrows
        stylesheet = [
            {
                'selector': edge['data']['id'],
                'style': {'target-arrow-shape': 'triangle'}
            }
            for edge in edges
        ]
        # Add labels
        stylesheet.append(
            {
                'selector': 'node',
                'style': {
                    'label': 'data(id)'
                }
            }
        )
        # Create the graph object
        graph = cyto.Cytoscape(
            id='cytoscape-graph',
            layout={
                'name': 'cose',
                'roots': '#s, #t'
            },
            style={'width': '100%', 'height': '400px'},
            elements=elements + edges,
            stylesheet=stylesheet
        )
        return graph
