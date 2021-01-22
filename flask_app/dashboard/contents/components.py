import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

load_components = html.Div(id='load-components-list', children=[
    html.Br(),
    html.H3("List of components"),
    html.P("Select a list of components from the dropdown menu."),
    dcc.Dropdown(
        id='systems-dropdown',
        placeholder='Select a list of components...',
        # value=list_components()[0]['value'],
        # options=list_components()
    )
])

components_table = html.Div(children=[
    html.H3("Component Parameters"),
    html.P("The paramtes of components can be declared using the table below.\
        The meaning of each paramter is explained in the following."),
    dash_table.DataTable(
        id='table-component-parameters',
        editable=True,
        row_deletable=True
    ),
    html.Br(),
    dbc.Button("Save parameters", id='save-button', color="primary"),
    dbc.Modal(
        [
            dbc.ModalHeader("Save components list"),
            dbc.ModalBody(
                [
                    "Provide a name for the list of components",
                    dbc.Input(
                        id="system-name-input",
                        type="text",
                        placeholder="Provide a name for the system.",
                    ),
                ]
            ),
            dbc.ModalFooter([
                dbc.Button(
                    "Save",
                    id="btn-system-save",
                    className="mr-auto",
                    color='primary'),
                dbc.Button(
                    "Close",
                    id="btn-system-close",
                    className="ml-auto",
                    color='primary')
            ]),
        ],
        id='modal-system',
        size='lg'
    )
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
    html.Div(children=[
        dcc.Dropdown(
            id='components-dropdown',
            placeholder='Select a component...',
            multi=True
        ),
        components_plots
    ]),
    html.Div(id="hidden-div", style={"display": "none"}),
])
