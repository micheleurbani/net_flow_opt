import dash_html_components as html
import dash_core_components as dcc


with open('flask_app/dashboard/contents/overview.md', 'r') as f:
    overview = f.read()

overview = html.Div(children=[
    dcc.Markdown(overview)
])
