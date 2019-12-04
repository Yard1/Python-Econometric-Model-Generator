import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import emg


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions=True
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Hr(),
    html.Div(id='opt-dropdown-div', children=[ 
        dcc.Dropdown(
            id='opt-dropdown',
            options=[{'value': 'test', 'label': 'test'}],
            value='test'
            ),
            ],style={'width': '20%', 'display': 'inline-block'}
        ),
    html.Div(id='display-selected-values'),
    html.Div(id='output-data-upload'),
    html.Div(id='formula-table-div'),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_json(date_format='iso', orient='split')


@app.callback(Output('intermediate-value', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_data(contents, filename):
    if contents:
        return parse_contents(contents, filename)

@app.callback(Output('output-data-upload', 'children'),
              [Input('intermediate-value', 'children')])
def update_file_table(data):
    if data:
        df = pd.read_json(data, orient='split')
        return html.Details([
            html.Summary('Label of the item'),
            html.Div([
            dash_table.DataTable(
                id='data-table',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns]
            ),

            html.Hr()  # horizontal line
        ])])

@app.callback(
    Output('opt-dropdown', 'options'),
    [Input('data-table', 'columns')])
def update_output_dropdown(columns):
    if columns:
        return [{'label': i['name'], 'value': i['id']} for i in columns]

@app.callback(Output('formula-table-div', 'children'),
              [Input('intermediate-value', 'children'), Input('opt-dropdown', 'value')])
def update_formula_table(data, dvar):
    if data:
        df = pd.read_json(data, orient='split')
        if dvar in df.columns:
            df_f = emg.main(df, dvar)
            return html.Div([
                dash_table.DataTable(
                    data=df_f[1].to_dict('records'),
                    row_selectable='single',
                    sort_action='native',
                    row_deletable=True,
                    columns=[{'name': i, 'id': i} for i in df_f[1].columns]
                ),

                html.Hr()  # horizontal line
            ])

#@app.callback(
#    Output('display-selected-values', 'children'),
#    [Input('opt-dropdown', 'value')])
#def update_output_dvar(value):
#    if value:
#        return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)