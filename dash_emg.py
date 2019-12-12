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
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import plotly.graph_objects as go
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
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
            id='opt-dropdown'
            ),
            ],style={'width': '20%', 'display': 'inline-block'}
        ),
    html.Div(id='display-selected-values'),
    html.Div(id='output-data-upload'),
    html.Div(id='formula-table-div',children=[
                        dash_table.DataTable(
                    id='formula-table',
                    row_selectable='single',
                    sort_action='native',
                    row_deletable=True
                ),

                html.Hr()  # horizontal line
    ]),
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
            html.Summary('Data'),
            html.Div([
            dash_table.DataTable(
                id='data-table',
                data=df.to_dict('records'),
                style_table={
                    'maxHeight': '300px',
                    'overflowY': 'scroll'
                },
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
                    id='formula-table',
                    data=df_f[1].to_dict('records'),
                    row_selectable='single',
                    sort_action='native',
                    selected_rows=[],
                    style_table={
                        'maxHeight': '300px',
                        'overflowY': 'scroll'
                    },
                    row_deletable=True,
                    columns=[{'name': i, 'id': i} for i in df_f[1].columns],
                    style_data_conditional=[
                        {
                        "if": {'filter_query': '({each_variable_important_result} > 0) && ({het_white_result} = 0) && ({het_breuschpagan_result} = 0) && ({reset_ramsey_result} = 0) && ({linear_harvey_collier_result} = 0) && ({vif_result} = 0)'},
                        "backgroundColor": "#3D9970",
                        'color': 'white'
                        },
                        {
                        "if": {'filter_query': '({each_variable_important_result} = 0) || ({reset_ramsey_result} > 0)'},
                        "backgroundColor": "#B33A3A",
                        'color': 'white'
                        }
                    ]
                ),

                html.Hr(),  # horizontal line
                html.Div(id='graph')
            ])

#@app.callback(
#    Output('display-selected-values', 'children'),
#    [Input('opt-dropdown', 'value')])
#def update_output_dvar(value):
#    if value:
#        return 'You have selected "{}"'.format(value)

@app.callback(
    Output('graph','children'),
    [Input('intermediate-value', 'children'),
     Input('opt-dropdown', 'value'),
     Input('formula-table', "derived_virtual_data"),
     Input('formula-table', "derived_virtual_selected_rows")])
def update_figure(data,dependent_variable,rows,selected_row_indices):
    if data:
        df = pd.read_json(data, orient='split')
        if dependent_variable in df.columns and selected_row_indices:
            selected_row=rows[selected_row_indices[0]]
            if selected_row:
                print(selected_row['Formula'])
                mod = sm.OLS.from_formula(selected_row['Formula'], data=df).fit()
                dependent_variable_str = dependent_variable
                dependent_variable = df[dependent_variable]
                prstd, iv_l, iv_u = wls_prediction_std(mod)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                # Add traces
                fig.add_trace(
                    go.Scatter(y=dependent_variable, name="data"),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(y=mod.fittedvalues, name="OLS",
                    mode='lines+markers',
                    line={'dash': 'solid', 'color': 'red'}),
                    secondary_y=True
                )
                fig.add_trace(
                    go.Scatter(y=iv_u, name="Upper confidence bound",
                    line={'dash': 'dash', 'color': 'red'}),
                    secondary_y=True
                )
                fig.add_trace(
                    go.Scatter(y=iv_l, name="Lower confidence bound",
                    line={'dash': 'dash', 'color': 'red'}),
                    secondary_y=True,
                )
                fig.update_layout(
                    title_text='Obs. sorted by ' + dependent_variable_str
                )
                print(fig)
                return html.Div([dcc.Graph(
                    id='prediction',
                    figure=fig
                )])

if __name__ == '__main__':
    app.run_server(debug=True)