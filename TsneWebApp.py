# -*- coding: utf-8 -*-
# author: Ryohei Yamaguchi

import os

import numpy as np
import pandas as pd
import base64
import io

from sklearn.manifold import TSNE

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

from sklearn import datasets

"""global variable"""
# train = pd.DataFrame()
train = pd.DataFrame(datasets.load_wine()["data"])

col_options = [dict(label=x, value=x) for x in train.columns]

test = None
n_iter = 1000
n_components = 3

"""app"""
app = dash.Dash(__name__)

# mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
# app.scripts.append_script({'external_url': mathjax})

app.layout = html.Div(
    [
        html.H1("t-SNE Visualize"),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("train data"),
                        dcc.Upload(
                            id="update_train",
                            children=html.Div([
                                "Drug and Drop or ",
                                html.A("Select File")
                            ]),
                            style={
                                "width": "95%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "textAlign": "center",
                                "borderRadius": "5px"
                            },
                            multiple=False
                        ),
                    ],
                ),  # train
                html.Div(
                    [
                        html.H2("test data"),
                        dcc.Upload(
                            id="update_test",
                            children=html.Div([
                                "Drug and Drop or ",
                                html.A("Select File")
                            ]),
                            style={
                                "width": "95%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "textAlign": "center",
                                "borderRadius": "5px"
                            },
                            multiple=False
                        ),

                    ]
                ),  # test
                html.Div(
                    [
                         html.H2("initial column"),
                         dcc.Dropdown(id="column", options=col_options)
                    ]
                ),  # initial column
                html.Div(
                    [
                        html.H2("perplexity"),
                        dcc.Input(
                            id="perplexity",
                            type="number",
                            min=10,
                            max=200,
                            step=10
                        )
                    ]
                ),  # perplexity Input ==> Slider
                html.Div(
                    [
                        html.H2("learning rate"),
                        dcc.Input(
                            id="learning_rate",
                            type="number",
                            min=1,
                            max=40
                        )
                    ]
                ),  # learning rate
                html.Div(
                    [
                        html.H2("calculation"),
                        html.Button(
                            "Run",
                            id="run_button",
                            n_clicks=0
                        )
                    ]
                ),  # run
                html.Div(
                    [
                        html.H2("color"),
                        dcc.Dropdown(id="color", options=col_options)
                    ]
                )  # color
            ],
            style={"width": "25%", "float": "left"}
        ),
        dcc.Graph(id="graph", style={"width": "70%", "display": "inline-block"})
    ]
)


@app.callback(
    Output("graph", "figure"),
    [
        Input("train data", "contents"),
        Input("column", "value"),
        Input("run_button", "n_clicks"),
        Input("perplexity", "value"),
        Input("learning_rate", "value"),
        Input("color", "value")
    ]
)
def calc_tsne(train_contents, column, n_clicks, perplexity, learning_rate, color):
    global train
    if train_contents is not None:
        train = contents2dataframe(train_contents)
    if perplexity is None:
        perplexity = 30
    if learning_rate is None:
        learning_rate = 10
    if n_clicks > 0:
        train_reduced = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter
        ).fit_transform(train.loc[:, column:])
        train_reduced = pd.DataFrame(train_reduced, columns=[f"components {i+1}" for i in range(n_components)])
        train_reduced = pd.concat([train.loc[:, column:], train_reduced], axis=1)
        x, y, z = "components 1", "components 2", "components 3"

    else:
        train_reduced = train.copy()
        col = train_reduced.columns
        x, y, z = col[0], col[1], col[2]

    return px.scatter_3d(
        train_reduced,
        x=x,
        y=y,
        z=z,
        color=color
    )


def contents2dataframe(contents):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df


app.run_server(port=8045, debug=True)
