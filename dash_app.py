from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global configurations
DATA_DIR = "static"
SYNTHETIC_DATA_FILE = os.path.join(DATA_DIR, "synthetic_data.csv")

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H1("Drone Simulation Dashboard", className="text-center"), width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Controls"),
                            dbc.CardBody(
                                [
                                    html.Div("Number of Samples:"),
                                    dcc.Input(
                                        id="num-samples",
                                        type="number",
                                        value=100,
                                        placeholder="Enter number of samples",
                                        className="mb-3",
                                    ),
                                    dbc.Button("Generate Data", id="generate-btn", color="primary", className="mb-3"),
                                    html.Div(id="generate-status"),
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Statistics"),
                            dbc.CardBody(
                                [
                                    html.Div(id="stats-display", children="No statistics available."),
                                    dbc.Button("Refresh Stats", id="refresh-stats-btn", color="info", className="mt-3"),
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Filters"),
                            dbc.CardBody(
                                [
                                    html.Div("Legality Score Threshold:"),
                                    dcc.RangeSlider(
                                        id="legality-threshold",
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        marks={i / 10: f"{i / 10}" for i in range(0, 11)},
                                        value=[0.2, 0.8],
                                    ),
                                    html.Div(id="filter-status", className="mt-3"),
                                ]
                            ),
                        ]
                    ),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="legality-chart"), width=12),
            ]
        ),
    ],
    fluid=True,
)

# Callback to generate data
@app.callback(
    Output("generate-status", "children"),
    Input("generate-btn", "n_clicks"),
    State("num-samples", "value"),
)
def generate_data(n_clicks, num_samples):
    if n_clicks is None:
        return "Click 'Generate Data' to start."
    try:
        # Corrected data generation using numpy
        synthetic_data = pd.DataFrame(
            {
                "roll": np.random.randn(num_samples),
                "pitch": np.random.randn(num_samples),
                "yaw": np.random.randn(num_samples),
                "throttle": np.random.randn(num_samples),
                "legality_score": np.random.rand(num_samples),
            }
        )
        synthetic_data.to_csv(SYNTHETIC_DATA_FILE, index=False)
        return f"Data generated successfully with {num_samples} samples!"
    except Exception as e:
        return f"Error generating data: {e}"

# Callback to refresh stats
@app.callback(
    Output("stats-display", "children"),
    Input("refresh-stats-btn", "n_clicks"),
)
def refresh_stats(n_clicks):
    if n_clicks is None:
        return "No statistics available."
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            return "Synthetic data file not found. Please generate data first."
        
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        
        if "legality_score" not in df.columns:
            return "Error: 'legality_score' column not found in the data. Please regenerate data."

        stats = df["legality_score"].describe().to_dict()
        stats_display = [html.Div(f"{key}: {value:.2f}") for key, value in stats.items()]
        return stats_display
    except Exception as e:
        return f"Error fetching statistics: {e}"

# Callback to update chart based on legality score filter
@app.callback(
    Output("legality-chart", "figure"),
    Input("legality-threshold", "value"),
)
def update_chart(threshold):
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            return {}
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        filtered_df = df[
            (df["legality_score"] >= threshold[0]) & (df["legality_score"] <= threshold[1])
        ]
        fig = px.scatter(
            filtered_df,
            x="roll",
            y="pitch",
            color="legality_score",
            title="Drone Flight Data (Filtered by Legality Score)",
        )
        return fig
    except Exception as e:
        return {}

if __name__ == "__main__":
    app.run_server(debug=True)
