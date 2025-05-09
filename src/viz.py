import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
pio.templates.default = "plotly_white"

import plotly.express as px

def plot_global_scores(scores_df, list_metrics=None):
    
    if list_metrics is None:
        list_metrics = [c for c in scores_df.columns if c not in ["count", "trial"]]
    
    scores_df = scores_df.copy().drop('count', axis=1)
    scores_df = (
        scores_df[["trial"] + list_metrics]
        .melt(id_vars="trial", value_name="metric_value", var_name="metric_name")
    )
    
    fig = px.bar(
        scores_df,
        x="metric_name",
        y="metric_value",
        color="trial",
        barmode="group"  # Cette ligne ne fait rien ici, déplacée plus bas
    )
    
    fig.update_layout(
        title="Global Forecast Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        template="plotly_white",
        width=600,
        height=400,
        barmode="group"  # <-- Ajout ici pour barres côte à côte
    )
    
    return fig

def plot_scores_per_ts(scores_per_ts_df, list_metrics=None):
    if list_metrics is None:
        list_metrics = [c for c in scores_per_ts_df.columns if c not in ["count", "trial", "id"]]
    
    melted = scores_per_ts_df.melt(
        id_vars=["id", "trial"], 
        var_name="metric", 
        value_name="value"
    )
    
    melted = melted[melted["metric"].isin(list_metrics)]
    
    fig = px.box(
        melted,
        x="metric",
        y="value",
        color="trial",
        points="all",
        title="Distribution of Metrics per Time Series by Trial",
        template="plotly_white",
        width=800,
        height=500,
        hover_data="id"
    )
    return fig


def plot_forecasts_with_train(train_df, test_df, forecast_df, scores_per_ts_df=None, metric='MAE', ids_to_plot=None, train_tail=30):
    """
    Plot the forecast, test, and last steps of train for multiple time series IDs,
    with train → test → forecast lines visually connected. Includes a selected metric in the plot title.

    Parameters:
    - train_df: DataFrame containing training data with 'id', 'date', 'sales'
    - test_df: DataFrame containing test data with 'id', 'date', 'sales'
    - forecast_df: DataFrame containing forecasts with 'id', 'date', 'TimeGPT'
    - scores_per_ts_df: DataFrame with scores per 'id', should contain columns like 'MAE', 'RMSE', etc.
    - metric: str, the metric to include in the plot title (e.g. 'MAE', 'RMSE', 'MAPE', 'R2', 'count')
    - ids_to_plot: list of IDs to plot (default = all unique IDs from test_df)
    - train_tail: number of last train steps to display
    """
    if ids_to_plot is None:
        ids_to_plot = test_df['id'].unique()

    for uid in ids_to_plot:
        train_sub = train_df[train_df['id'] == uid].sort_values("date").copy()
        test_sub = test_df[test_df['id'] == uid].sort_values("date").copy()
        forecast_sub = forecast_df[forecast_df['id'] == uid].sort_values("date").copy()

        train_tail_df = train_sub.tail(train_tail)

        if not train_tail_df.empty and not test_sub.empty:
            last_train_point = train_tail_df.iloc[[-1]]
            test_sub = pd.concat([last_train_point, test_sub])

        if not test_sub.empty and not forecast_sub.empty:
            first_test_point = test_sub.iloc[[0]].copy()
            first_test_point["TimeGPT"] = None
            forecast_sub = pd.concat([first_test_point[["date", "TimeGPT"]], forecast_sub])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train_tail_df['date'], y=train_tail_df['sales'],
            line=dict(color="lightgray"),
            name="Train Sales"
        ))

        fig.add_trace(go.Scatter(
            x=test_sub['date'], y=test_sub['sales'],
            line=dict(color="coral"),
            name="Test Sales"
        ))

        fig.add_trace(go.Scatter(
            x=forecast_sub['date'], y=forecast_sub['TimeGPT'],
            line=dict(color="coral", dash="dash"),
            name="Forecast"
        ))

        # Compose title
        if scores_per_ts_df is not None and uid in scores_per_ts_df['id'].values:
            row = scores_per_ts_df[scores_per_ts_df['id'] == uid].iloc[0]
            metric_value = row.get(metric, None)
            if pd.notnull(metric_value):
                title = f"ID: {uid} | {metric}: {metric_value:.2f}"
            else:
                title = f"ID: {uid} | {metric}: N/A"
        else:
            title = f"Forecast with Train/Test Split for ID: {uid}"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white",
            width=800,
            height=400
        )

        fig.show()


def plot_forecast_with_ci(train_df, test_df, forecast_df, uid, level=9, train_tail=30):
    """
    Plot the forecast with confidence intervals, actual test data, and training data for a given time series ID.

    Parameters:
    - train_df: DataFrame with training data.
    - test_df: DataFrame with ground truth test data.
    - forecast_df: DataFrame returned by Nixtla's forecast, including quantile columns.
    - uid: The time series ID to plot.
    - level: Confidence interval level (e.g., 90 for 90% interval).
    """
    # Extract the quantile boundaries based on the level
    lower_q = f"TimeGPT-lo-{level}"
    upper_q = f"TimeGPT-hi-{level}"

    # Subset the data for the specific ID
    fc = forecast_df[forecast_df["id"] == uid]
    test = test_df[test_df["id"] == uid]
    train = train_df[train_df["id"] == uid]

    # Keep only the last `train_tail` rows
    train = train.tail(train_tail)
    
    # Ensure visual continuity: add last train point to test (only for plotting)
    if not train.empty and not test.empty:
        last_train_point = train.iloc[[-1]]
        test = pd.concat([last_train_point, test])
   
        first_test_point = train.iloc[[-1]].copy()
        first_test_point["TimeGPT"] = last_train_point["sales"] 
        first_test_point[lower_q] = last_train_point["sales"]
        first_test_point[upper_q] = last_train_point["sales"]
        fc = pd.concat([first_test_point[["date", "TimeGPT", lower_q, upper_q]], fc])

    fig = go.Figure()

    # Plot recent train data
    fig.add_trace(go.Scatter(
        x=train["date"], y=train["sales"],
        mode="lines",
        name="Train Sales (last steps)",
        line=dict(color="lightgray")
    ))

    # Plot test (actual) data
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["sales"],
        mode="lines",
        name="Test Sales",
        line=dict(color="coral")
    ))

    # Plot lower bound of confidence interval
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc[lower_q],
        mode="lines",
        line=dict(width=0),
        name=f"{lower_q}",
        showlegend=False
    ))

    # Plot upper bound and fill area between quantiles
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc[upper_q],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 127, 80, 0.3)",  # translucent orange
        name=f"{level}% Confidence Interval"
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["TimeGPT"],
        mode="lines",
        name="Forecast",
        line=dict(color="coral", dash="dash")
    ))

    fig.update_layout(
        title=f"Forecast with {level}% CI for ID: {uid}",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        width=1000,
        height=500
    )

    return fig
