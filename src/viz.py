import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
pio.templates.default = "plotly_white"

import plotly.express as px

def plot_global_scores(scores_df, list_metrics=None):
    
    if list_metrics is None:
        list_metrics = [c for c in scores_df.columns if c not in ["count", "model"]]
    
    scores_df = scores_df.copy().drop('count', axis=1)
    scores_df = (
        scores_df[["model"] + list_metrics]
        .melt(id_vars="model", value_name="metric_value", var_name="metric_name")
    )
    
    fig = px.bar(
        scores_df,
        x="metric_name",
        y="metric_value",
        color="model",
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

def plot_scores_per_ts(scores_per_ts_df, column_id, list_metrics=None):
    if list_metrics is None:
        list_metrics = [c for c in scores_per_ts_df.columns if c not in ["count", "model", column_id]]
    
    melted = scores_per_ts_df.melt(
        id_vars=[column_id, "model"], 
        var_name="metric", 
        value_name="value"
    )
    
    melted = melted[melted["metric"].isin(list_metrics)]
    
    fig = px.box(
        melted,
        x="metric",
        y="value",
        color="model",
        points="all",
        title="Distribution of metrics per time series by model",
        template="plotly_white",
        width=800,
        height=500,
        hover_data=column_id
    )
    return fig

def plot_forecasts_uid(
    uid,
    forecasts_df, 
    column_id,
    column_date,
    column_target,
    map_columns_forecasts, 
    scores_per_ts_df,
    metric,
    train_tail,
    map_color_forecasts=None,
    as_percentage=False
    ):
    
    forecast_uid_df = forecasts_df[forecasts_df[column_id] == uid]
    
    for i, (column_name, column_value) in enumerate(map_columns_forecasts.items()):
        
        if i == 0:
            train_sub = forecast_uid_df[forecast_uid_df[column_value].isna()].sort_values(column_date).copy()
            test_sub = forecast_uid_df[~forecast_uid_df[column_value].isna()].sort_values(column_date).copy()
            train_tail_df = train_sub.tail(train_tail)

            if not train_tail_df.empty and not test_sub.empty:
                last_train_point = train_tail_df.iloc[[-1]]
                test_sub = pd.concat([last_train_point, test_sub])

            fig = go.Figure()

            # Plot train target
            fig.add_trace(go.Scatter(
                x=train_tail_df[column_date], y=train_tail_df[column_target],
                line=dict(color="lightgray"),
                name=f"Train {column_target}"
            ))

            # Plot test target
            fig.add_trace(go.Scatter(
                x=test_sub[column_date], y=test_sub[column_target],
                line=dict(color="royalblue"),
                name=f"Test {column_target}"
            ))

        forecast_sub = forecast_uid_df[~forecast_uid_df[column_value].isna()].sort_values(column_date).copy()

        if not test_sub.empty and not forecast_sub.empty:
            first_test_point = test_sub.iloc[[0]].copy()
            first_test_point[column_value] = None
            forecast_sub = pd.concat([first_test_point[[column_date, column_value]], forecast_sub])
        
        metric_value = None
        if scores_per_ts_df is not None:
            row_match = scores_per_ts_df[
                (scores_per_ts_df[column_id] == uid) &
                (scores_per_ts_df["model"] == column_value)
            ]
            if not row_match.empty and metric in row_match.columns:
                metric_value = row_match.iloc[0][metric]

        # Format legend label
        if metric_value is not None:
            display_value = metric_value * 100 if as_percentage else metric_value
            suffix = "%" if as_percentage else ""
            legend_label = f"{column_name} ({metric}: {display_value:.2f}{suffix})"
        else:
            legend_label = column_name

        # Assign color if provided
        color = map_color_forecasts.get(column_value, None) if map_color_forecasts else None
        
        fig.add_trace(go.Scatter(
            x=forecast_sub[column_date],
            y=forecast_sub[column_value],
            line=dict(dash="dash", color=color),
            name=legend_label
        ))

    fig.update_layout(
        title=f"Forecast for ID: {uid}",
        xaxis_title=column_date,
        yaxis_title=column_target,
        template="plotly_white",
        width=1000,
        height=500
    )

    return fig

def plot_forecast_with_ci(
    forecast_df, 
    column_id,
    column_date,
    column_target,
    column_forecast,
    uid, 
    model_name,
    level=90, 
    train_tail=30
    ):
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
    lower_q = f"{model_name}-lo-{level}"
    upper_q = f"{model_name}-hi-{level}"

    # Subset the data for the specific ID
    forecast_uid_df = forecast_df[forecast_df[column_id] == uid]
    fc = forecast_uid_df[forecast_df[lower_q].notna()]
    test = forecast_uid_df[forecast_df[lower_q].notna()]
    train = forecast_uid_df[forecast_df[lower_q].isna()]

    # Keep only the last `train_tail` rows
    train = train.tail(train_tail)
    
    # Ensure visual continuity: add last train point to test (only for plotting)
    if not train.empty and not test.empty:
        last_train_point = train.iloc[[-1]]
        test = pd.concat([last_train_point, test])
   
        first_test_point = train.iloc[[-1]].copy()
        first_test_point[column_forecast] = last_train_point[column_target] 
        first_test_point[lower_q] = last_train_point[column_target]
        first_test_point[upper_q] = last_train_point[column_target]
        fc = pd.concat([first_test_point[[column_date, column_forecast, lower_q, upper_q]], fc])

    fig = go.Figure()

    # Plot recent train data
    fig.add_trace(go.Scatter(
        x=train[column_date], y=train[column_target],
        mode="lines",
        name="Train Sales (last steps)",
        line=dict(color="lightgray")
    ))

    # Plot test (actual) data
    fig.add_trace(go.Scatter(
        x=test[column_date], y=test[column_target],
        mode="lines",
        name="Test Sales",
        line=dict(color="coral")
    ))

    # Plot lower bound of confidence interval
    fig.add_trace(go.Scatter(
        x=fc[column_date], y=fc[lower_q],
        mode="lines",
        line=dict(width=0),
        name=f"{lower_q}",
        showlegend=False
    ))

    # Plot upper bound and fill area between quantiles
    fig.add_trace(go.Scatter(
        x=fc[column_date], y=fc[upper_q],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 127, 80, 0.3)",  # translucent orange
        name=f"{level}% Confidence Interval"
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=fc[column_date], y=fc[column_forecast],
        mode="lines",
        name="Forecast",
        line=dict(color="coral", dash="dash")
    ))

    fig.update_layout(
        title=f"Forecast with {level}% CI for ID: {uid}",
        xaxis_title=column_date,
        yaxis_title=column_target,
        template="plotly_white",
        width=1000,
        height=500
    )

    return fig
