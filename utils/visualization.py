"""
Visualization utilities for the SHAP-LIME Hybrid Explainer.
Creates interactive Plotly charts for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'actual': '#1f77b4',
    'predicted': '#ff7f0e',
    'confidence': 'rgba(255, 127, 14, 0.2)',
    'shap': '#2ca02c',
    'lime': '#d62728',
    'hybrid': '#9467bd',
    'positive': '#2ca02c',
    'negative': '#d62728'
}


def create_forecast_plot(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    confidence_lower: Optional[np.ndarray] = None,
    confidence_upper: Optional[np.ndarray] = None,
    train_end_idx: Optional[int] = None,
    title: str = "Time Series Forecast",
    height: int = 400
) -> go.Figure:
    """
    Create interactive forecast visualization with confidence intervals.
    
    Args:
        dates: Timestamps for x-axis
        actual: Actual values
        predicted: Predicted values  
        confidence_lower: Lower confidence bound
        confidence_upper: Upper confidence bound
        train_end_idx: Index where training data ends
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Confidence interval (add first so it's behind other traces)
    if confidence_lower is not None and confidence_upper is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=confidence_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=confidence_lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=COLORS['confidence'],
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color=COLORS['actual'], width=2),
        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color=COLORS['predicted'], width=2, dash='dash'),
        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Add vertical line for train/test split
    if train_end_idx is not None and train_end_idx < len(dates):
        fig.add_vline(
            x=dates[train_end_idx],
            line_dash="dash",
            line_color="gray",
            annotation_text="Train/Test Split",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_feature_importance_heatmap(
    importance_matrix: np.ndarray,
    feature_names: List[str],
    timestamps: Optional[pd.DatetimeIndex] = None,
    title: str = "Feature Importance Over Time",
    height: int = 500,
    top_k: int = 15
) -> go.Figure:
    """
    Create time-aware feature importance heatmap.
    
    This is the CORE VISUALIZATION showing how feature importance evolves.
    
    Args:
        importance_matrix: (n_timesteps, n_features) importance scores
        feature_names: Names of features
        timestamps: Timestamps for x-axis
        title: Plot title
        height: Plot height
        top_k: Show only top-k most important features
        
    Returns:
        Plotly figure
    """
    # Select top-k features by average absolute importance
    avg_importance = np.mean(np.abs(importance_matrix), axis=0)
    top_indices = np.argsort(avg_importance)[-top_k:][::-1]
    
    selected_matrix = importance_matrix[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]
    
    # Create x-axis labels
    if timestamps is not None:
        x_labels = [ts.strftime('%Y-%m-%d') for ts in timestamps]
    else:
        x_labels = [f't-{i}' for i in range(len(importance_matrix)-1, -1, -1)]
    
    # Normalize for better visualization
    vmax = np.percentile(np.abs(selected_matrix), 95)
    
    fig = go.Figure(data=go.Heatmap(
        z=selected_matrix.T,
        x=x_labels,
        y=selected_names,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-vmax,
        zmax=vmax,
        colorbar=dict(
            title=dict(text='Importance', side='right')
        ),
        hovertemplate='Time: %{x}<br>Feature: %{y}<br>Importance: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Feature",
        height=height,
        xaxis=dict(tickangle=-45)
    )
    
    return fig


def create_shap_lime_comparison(
    shap_scores: np.ndarray,
    lime_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    title: str = "SHAP vs LIME vs Hybrid Comparison",
    top_k: int = 10
) -> go.Figure:
    """
    Create side-by-side comparison of SHAP, LIME, and Hybrid explanations.
    
    Args:
        shap_scores: SHAP importance scores
        lime_scores: LIME importance scores
        hybrid_scores: Hybrid importance scores
        feature_names: Names of features
        feature_values: Actual feature values
        title: Plot title
        top_k: Show top-k features
        
    Returns:
        Plotly figure with three panels
    """
    # Select top features by hybrid importance
    top_indices = np.argsort(np.abs(hybrid_scores))[-top_k:][::-1]
    
    selected_features = [feature_names[i] for i in top_indices]
    selected_shap = shap_scores[top_indices]
    selected_lime = lime_scores[top_indices]
    selected_hybrid = hybrid_scores[top_indices]
    selected_values = feature_values[top_indices]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('SHAP', 'Hybrid (Ours)', 'LIME'),
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    # SHAP waterfall-style bars
    colors_shap = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in selected_shap]
    fig.add_trace(go.Bar(
        y=selected_features,
        x=selected_shap,
        orientation='h',
        marker_color=colors_shap,
        name='SHAP',
        hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>'
    ), row=1, col=1)
    
    # Hybrid bars (center - highlight as the novel contribution)
    colors_hybrid = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in selected_hybrid]
    fig.add_trace(go.Bar(
        y=selected_features,
        x=selected_hybrid,
        orientation='h',
        marker_color=colors_hybrid,
        name='Hybrid',
        hovertemplate='%{y}<br>Hybrid: %{x:.4f}<extra></extra>'
    ), row=1, col=2)
    
    # LIME bars
    colors_lime = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in selected_lime]
    fig.add_trace(go.Bar(
        y=selected_features,
        x=selected_lime,
        orientation='h',
        marker_color=colors_lime,
        name='LIME',
        hovertemplate='%{y}<br>LIME: %{x:.4f}<extra></extra>'
    ), row=1, col=3)
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    
    return fig


def create_temporal_coherence_plot(
    shap_coherence: np.ndarray,
    lime_coherence: np.ndarray,
    hybrid_coherence: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    title: str = "Temporal Coherence Comparison",
    height: int = 300
) -> go.Figure:
    """
    Create plot showing temporal coherence of different methods.
    
    This demonstrates that our hybrid method maintains better
    temporal consistency than vanilla SHAP/LIME.
    
    Args:
        shap_coherence: Coherence scores for SHAP over time
        lime_coherence: Coherence scores for LIME over time
        hybrid_coherence: Coherence scores for Hybrid over time
        timestamps: Timestamps for x-axis
        title: Plot title
        height: Plot height
        
    Returns:
        Plotly figure
    """
    if timestamps is None:
        x_values = list(range(len(shap_coherence)))
    else:
        x_values = timestamps
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=shap_coherence,
        mode='lines',
        name='SHAP',
        line=dict(color=COLORS['shap'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=lime_coherence,
        mode='lines',
        name='LIME',
        line=dict(color=COLORS['lime'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=hybrid_coherence,
        mode='lines',
        name='Hybrid (Ours)',
        line=dict(color=COLORS['hybrid'], width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time" if timestamps is not None else "Timestep",
        yaxis_title="Coherence Score",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_counterfactual_plot(
    original_pred: float,
    modified_pred: float,
    feature_name: str,
    original_value: float,
    modified_value: float,
    title: str = "Counterfactual Analysis"
) -> go.Figure:
    """
    Create visualization for counterfactual analysis.
    
    Shows how prediction changes when a feature is modified.
    """
    fig = go.Figure()
    
    # Bar chart showing original vs modified prediction
    fig.add_trace(go.Bar(
        x=['Original', 'Modified'],
        y=[original_pred, modified_pred],
        marker_color=[COLORS['actual'], COLORS['predicted']],
        text=[f'{original_pred:.2f}', f'{modified_pred:.2f}'],
        textposition='outside'
    ))
    
    change = modified_pred - original_pred
    change_pct = (change / original_pred) * 100 if original_pred != 0 else 0
    
    fig.update_layout(
        title=f"{title}<br><sub>{feature_name}: {original_value:.2f} → {modified_value:.2f} | Change: {change:+.2f} ({change_pct:+.1f}%)</sub>",
        yaxis_title="Prediction",
        height=300,
        showlegend=False
    )
    
    return fig


def create_metrics_comparison_chart(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Method Comparison Metrics"
) -> go.Figure:
    """
    Create bar chart comparing metrics across methods.
    
    Args:
        metrics: Dict of {method_name: {metric_name: value}}
        title: Plot title
        
    Returns:
        Plotly figure
    """
    methods = list(metrics.keys())
    metric_names = list(metrics[methods[0]].keys())
    
    fig = go.Figure()
    
    colors = [COLORS['shap'], COLORS['hybrid'], COLORS['lime']]
    
    for i, method in enumerate(methods):
        values = [metrics[method][m] for m in metric_names]
        fig.add_trace(go.Bar(
            name=method,
            x=metric_names,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f'{v:.3f}' for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode='group',
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_feature_waterfall(
    feature_names: List[str],
    importance_scores: np.ndarray,
    base_value: float,
    prediction: float,
    title: str = "Feature Contribution Waterfall",
    top_k: int = 10
) -> go.Figure:
    """
    Create waterfall chart showing feature contributions.
    """
    # Select top features
    top_indices = np.argsort(np.abs(importance_scores))[-top_k:][::-1]
    
    selected_names = [feature_names[i] for i in top_indices]
    selected_scores = importance_scores[top_indices]
    
    # Build waterfall data
    y_values = ['Base Value'] + selected_names + ['Prediction']
    measure = ['absolute'] + ['relative'] * len(selected_names) + ['total']
    values = [base_value] + list(selected_scores) + [prediction]
    
    colors = ['gray'] + [COLORS['positive'] if s >= 0 else COLORS['negative'] for s in selected_scores] + ['blue']
    
    fig = go.Figure(go.Waterfall(
        orientation='v',
        x=y_values,
        y=values,
        measure=measure,
        connector=dict(line=dict(color='gray', width=1)),
        decreasing=dict(marker=dict(color=COLORS['negative'])),
        increasing=dict(marker=dict(color=COLORS['positive'])),
        totals=dict(marker=dict(color='blue'))
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def create_agreement_scatter(
    shap_scores: np.ndarray,
    lime_scores: np.ndarray,
    feature_names: List[str],
    title: str = "SHAP vs LIME Agreement"
) -> go.Figure:
    """
    Create scatter plot showing SHAP vs LIME agreement.
    
    Points near the diagonal indicate agreement.
    """
    # Normalize for comparison
    shap_norm = shap_scores / (np.abs(shap_scores).max() + 1e-8)
    lime_norm = lime_scores / (np.abs(lime_scores).max() + 1e-8)
    
    fig = go.Figure()
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[-1, 1],
        y=[-1, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Agreement',
        showlegend=True
    ))
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=shap_norm,
        y=lime_norm,
        mode='markers+text',
        marker=dict(size=10, color=COLORS['hybrid']),
        text=feature_names,
        textposition='top center',
        name='Features',
        hovertemplate='%{text}<br>SHAP: %{x:.3f}<br>LIME: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="SHAP (normalized)",
        yaxis_title="LIME (normalized)",
        height=400,
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1])
    )
    
    return fig
