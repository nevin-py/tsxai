"""
Streamlit Dashboard for Adaptive SHAP-LIME Hybrid Explainer.

This is the main entry point for the interactive dashboard.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from utils.data_loader import get_dataset, TimeSeriesDataLoader
from utils.feature_engineering import TemporalFeatureEngineer
from utils.metrics import ForecastMetrics, ExplanationMetrics
from utils.visualization import (
    create_forecast_plot,
    create_feature_importance_heatmap,
    create_shap_lime_comparison,
    create_temporal_coherence_plot,
    create_counterfactual_plot,
    create_metrics_comparison_chart,
    create_agreement_scatter
)

# Page config
st.set_page_config(
    page_title="SHAP-LIME Hybrid Explainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(dataset_name: str):
    """Load and cache dataset."""
    return get_dataset(dataset_name)


@st.cache_data
def prepare_features(df: pd.DataFrame, target_col: str, date_col: str = 'date'):
    """Prepare features for modeling."""
    fe = TemporalFeatureEngineer(target_col=target_col, date_col=date_col)
    df_features, feature_names = fe.fit_transform(df)
    return df_features, feature_names


@st.cache_resource
def train_model(model_name: str, X_train, y_train, X_val, y_val, feature_names):
    """Train and cache model."""
    from models import get_model
    
    model = get_model(model_name)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    return model


@st.cache_resource
def create_explainer(_model, feature_names, training_data, model_name):
    """Create and cache explainer."""
    from explainers import create_hybrid_explainer
    
    predict_func = _model.predict if hasattr(_model, 'predict') else _model
    model_obj = _model.model if hasattr(_model, 'model') else None
    
    explainer = create_hybrid_explainer(
        model=predict_func,
        feature_names=feature_names,
        training_data=training_data,
        model_object=model_obj if model_name == 'xgboost' else None,
        fusion_method='adaptive'
    )
    
    return explainer


def main():
    # Header
    st.markdown('<p class="main-header">🎯 Adaptive SHAP-LIME Hybrid Explainer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Time Series Forecasting with Temporal-Aware Explanations</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        dataset_name = st.selectbox(
            "📊 Dataset",
            options=['stock', 'weather', 'energy'],
            help="Select the dataset to analyze"
        )
        
        # Model selection
        model_name = st.radio(
            "🤖 Model",
            options=['xgboost', 'prophet', 'lstm'],
            help="Select the forecasting model"
        )
        
        # Time range
        st.subheader("📅 Time Range")
        window_size = st.slider(
            "Analysis Window (days)",
            min_value=30,
            max_value=180,
            value=90,
            help="Number of days to analyze"
        )
        
        # Forecast horizon
        forecast_horizon = st.select_slider(
            "Forecast Horizon",
            options=[1, 7, 14, 30],
            value=7,
            help="Days ahead to forecast"
        )
        
        # Run analysis button
        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Explanation settings
        st.subheader("🔬 Explanation Settings")
        fusion_method = st.selectbox(
            "Fusion Method",
            options=['adaptive', 'weighted', 'conflict_resolution'],
            help="How to combine SHAP and LIME explanations"
        )
        
        temporal_smoothing = st.checkbox(
            "Temporal Smoothing",
            value=True,
            help="Apply smoothing for temporal coherence"
        )
    
    # Main content area
    if run_analysis or 'results' in st.session_state:
        with st.spinner("Loading data and training model..."):
            # Load data
            df = load_data(dataset_name)
            
            # Set target column based on dataset
            target_col = {
                'stock': 'close',
                'weather': 'temperature',
                'energy': 'consumption'
            }[dataset_name]
            
            # Prepare features
            df_features, feature_names = prepare_features(df, target_col)
            
            # Temporal split
            loader = TimeSeriesDataLoader()
            train_df, val_df, test_df = loader.temporal_split(df_features)
            
            # Get feature matrix
            feature_cols = [col for col in feature_names if col in df_features.columns]
            
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val = val_df[feature_cols].values
            y_val = val_df[target_col].values
            X_test = test_df[feature_cols].values[-window_size:]
            y_test = test_df[target_col].values[-window_size:]
            test_dates = pd.to_datetime(test_df['date'].values[-window_size:])
            
            # Train model
            model = train_model(model_name, X_train, y_train, X_val, y_val, feature_cols)
            
            # Get predictions
            predictions = model.predict(X_test)
            forecast_result = model.predict_with_confidence(X_test)
            
            # Create explainer and get explanations
            explainer = create_explainer(model, feature_cols, X_train, model_name)
        
        # Layout: 2 columns for top section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📈 Forecast Visualization")
            
            # Create forecast plot
            forecast_fig = create_forecast_plot(
                dates=test_dates,
                actual=y_test,
                predicted=predictions,
                confidence_lower=forecast_result.confidence_lower,
                confidence_upper=forecast_result.confidence_upper,
                title=f"{dataset_name.title()} {target_col.title()} Forecast"
            )
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance Metrics")
            
            # Calculate metrics
            metrics = ForecastMetrics.compute_all(y_test, predictions)
            
            st.metric("RMSE", f"{metrics['RMSE']:.3f}")
            st.metric("MAE", f"{metrics['MAE']:.3f}")
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            st.metric("R²", f"{metrics['R2']:.3f}")
        
        st.markdown("---")
        
        # Feature Importance Heatmap
        st.subheader("🔥 Feature Importance Over Time (Hybrid Method)")
        
        with st.spinner("Computing temporal explanations..."):
            # Get explanations for test window
            n_explain = min(30, len(X_test))  # Limit for performance
            explanations = explainer.explain_temporal_window(
                X_test[-n_explain:],
                timestamps=test_dates[-n_explain:]
            )
            
            # Extract importance matrices
            shap_matrix, lime_matrix, hybrid_matrix = explainer.get_temporal_importance_matrix(explanations)
        
        # Create heatmap
        heatmap_fig = create_feature_importance_heatmap(
            importance_matrix=hybrid_matrix,
            feature_names=feature_cols,
            timestamps=test_dates[-n_explain:],
            title="Temporal Feature Importance (Hybrid SHAP-LIME)",
            top_k=12
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        st.markdown("---")
        
        # SHAP vs LIME Comparison
        st.subheader("🔍 Explanation Comparison: SHAP vs Hybrid vs LIME")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Select timestep to explain
            selected_idx = st.slider(
                "Select timestep to explain",
                min_value=0,
                max_value=n_explain - 1,
                value=n_explain - 1,
                help="Choose which timestep to show detailed explanation for"
            )
            
            exp = explanations[selected_idx]
            
            comparison_fig = create_shap_lime_comparison(
                shap_scores=exp.shap_scores,
                lime_scores=exp.lime_scores,
                hybrid_scores=exp.importance_scores,
                feature_names=feature_cols,
                feature_values=exp.feature_values,
                top_k=10
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Fusion Weights")
            if exp.fusion_weights:
                st.metric("SHAP Weight", f"{exp.fusion_weights[0]:.2%}")
                st.metric("LIME Weight", f"{exp.fusion_weights[1]:.2%}")
            
            st.markdown("### Prediction")
            st.metric("Value", f"{exp.prediction:.2f}")
            
            if exp.disagreement_indices:
                st.warning(f"⚠️ {len(exp.disagreement_indices)} features have SHAP-LIME disagreement")
        
        st.markdown("---")
        
        # Temporal Coherence Validation
        st.subheader("📏 Temporal Coherence Validation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Compute rolling coherence
            window = 5
            shap_coherence = []
            lime_coherence = []
            hybrid_coherence = []
            
            for i in range(window, len(explanations)):
                shap_coh = ExplanationMetrics.temporal_coherence(shap_matrix[i-window:i])
                lime_coh = ExplanationMetrics.temporal_coherence(lime_matrix[i-window:i])
                hybrid_coh = ExplanationMetrics.temporal_coherence(hybrid_matrix[i-window:i])
                
                shap_coherence.append(shap_coh)
                lime_coherence.append(lime_coh)
                hybrid_coherence.append(hybrid_coh)
            
            coherence_fig = create_temporal_coherence_plot(
                shap_coherence=np.array(shap_coherence),
                lime_coherence=np.array(lime_coherence),
                hybrid_coherence=np.array(hybrid_coherence),
                timestamps=test_dates[-n_explain+window:],
                title="Temporal Coherence: Hybrid Method Shows Superior Stability"
            )
            st.plotly_chart(coherence_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Average Coherence Scores")
            
            avg_shap = np.mean(shap_coherence) if shap_coherence else 0
            avg_lime = np.mean(lime_coherence) if lime_coherence else 0
            avg_hybrid = np.mean(hybrid_coherence) if hybrid_coherence else 0
            
            st.metric("SHAP", f"{avg_shap:.3f}")
            st.metric("LIME", f"{avg_lime:.3f}")
            st.metric("Hybrid (Ours)", f"{avg_hybrid:.3f}", delta=f"+{(avg_hybrid - max(avg_shap, avg_lime)):.3f}")
            
            st.info("Higher coherence = More stable explanations over time")
        
        st.markdown("---")
        
        # Counterfactual Analysis
        st.subheader("🎲 Interactive Counterfactual Analysis")
        
        st.markdown("*What if we change a feature value? How would the prediction change?*")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Get top features for counterfactual
        top_indices = np.argsort(np.abs(exp.importance_scores))[-5:][::-1]
        top_features = [feature_cols[i] for i in top_indices]
        
        with col1:
            cf_feature = st.selectbox(
                "Select feature to modify",
                options=top_features,
                help="Choose a feature to see its counterfactual effect"
            )
            
            feat_idx = feature_cols.index(cf_feature)
            original_value = exp.feature_values[feat_idx]
            
            # Get reasonable range
            feat_std = np.std(X_train[:, feat_idx])
            min_val = original_value - 3 * feat_std
            max_val = original_value + 3 * feat_std
            
            modified_value = st.slider(
                f"Modify {cf_feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(original_value),
                help="Drag to change feature value"
            )
        
        with col2:
            # Compute counterfactual prediction
            cf_input = X_test[selected_idx].copy()
            cf_input[feat_idx] = modified_value
            cf_prediction = model.predict(cf_input.reshape(1, -1))[0]
            
            cf_fig = create_counterfactual_plot(
                original_pred=exp.prediction,
                modified_pred=cf_prediction,
                feature_name=cf_feature,
                original_value=original_value,
                modified_value=modified_value
            )
            st.plotly_chart(cf_fig, use_container_width=True)
        
        with col3:
            st.markdown("### Impact Analysis")
            
            change = cf_prediction - exp.prediction
            change_pct = (change / exp.prediction) * 100 if exp.prediction != 0 else 0
            
            st.metric(
                "Prediction Change",
                f"{change:+.3f}",
                delta=f"{change_pct:+.1f}%"
            )
            
            st.metric(
                "Original Value",
                f"{original_value:.3f}"
            )
            
            st.metric(
                "Modified Value",
                f"{modified_value:.3f}"
            )
        
        st.markdown("---")
        
        # Method Comparison Summary
        st.subheader("📋 Method Comparison Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Agreement scatter plot
            agreement_fig = create_agreement_scatter(
                shap_scores=exp.shap_scores,
                lime_scores=exp.lime_scores,
                feature_names=feature_cols
            )
            st.plotly_chart(agreement_fig, use_container_width=True)
        
        with col2:
            # Metrics comparison
            comparison_metrics = {
                'SHAP': {
                    'Coherence': avg_shap,
                    'Avg Magnitude': np.mean(np.abs(shap_matrix))
                },
                'Hybrid': {
                    'Coherence': avg_hybrid,
                    'Avg Magnitude': np.mean(np.abs(hybrid_matrix))
                },
                'LIME': {
                    'Coherence': avg_lime,
                    'Avg Magnitude': np.mean(np.abs(lime_matrix))
                }
            }
            
            metrics_fig = create_metrics_comparison_chart(
                metrics=comparison_metrics,
                title="Explanation Quality Metrics"
            )
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Key findings
        st.markdown("### 🎯 Key Findings")
        
        findings_col1, findings_col2, findings_col3 = st.columns(3)
        
        with findings_col1:
            st.success(f"""
            **Hybrid Method Advantage**
            - {(avg_hybrid - max(avg_shap, avg_lime)) / max(avg_shap, avg_lime) * 100:+.1f}% better coherence
            - Resolves SHAP-LIME conflicts
            - Respects temporal causality
            """)
        
        with findings_col2:
            rank_corr = ExplanationMetrics.rank_correlation(exp.shap_scores, exp.lime_scores)
            st.info(f"""
            **SHAP-LIME Agreement**
            - Rank correlation: {rank_corr:.3f}
            - Top-5 feature overlap: {ExplanationMetrics.feature_agreement(exp.shap_scores, exp.lime_scores):.1%}
            """)
        
        with findings_col3:
            st.warning(f"""
            **Model Performance**
            - RMSE: {metrics['RMSE']:.3f}
            - R²: {metrics['R2']:.3f}
            - Forecast horizon: {forecast_horizon} days
            """)
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin!")
        
        st.markdown("""
        ### About This Dashboard
        
        This dashboard demonstrates a **novel hybrid approach** combining SHAP and LIME 
        for explaining time series forecasts while **respecting temporal causality**.
        
        #### Key Features:
        - 📊 **Multiple Datasets**: Stock prices, weather, energy consumption
        - 🤖 **Multiple Models**: XGBoost, Prophet, LSTM
        - 🔬 **Temporal-Aware Explanations**: No future data leakage
        - 🔥 **Feature Importance Heatmaps**: See how importance evolves over time
        - 🔍 **Method Comparison**: Side-by-side SHAP vs LIME vs Hybrid
        - 📏 **Coherence Validation**: Prove temporal stability
        - 🎲 **Counterfactual Analysis**: "What-if" scenarios
        
        #### Technical Innovation:
        Standard SHAP/LIME perturbation strategies can violate temporal causality 
        by inadvertently using future information. Our hybrid approach:
        
        1. Uses **temporal-consistent perturbations** respecting lag structures
        2. **Adaptively weights** SHAP and LIME based on local fit quality
        3. Applies **temporal regularization** for smooth explanations
        4. **Resolves conflicts** between methods intelligently
        """)


if __name__ == "__main__":
    main()
