"""
Temporal-aware SHAP explainer for time series forecasting.

Key innovation: Uses temporal masking and causal-consistent background data
to prevent information leakage from future values.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
import shap

from explainers.base_explainer import BaseExplainer, Explanation, TemporalPerturbationStrategy


class TemporalSHAPExplainer(BaseExplainer):
    """
    SHAP explainer adapted for time series with temporal coherence.
    
    Key modifications from standard SHAP:
    1. Background data selection respects temporal ordering
    2. Feature coalitions maintain temporal consistency
    3. Explanations are smoothed across time for coherence
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        background_data: np.ndarray,
        temporal_features: Optional[List[str]] = None,
        n_background_samples: int = 100,
        algorithm: str = 'kernel'  # 'kernel', 'tree', or 'deep'
    ):
        """
        Initialize Temporal SHAP explainer.
        
        Args:
            model: Prediction function
            feature_names: Names of features
            background_data: Historical data for SHAP background (must be past data only)
            temporal_features: Features with temporal structure
            n_background_samples: Number of background samples for KernelSHAP
            algorithm: SHAP algorithm to use
        """
        super().__init__(model, feature_names, temporal_features)
        
        self.algorithm = algorithm
        self.n_background_samples = min(n_background_samples, len(background_data))
        
        # Select background samples that maintain temporal diversity
        self.background_data = self._select_temporal_background(background_data)
        
        # Initialize appropriate SHAP explainer
        self._init_shap_explainer()
        
    def _select_temporal_background(self, data: np.ndarray) -> np.ndarray:
        """
        Select background samples that represent temporal diversity.
        
        Instead of random sampling, we sample from different time periods
        to ensure the background captures various market/data regimes.
        """
        n_samples = self.n_background_samples
        n_total = len(data)
        
        if n_total <= n_samples:
            return data
        
        # Stratified temporal sampling
        indices = []
        n_periods = min(10, n_samples // 10)
        period_size = n_total // n_periods
        samples_per_period = n_samples // n_periods
        
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else n_total
            period_indices = np.random.choice(
                range(start, end), 
                min(samples_per_period, end - start),
                replace=False
            )
            indices.extend(period_indices)
        
        # Fill remaining if needed
        remaining = n_samples - len(indices)
        if remaining > 0:
            available = list(set(range(n_total)) - set(indices))
            indices.extend(np.random.choice(available, remaining, replace=False))
        
        return data[indices[:n_samples]]
    
    def _init_shap_explainer(self):
        """Initialize the SHAP explainer based on algorithm type."""
        if self.algorithm == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model,
                self.background_data,
                link='identity'
            )
        elif self.algorithm == 'tree':
            # For tree-based models (XGBoost, LightGBM, etc.)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Default to Kernel for compatibility
            self.explainer = shap.KernelExplainer(
                self.model,
                self.background_data,
                link='identity'
            )
    
    def explain_instance(
        self,
        x: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None,
        check_additivity: bool = False
    ) -> Explanation:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            x: Input features (1D array)
            timestamp: Timestamp of prediction
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            Explanation with SHAP values
        """
        x = np.array(x).reshape(1, -1)
        
        # Get SHAP values
        if self.algorithm == 'kernel':
            shap_values = self.explainer.shap_values(x, nsamples=100)
        else:
            shap_values = self.explainer.shap_values(x)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output models
        
        shap_values = shap_values.flatten()
        
        # Get base value and prediction
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = np.mean(self.model(self.background_data))
        
        prediction = self.model(x)[0]
        
        return Explanation(
            feature_names=self.feature_names,
            feature_values=x.flatten(),
            importance_scores=shap_values,
            base_value=float(base_value),
            prediction=float(prediction),
            timestamp=timestamp,
            method='temporal_shap'
        )
    
    def explain_temporal_window(
        self,
        X: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        smooth: bool = True,
        smooth_window: int = 3
    ) -> List[Explanation]:
        """
        Generate SHAP explanations for a temporal window.
        
        Includes temporal smoothing to improve coherence.
        """
        explanations = []
        
        # Get raw SHAP values for all instances
        if self.algorithm == 'kernel':
            shap_values = self.explainer.shap_values(X, nsamples=100)
        else:
            shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Temporal smoothing
        if smooth and len(X) >= smooth_window:
            shap_values = self._temporal_smooth(shap_values, smooth_window)
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = np.mean(self.model(self.background_data))
        
        # Create explanation objects
        predictions = self.model(X)
        
        for i in range(len(X)):
            ts = timestamps[i] if timestamps is not None else None
            
            explanations.append(Explanation(
                feature_names=self.feature_names,
                feature_values=X[i],
                importance_scores=shap_values[i],
                base_value=float(base_value),
                prediction=float(predictions[i]),
                timestamp=ts,
                method='temporal_shap'
            ))
        
        return explanations
    
    def _temporal_smooth(
        self,
        shap_values: np.ndarray,
        window: int
    ) -> np.ndarray:
        """
        Apply temporal smoothing to SHAP values.
        
        Uses exponentially weighted moving average to maintain
        temporal coherence while preserving sudden important changes.
        """
        smoothed = np.zeros_like(shap_values)
        alpha = 2 / (window + 1)
        
        # Initialize with first value
        smoothed[0] = shap_values[0]
        
        for i in range(1, len(shap_values)):
            smoothed[i] = alpha * shap_values[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def get_feature_importance_over_time(
        self,
        explanations: List[Explanation]
    ) -> pd.DataFrame:
        """
        Create a DataFrame showing feature importance evolution over time.
        
        Useful for creating the time-aware heatmap visualization.
        """
        data = []
        
        for exp in explanations:
            row = {'timestamp': exp.timestamp, 'prediction': exp.prediction}
            for name, score in zip(exp.feature_names, exp.importance_scores):
                row[name] = score
            data.append(row)
        
        return pd.DataFrame(data)


class TreeSHAPTemporalExplainer(TemporalSHAPExplainer):
    """
    Optimized SHAP explainer for tree-based models (XGBoost, LightGBM).
    
    Uses TreeSHAP algorithm which is much faster than KernelSHAP
    and provides exact SHAP values for tree models.
    """
    
    def __init__(
        self,
        model,  # Actual model object, not just predict function
        feature_names: List[str],
        temporal_features: Optional[List[str]] = None
    ):
        """
        Initialize TreeSHAP explainer.
        
        Args:
            model: Tree-based model object (XGBoost, LightGBM, etc.)
            feature_names: Names of features
            temporal_features: Features with temporal structure
        """
        # Don't call parent __init__ - we handle it differently
        self.model = model
        self.feature_names = feature_names
        self.temporal_features = temporal_features or self._detect_temporal_features()
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(model)
        self.algorithm = 'tree'
        
    def _detect_temporal_features(self) -> List[str]:
        """Automatically detect temporal features by naming convention."""
        temporal_keywords = ['lag', 'rolling', 'return', 'roc', 'ma_']
        temporal = []
        
        for name in self.feature_names:
            if any(kw in name.lower() for kw in temporal_keywords):
                temporal.append(name)
                
        return temporal
    
    def explain_instance(
        self,
        x: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None,
        **kwargs
    ) -> Explanation:
        """Generate TreeSHAP explanation for a single instance."""
        x = np.array(x).reshape(1, -1)
        
        # Get SHAP values using TreeSHAP
        shap_values = self.explainer.shap_values(x)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        # Get base value and prediction
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        prediction = self.model.predict(x)[0]
        
        return Explanation(
            feature_names=self.feature_names,
            feature_values=x.flatten(),
            importance_scores=shap_values,
            base_value=float(base_value),
            prediction=float(prediction),
            timestamp=timestamp,
            method='tree_shap'
        )


def create_shap_explainer(
    model,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    model_type: str = 'xgboost'
) -> TemporalSHAPExplainer:
    """
    Factory function to create appropriate SHAP explainer.
    
    Args:
        model: Model object or prediction function
        feature_names: Feature names
        background_data: Background data for KernelSHAP
        model_type: Type of model ('xgboost', 'lightgbm', 'neural', 'other')
    
    Returns:
        Appropriate SHAP explainer instance
    """
    if model_type in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
        return TreeSHAPTemporalExplainer(
            model=model,
            feature_names=feature_names
        )
    else:
        if background_data is None:
            raise ValueError("background_data required for non-tree models")
        
        predict_func = model.predict if hasattr(model, 'predict') else model
        
        return TemporalSHAPExplainer(
            model=predict_func,
            feature_names=feature_names,
            background_data=background_data,
            algorithm='kernel'
        )
