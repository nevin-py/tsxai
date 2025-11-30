"""
Temporal-aware LIME explainer for time series forecasting.

Key innovation: Custom perturbation strategy that respects temporal causality
and generates realistic time series variations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from explainers.base_explainer import BaseExplainer, Explanation, TemporalPerturbationStrategy


class TemporalLIMEExplainer(BaseExplainer):
    """
    LIME explainer adapted for time series with temporal-aware perturbations.
    
    Key modifications from standard LIME:
    1. Perturbations respect temporal structure (lag relationships)
    2. Kernel weights consider temporal similarity
    3. Uses realistic historical variations instead of random noise
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        training_data: np.ndarray,
        temporal_features: Optional[List[str]] = None,
        kernel_width: float = 0.75,
        n_samples: int = 500
    ):
        """
        Initialize Temporal LIME explainer.
        
        Args:
            model: Prediction function
            feature_names: Names of features
            training_data: Historical data for perturbation reference
            temporal_features: Features with temporal structure
            kernel_width: Width of exponential kernel
            n_samples: Number of perturbed samples for local fitting
        """
        super().__init__(model, feature_names, temporal_features)
        
        self.training_data = training_data
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        
        # Initialize perturbation strategy
        self.perturbation_strategy = TemporalPerturbationStrategy(
            reference_data=training_data,
            feature_names=feature_names
        )
        
        # Feature statistics for perturbation
        self.feature_stats = self._compute_feature_stats()
        
    def _compute_feature_stats(self) -> Dict[str, Dict]:
        """Compute statistics for each feature."""
        stats = {}
        for i, name in enumerate(self.feature_names):
            values = self.training_data[:, i]
            stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return stats
    
    def _generate_temporal_perturbations(
        self,
        x: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate temporally-consistent perturbations.
        
        Returns:
            perturbed_data: Array of perturbed instances
            binary_data: Binary mask indicating which features were perturbed
        """
        n_features = len(x)
        perturbed_data = np.zeros((n_samples, n_features))
        binary_data = np.ones((n_samples, n_features))
        
        # Identify lag groups
        lag_groups = self._get_lag_groups()
        
        for i in range(n_samples):
            perturbed = x.copy()
            binary = np.ones(n_features)
            
            # Decide perturbation strategy for this sample
            strategy = np.random.choice(['gaussian', 'temporal', 'replacement'], p=[0.3, 0.4, 0.3])
            
            if strategy == 'gaussian':
                # Standard Gaussian perturbation with feature masking
                n_mask = np.random.randint(1, max(2, n_features // 3))
                mask_indices = np.random.choice(n_features, n_mask, replace=False)
                
                for idx in mask_indices:
                    std = self.feature_stats[self.feature_names[idx]]['std']
                    perturbed[idx] = x[idx] + np.random.normal(0, std * 0.3)
                    binary[idx] = 0
                    
            elif strategy == 'temporal':
                # Perturb lag groups together
                if lag_groups:
                    group_name = np.random.choice(list(lag_groups.keys()))
                    group_indices = lag_groups[group_name]
                    
                    # Apply correlated perturbation
                    base_shift = np.random.normal(0, 0.2)
                    for idx in group_indices:
                        std = self.feature_stats[self.feature_names[idx]]['std']
                        perturbed[idx] = x[idx] + base_shift * std
                        binary[idx] = 0
                        
            else:  # replacement
                # Replace with values from similar historical instance
                distances = np.linalg.norm(self.training_data - x, axis=1)
                similar_idx = np.random.choice(
                    np.argsort(distances)[:50]  # Top 50 similar
                )
                
                n_replace = np.random.randint(1, max(2, n_features // 4))
                replace_indices = np.random.choice(n_features, n_replace, replace=False)
                
                perturbed[replace_indices] = self.training_data[similar_idx, replace_indices]
                binary[replace_indices] = 0
            
            perturbed_data[i] = perturbed
            binary_data[i] = binary
            
        return perturbed_data, binary_data
    
    def _get_lag_groups(self) -> Dict[str, List[int]]:
        """Group features by their lag relationship."""
        groups = {}
        
        for i, name in enumerate(self.feature_names):
            # Extract base name (e.g., 'close' from 'close_lag_1')
            if '_lag_' in name:
                base = name.split('_lag_')[0]
            elif '_rolling_' in name:
                parts = name.rsplit('_rolling_', 1)
                base = parts[0]
            else:
                continue  # Non-temporal feature
                
            if base not in groups:
                groups[base] = []
            groups[base].append(i)
            
        return groups
    
    def _compute_kernel_weights(
        self,
        x: np.ndarray,
        perturbed_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute kernel weights based on distance from original instance.
        
        Uses exponential kernel with temporal-aware distance.
        """
        # Normalize features for distance calculation
        std = np.array([self.feature_stats[n]['std'] for n in self.feature_names])
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        x_normalized = x / std
        perturbed_normalized = perturbed_data / std
        
        # Euclidean distance
        distances = np.sqrt(np.sum((perturbed_normalized - x_normalized) ** 2, axis=1))
        
        # Exponential kernel
        weights = np.exp(-distances ** 2 / (self.kernel_width ** 2))
        
        return weights
    
    def _fit_local_model(
        self,
        x: np.ndarray,
        perturbed_data: np.ndarray,
        binary_data: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Fit weighted linear model locally.
        
        Returns:
            coefficients: Feature importance scores
            r2: Local model fit quality
        """
        # Fit ridge regression
        model = Ridge(alpha=1.0)
        model.fit(binary_data, predictions, sample_weight=weights)
        
        coefficients = model.coef_
        
        # Compute local R²
        predictions_local = model.predict(binary_data)
        r2 = r2_score(predictions, predictions_local, sample_weight=weights)
        
        return coefficients, r2
    
    def explain_instance(
        self,
        x: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Explanation:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            x: Input features (1D array)
            timestamp: Timestamp of prediction
            
        Returns:
            Explanation with LIME coefficients
        """
        x = np.array(x).flatten()
        
        # Generate perturbations
        perturbed_data, binary_data = self._generate_temporal_perturbations(x, self.n_samples)
        
        # Get predictions for perturbed samples
        predictions = self.model(perturbed_data)
        
        # Compute kernel weights
        weights = self._compute_kernel_weights(x, perturbed_data)
        
        # Fit local linear model
        coefficients, r2 = self._fit_local_model(
            x, perturbed_data, binary_data, predictions, weights
        )
        
        # Get original prediction
        original_prediction = self.model(x.reshape(1, -1))[0]
        
        return Explanation(
            feature_names=self.feature_names,
            feature_values=x,
            importance_scores=coefficients,
            base_value=np.mean(predictions),  # Local model intercept
            prediction=float(original_prediction),
            timestamp=timestamp,
            method='temporal_lime',
            confidence=np.array([r2])  # Store R² as confidence measure
        )
    
    def explain_temporal_window(
        self,
        X: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[Explanation]:
        """
        Generate LIME explanations for a temporal window.
        """
        explanations = []
        
        for i in range(len(X)):
            ts = timestamps[i] if timestamps is not None else None
            exp = self.explain_instance(X[i], timestamp=ts)
            explanations.append(exp)
        
        return explanations
    
    def get_perturbation_quality_metrics(
        self,
        x: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate quality of perturbation strategy.
        
        Returns metrics indicating how realistic perturbations are.
        """
        perturbed_data, _ = self._generate_temporal_perturbations(x, n_samples)
        
        metrics = {}
        
        # Check if perturbed values are within realistic ranges
        in_range_count = 0
        for i, name in enumerate(self.feature_names):
            min_val = self.feature_stats[name]['min']
            max_val = self.feature_stats[name]['max']
            in_range = np.sum((perturbed_data[:, i] >= min_val) & 
                            (perturbed_data[:, i] <= max_val))
            in_range_count += in_range
        
        metrics['in_range_ratio'] = in_range_count / (n_samples * len(self.feature_names))
        
        # Check temporal consistency for lag features
        lag_groups = self._get_lag_groups()
        temporal_consistency = []
        
        for group_name, indices in lag_groups.items():
            if len(indices) > 1:
                for i in range(n_samples):
                    # Check if lag relationships are preserved
                    values = perturbed_data[i, indices]
                    # In a consistent perturbation, lag_1 should be close to lag_2's shifted value
                    consistency = np.corrcoef(values[:-1], values[1:])[0, 1]
                    if not np.isnan(consistency):
                        temporal_consistency.append(consistency)
        
        metrics['temporal_consistency'] = np.mean(temporal_consistency) if temporal_consistency else 1.0
        
        return metrics


class RollingWindowLIME(TemporalLIMEExplainer):
    """
    LIME variant that uses rolling window for perturbation generation.
    
    Only uses past data in a rolling window to generate perturbations,
    ensuring no future information leakage.
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        training_data: np.ndarray,
        window_size: int = 50,
        **kwargs
    ):
        super().__init__(model, feature_names, training_data, **kwargs)
        self.window_size = window_size
        
    def explain_with_rolling_reference(
        self,
        X: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        full_history: np.ndarray = None
    ) -> List[Explanation]:
        """
        Generate explanations using rolling window of past data.
        
        For each timestep t, only data from [t-window_size, t) is used
        for generating perturbations.
        """
        if full_history is None:
            full_history = self.training_data
            
        explanations = []
        
        for i in range(len(X)):
            # Update perturbation reference to only use past data
            start_idx = max(0, len(full_history) - self.window_size + i)
            end_idx = len(full_history) + i
            
            if end_idx > len(full_history):
                # Use most recent available data
                window_data = full_history[-self.window_size:]
            else:
                window_data = full_history[start_idx:end_idx]
            
            # Temporarily update perturbation strategy
            self.perturbation_strategy = TemporalPerturbationStrategy(
                reference_data=window_data,
                feature_names=self.feature_names
            )
            
            # Generate explanation
            ts = timestamps[i] if timestamps is not None else None
            exp = self.explain_instance(X[i], timestamp=ts)
            explanations.append(exp)
        
        # Restore original
        self.perturbation_strategy = TemporalPerturbationStrategy(
            reference_data=self.training_data,
            feature_names=self.feature_names
        )
        
        return explanations


def create_lime_explainer(
    model: Callable,
    feature_names: List[str],
    training_data: np.ndarray,
    use_rolling: bool = False,
    **kwargs
) -> TemporalLIMEExplainer:
    """
    Factory function to create LIME explainer.
    
    Args:
        model: Prediction function
        feature_names: Feature names
        training_data: Reference data for perturbations
        use_rolling: Whether to use rolling window variant
        **kwargs: Additional arguments for explainer
    
    Returns:
        LIME explainer instance
    """
    if use_rolling:
        return RollingWindowLIME(
            model=model,
            feature_names=feature_names,
            training_data=training_data,
            **kwargs
        )
    else:
        return TemporalLIMEExplainer(
            model=model,
            feature_names=feature_names,
            training_data=training_data,
            **kwargs
        )
