"""
Base explainer class for time series explanations.
Defines the interface and common functionality.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class Explanation:
    """Container for explanation results."""
    feature_names: List[str]
    feature_values: np.ndarray
    importance_scores: np.ndarray
    base_value: float
    prediction: float
    timestamp: Optional[pd.Timestamp] = None
    method: str = "base"
    confidence: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'feature_names': self.feature_names,
            'feature_values': self.feature_values.tolist(),
            'importance_scores': self.importance_scores.tolist(),
            'base_value': float(self.base_value),
            'prediction': float(self.prediction),
            'timestamp': str(self.timestamp) if self.timestamp else None,
            'method': self.method,
            'confidence': self.confidence.tolist() if self.confidence is not None else None
        }
    
    def top_k_features(self, k: int = 5) -> List[Tuple[str, float, float]]:
        """Get top-k most important features."""
        indices = np.argsort(np.abs(self.importance_scores))[-k:][::-1]
        return [
            (self.feature_names[i], self.feature_values[i], self.importance_scores[i])
            for i in indices
        ]


class BaseExplainer(ABC):
    """
    Abstract base class for time series explainers.
    
    Key principle: All perturbation strategies must respect temporal ordering
    to prevent data leakage.
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        temporal_features: Optional[List[str]] = None
    ):
        """
        Initialize explainer.
        
        Args:
            model: Prediction function that takes X and returns predictions
            feature_names: Names of all features
            temporal_features: Features that represent temporal information (lags, rolling)
        """
        self.model = model
        self.feature_names = feature_names
        self.temporal_features = temporal_features or self._detect_temporal_features()
        
    def _detect_temporal_features(self) -> List[str]:
        """Automatically detect temporal features by naming convention."""
        temporal_keywords = ['lag', 'rolling', 'return', 'roc', 'ma_']
        temporal = []
        
        for name in self.feature_names:
            if any(kw in name.lower() for kw in temporal_keywords):
                temporal.append(name)
                
        return temporal
    
    def _get_feature_temporal_order(self) -> Dict[str, int]:
        """
        Map features to their temporal order.
        
        Features with higher lag numbers are further in the past.
        Returns dict mapping feature_name -> temporal_distance
        """
        order = {}
        
        for name in self.feature_names:
            # Extract lag number if present
            if '_lag_' in name:
                try:
                    lag = int(name.split('_lag_')[1])
                    order[name] = lag
                except:
                    order[name] = 0
            elif '_rolling_' in name:
                # Rolling features look back, extract window
                try:
                    window = int(name.split('_')[-1])
                    order[name] = window
                except:
                    order[name] = 0
            else:
                order[name] = 0  # Non-temporal features
                
        return order
    
    @abstractmethod
    def explain_instance(
        self,
        x: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Explanation:
        """
        Generate explanation for a single instance.
        
        Args:
            x: Input features (1D array)
            timestamp: Timestamp of the prediction (for context)
            
        Returns:
            Explanation object with importance scores
        """
        pass
    
    @abstractmethod
    def explain_temporal_window(
        self,
        X: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[Explanation]:
        """
        Generate explanations for a window of time steps.
        
        Returns list of Explanation objects, one per timestep.
        """
        pass
    
    def validate_temporal_coherence(
        self,
        explanations: List[Explanation],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Validate that explanations are temporally coherent.
        
        Explanations shouldn't change drastically between consecutive timesteps
        unless there's a regime change in the data.
        """
        if len(explanations) < 2:
            return True, 1.0
            
        changes = []
        for i in range(1, len(explanations)):
            prev = np.array(explanations[i-1].importance_scores)
            curr = np.array(explanations[i].importance_scores)
            
            # Cosine similarity
            similarity = np.dot(prev, curr) / (np.linalg.norm(prev) * np.linalg.norm(curr) + 1e-8)
            changes.append(similarity)
        
        avg_coherence = np.mean(changes)
        is_coherent = avg_coherence > threshold
        
        return is_coherent, avg_coherence


class TemporalPerturbationStrategy:
    """
    Base class for temporal-aware perturbation strategies.
    
    Standard LIME/SHAP perturbations can violate temporal causality.
    This class provides strategies that respect time ordering.
    """
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        """
        Args:
            reference_data: Historical data for realistic perturbations (n_samples, n_features)
            feature_names: Names of features
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.feature_stats = self._compute_feature_stats()
        
    def _compute_feature_stats(self) -> Dict[str, Dict]:
        """Compute statistics for each feature from reference data."""
        stats = {}
        
        for i, name in enumerate(self.feature_names):
            values = self.reference_data[:, i]
            stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentiles': np.percentile(values, [10, 25, 50, 75, 90])
            }
            
        return stats
    
    def perturb_gaussian(
        self,
        x: np.ndarray,
        n_samples: int = 100,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Gaussian perturbation scaled by feature standard deviation.
        
        This is a basic perturbation - may not respect temporal ordering.
        """
        perturbations = np.zeros((n_samples, len(x)))
        
        for i, name in enumerate(self.feature_names):
            std = self.feature_stats[name]['std']
            noise = np.random.normal(0, scale * std, n_samples)
            perturbations[:, i] = x[i] + noise
            
        return perturbations
    
    def perturb_temporal_consistent(
        self,
        x: np.ndarray,
        n_samples: int = 100,
        respect_lag_structure: bool = True
    ) -> np.ndarray:
        """
        Temporal-consistent perturbation strategy.
        
        Key insight: When perturbing lagged features, maintain consistency
        across different lags of the same underlying series.
        
        If we perturb lag_1, we should proportionally adjust lag_2, lag_3, etc.
        """
        perturbations = np.zeros((n_samples, len(x)))
        
        # Group features by their base (e.g., 'close_lag_1' -> 'close')
        feature_groups = self._group_temporal_features()
        
        for sample_idx in range(n_samples):
            perturbed_x = x.copy()
            
            for base_name, feature_indices in feature_groups.items():
                if len(feature_indices) > 1 and respect_lag_structure:
                    # Correlated perturbation for related features
                    base_perturbation = np.random.normal(0, 0.1)
                    
                    for feat_idx, lag in feature_indices:
                        std = self.feature_stats[self.feature_names[feat_idx]]['std']
                        # Perturbation decays with lag (older values less affected)
                        decay = 1.0 / (1.0 + 0.1 * lag)
                        perturbed_x[feat_idx] = x[feat_idx] + base_perturbation * std * decay
                else:
                    # Independent perturbation
                    for feat_idx, _ in feature_indices:
                        std = self.feature_stats[self.feature_names[feat_idx]]['std']
                        perturbed_x[feat_idx] = x[feat_idx] + np.random.normal(0, 0.1 * std)
                        
            perturbations[sample_idx] = perturbed_x
            
        return perturbations
    
    def perturb_by_replacement(
        self,
        x: np.ndarray,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Replace features with values from similar historical periods.
        
        This creates perturbations that are guaranteed to be realistic.
        """
        perturbations = np.zeros((n_samples, len(x)))
        
        # Find similar samples in reference data
        distances = np.linalg.norm(self.reference_data - x, axis=1)
        similar_indices = np.argsort(distances)[:n_samples * 2]
        
        for i in range(n_samples):
            # Select random subset of features to replace
            n_replace = np.random.randint(1, len(x) // 2 + 1)
            replace_indices = np.random.choice(len(x), n_replace, replace=False)
            
            # Get replacement values from similar sample
            similar_idx = np.random.choice(similar_indices)
            
            perturbed_x = x.copy()
            perturbed_x[replace_indices] = self.reference_data[similar_idx, replace_indices]
            perturbations[i] = perturbed_x
            
        return perturbations
    
    def _group_temporal_features(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Group related temporal features.
        
        Returns dict: base_name -> [(feature_idx, lag_value), ...]
        """
        groups = {}
        
        for i, name in enumerate(self.feature_names):
            if '_lag_' in name:
                parts = name.split('_lag_')
                base = parts[0]
                lag = int(parts[1])
            elif '_rolling_' in name:
                parts = name.rsplit('_', 2)
                base = '_'.join(parts[:-2])
                lag = int(parts[-1])
            else:
                base = name
                lag = 0
                
            if base not in groups:
                groups[base] = []
            groups[base].append((i, lag))
            
        return groups
