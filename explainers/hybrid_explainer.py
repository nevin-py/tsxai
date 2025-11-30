"""
Hybrid SHAP-LIME Explainer for Time Series Forecasting.

This is the CORE NOVELTY of the project:
Combines SHAP's global consistency with LIME's local interpretability
while respecting temporal causality.

Key innovations:
1. Temporal-aware fusion of SHAP and LIME explanations
2. Confidence-weighted combination based on local model fit
3. Temporal coherence regularization
4. Automatic method selection based on data characteristics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats

from explainers.base_explainer import BaseExplainer, Explanation
from explainers.temporal_shap import TemporalSHAPExplainer, create_shap_explainer
from explainers.temporal_lime import TemporalLIMEExplainer, create_lime_explainer


@dataclass
class HybridExplanation(Explanation):
    """Extended explanation with hybrid-specific attributes."""
    shap_scores: Optional[np.ndarray] = None
    lime_scores: Optional[np.ndarray] = None
    fusion_weights: Optional[Tuple[float, float]] = None
    disagreement_indices: Optional[List[int]] = None
    
    def get_discrepancy_features(self, threshold: float = 0.2) -> List[Tuple[str, float, float, float]]:
        """
        Get features where SHAP and LIME disagree significantly.
        
        Returns list of (feature_name, shap_score, lime_score, difference)
        """
        if self.shap_scores is None or self.lime_scores is None:
            return []
        
        discrepancies = []
        
        # Normalize scores for comparison
        shap_norm = self.shap_scores / (np.abs(self.shap_scores).max() + 1e-8)
        lime_norm = self.lime_scores / (np.abs(self.lime_scores).max() + 1e-8)
        
        for i, name in enumerate(self.feature_names):
            diff = abs(shap_norm[i] - lime_norm[i])
            if diff > threshold:
                discrepancies.append((
                    name,
                    self.shap_scores[i],
                    self.lime_scores[i],
                    diff
                ))
        
        return sorted(discrepancies, key=lambda x: -x[3])


class HybridSHAPLIMEExplainer(BaseExplainer):
    """
    Hybrid explainer combining SHAP and LIME for time series.
    
    Fusion Strategy:
    1. Compute both SHAP and LIME explanations
    2. Assess agreement and local model quality
    3. Use adaptive weighting based on:
       - LIME's local R² (local fit quality)
       - SHAP-LIME rank correlation (agreement)
       - Temporal coherence of each method
    4. Apply temporal regularization to final explanations
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: List[str],
        training_data: np.ndarray,
        model_object=None,  # For TreeSHAP
        temporal_features: Optional[List[str]] = None,
        fusion_method: str = 'adaptive',  # 'adaptive', 'average', 'weighted', 'conflict_resolution'
        shap_weight: float = 0.5,
        temporal_smoothing: bool = True
    ):
        """
        Initialize Hybrid SHAP-LIME explainer.
        
        Args:
            model: Prediction function
            feature_names: Feature names
            training_data: Historical data for reference
            model_object: Actual model object (for TreeSHAP optimization)
            temporal_features: Features with temporal structure
            fusion_method: How to combine SHAP and LIME
            shap_weight: Base weight for SHAP (only for 'weighted' fusion)
            temporal_smoothing: Apply temporal smoothing to explanations
        """
        super().__init__(model, feature_names, temporal_features)
        
        self.training_data = training_data
        self.fusion_method = fusion_method
        self.base_shap_weight = shap_weight
        self.temporal_smoothing = temporal_smoothing
        
        # Initialize component explainers
        self._init_explainers(model, model_object, training_data)
        
        # Cache for temporal coherence computation
        self._explanation_history = []
        
    def _init_explainers(
        self,
        model: Callable,
        model_object,
        training_data: np.ndarray
    ):
        """Initialize SHAP and LIME explainers."""
        # SHAP explainer
        if model_object is not None:
            self.shap_explainer = create_shap_explainer(
                model=model_object,
                feature_names=self.feature_names,
                model_type='xgboost'  # Assumes tree model if object provided
            )
        else:
            self.shap_explainer = create_shap_explainer(
                model=model,
                feature_names=self.feature_names,
                background_data=training_data,
                model_type='other'
            )
        
        # LIME explainer
        self.lime_explainer = create_lime_explainer(
            model=model,
            feature_names=self.feature_names,
            training_data=training_data
        )
    
    def _compute_adaptive_weights(
        self,
        shap_exp: Explanation,
        lime_exp: Explanation,
        x: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute adaptive weights for SHAP and LIME fusion.
        
        Factors considered:
        1. LIME local model fit (R²) - higher = more weight to LIME
        2. SHAP-LIME agreement - high agreement = equal weights
        3. Historical coherence of each method
        """
        # Factor 1: LIME local fit quality
        lime_r2 = lime_exp.confidence[0] if lime_exp.confidence is not None else 0.5
        lime_confidence = np.clip(lime_r2, 0.1, 0.9)
        
        # Factor 2: Agreement between methods
        shap_ranks = stats.rankdata(np.abs(shap_exp.importance_scores))
        lime_ranks = stats.rankdata(np.abs(lime_exp.importance_scores))
        rank_corr, _ = stats.spearmanr(shap_ranks, lime_ranks)
        rank_corr = max(0, rank_corr)  # Negative correlation is concerning
        
        # Factor 3: Temporal coherence (if history available)
        shap_coherence = 1.0
        lime_coherence = 1.0
        
        if len(self._explanation_history) > 0:
            prev_shap = self._explanation_history[-1].get('shap', shap_exp.importance_scores)
            prev_lime = self._explanation_history[-1].get('lime', lime_exp.importance_scores)
            
            shap_coherence = 1.0 / (1.0 + np.linalg.norm(shap_exp.importance_scores - prev_shap))
            lime_coherence = 1.0 / (1.0 + np.linalg.norm(lime_exp.importance_scores - prev_lime))
        
        # Combine factors
        # High LIME R² -> favor LIME
        # High agreement -> equal weights
        # Better coherence -> favor that method
        
        agreement_factor = 0.5 + 0.3 * (1 - rank_corr)  # Less agreement = more SHAP bias
        coherence_ratio = shap_coherence / (shap_coherence + lime_coherence + 1e-8)
        
        # Base weight adjusted by factors
        shap_weight = self.base_shap_weight
        shap_weight *= (1 + 0.2 * (1 - lime_confidence))  # Lower LIME R² -> more SHAP
        shap_weight *= (0.8 + 0.4 * coherence_ratio)  # Better SHAP coherence -> more SHAP
        shap_weight *= agreement_factor
        
        # Normalize
        shap_weight = np.clip(shap_weight, 0.2, 0.8)
        lime_weight = 1.0 - shap_weight
        
        return shap_weight, lime_weight
    
    def _conflict_resolution_fusion(
        self,
        shap_scores: np.ndarray,
        lime_scores: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Intelligent fusion that resolves conflicts between SHAP and LIME.
        
        Strategy:
        - Where methods agree: average
        - Where methods disagree: investigate why and choose appropriately
        """
        n_features = len(shap_scores)
        fused_scores = np.zeros(n_features)
        conflict_indices = []
        
        # Normalize for comparison
        shap_norm = shap_scores / (np.abs(shap_scores).max() + 1e-8)
        lime_norm = lime_scores / (np.abs(lime_scores).max() + 1e-8)
        
        for i in range(n_features):
            # Check for sign disagreement (fundamental conflict)
            sign_agree = np.sign(shap_norm[i]) == np.sign(lime_norm[i])
            
            # Check for magnitude disagreement
            magnitude_diff = abs(shap_norm[i] - lime_norm[i])
            
            if not sign_agree or magnitude_diff > 0.5:
                conflict_indices.append(i)
                
                # Resolution strategy: favor the method with stronger signal
                if abs(shap_scores[i]) > abs(lime_scores[i]):
                    fused_scores[i] = shap_scores[i] * 0.7 + lime_scores[i] * 0.3
                else:
                    fused_scores[i] = shap_scores[i] * 0.3 + lime_scores[i] * 0.7
            else:
                # Agreement: simple average
                fused_scores[i] = (shap_scores[i] + lime_scores[i]) / 2
        
        return fused_scores, conflict_indices
    
    def _temporal_regularization(
        self,
        current_scores: np.ndarray,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Apply temporal regularization to smooth explanations over time.
        
        Prevents explanations from changing too rapidly between timesteps.
        """
        if len(self._explanation_history) == 0:
            return current_scores
        
        prev_scores = self._explanation_history[-1].get('hybrid', current_scores)
        
        # Exponential moving average
        smoothed = alpha * current_scores + (1 - alpha) * prev_scores
        
        return smoothed
    
    def explain_instance(
        self,
        x: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None
    ) -> HybridExplanation:
        """
        Generate hybrid SHAP-LIME explanation.
        """
        x = np.array(x).flatten()
        
        # Get SHAP explanation
        shap_exp = self.shap_explainer.explain_instance(x, timestamp)
        
        # Get LIME explanation
        lime_exp = self.lime_explainer.explain_instance(x, timestamp)
        
        # Fuse explanations based on method
        if self.fusion_method == 'average':
            fused_scores = (shap_exp.importance_scores + lime_exp.importance_scores) / 2
            weights = (0.5, 0.5)
            conflict_indices = []
            
        elif self.fusion_method == 'weighted':
            weights = (self.base_shap_weight, 1 - self.base_shap_weight)
            fused_scores = (
                weights[0] * shap_exp.importance_scores + 
                weights[1] * lime_exp.importance_scores
            )
            conflict_indices = []
            
        elif self.fusion_method == 'adaptive':
            weights = self._compute_adaptive_weights(shap_exp, lime_exp, x)
            fused_scores = (
                weights[0] * shap_exp.importance_scores + 
                weights[1] * lime_exp.importance_scores
            )
            conflict_indices = []
            
        elif self.fusion_method == 'conflict_resolution':
            fused_scores, conflict_indices = self._conflict_resolution_fusion(
                shap_exp.importance_scores,
                lime_exp.importance_scores
            )
            weights = (0.5, 0.5)  # Varies per feature
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Apply temporal regularization if enabled
        if self.temporal_smoothing:
            fused_scores = self._temporal_regularization(fused_scores)
        
        # Update history
        self._explanation_history.append({
            'shap': shap_exp.importance_scores,
            'lime': lime_exp.importance_scores,
            'hybrid': fused_scores
        })
        
        # Keep history bounded
        if len(self._explanation_history) > 100:
            self._explanation_history = self._explanation_history[-100:]
        
        return HybridExplanation(
            feature_names=self.feature_names,
            feature_values=x,
            importance_scores=fused_scores,
            base_value=(shap_exp.base_value + lime_exp.base_value) / 2,
            prediction=shap_exp.prediction,
            timestamp=timestamp,
            method='hybrid_shap_lime',
            shap_scores=shap_exp.importance_scores,
            lime_scores=lime_exp.importance_scores,
            fusion_weights=weights,
            disagreement_indices=conflict_indices
        )
    
    def explain_temporal_window(
        self,
        X: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[HybridExplanation]:
        """
        Generate hybrid explanations for a temporal window.
        """
        # Reset history for new window
        self._explanation_history = []
        
        explanations = []
        
        for i in range(len(X)):
            ts = timestamps[i] if timestamps is not None else None
            exp = self.explain_instance(X[i], timestamp=ts)
            explanations.append(exp)
        
        return explanations
    
    def compare_methods(
        self,
        x: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate comparison table of SHAP, LIME, and Hybrid explanations.
        """
        exp = self.explain_instance(x)
        
        data = []
        for i, name in enumerate(self.feature_names):
            data.append({
                'Feature': name,
                'Value': exp.feature_values[i],
                'SHAP': exp.shap_scores[i],
                'LIME': exp.lime_scores[i],
                'Hybrid': exp.importance_scores[i],
                'Abs_SHAP': abs(exp.shap_scores[i]),
                'Abs_LIME': abs(exp.lime_scores[i]),
                'Abs_Hybrid': abs(exp.importance_scores[i]),
                'Conflict': i in (exp.disagreement_indices or [])
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Abs_Hybrid', ascending=False)
        
        return df.head(top_k)
    
    def get_temporal_importance_matrix(
        self,
        explanations: List[HybridExplanation]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create matrices of importance scores over time.
        
        Returns:
            shap_matrix: (n_timesteps, n_features) SHAP scores
            lime_matrix: (n_timesteps, n_features) LIME scores
            hybrid_matrix: (n_timesteps, n_features) Hybrid scores
        """
        n_timesteps = len(explanations)
        n_features = len(self.feature_names)
        
        shap_matrix = np.zeros((n_timesteps, n_features))
        lime_matrix = np.zeros((n_timesteps, n_features))
        hybrid_matrix = np.zeros((n_timesteps, n_features))
        
        for i, exp in enumerate(explanations):
            shap_matrix[i] = exp.shap_scores
            lime_matrix[i] = exp.lime_scores
            hybrid_matrix[i] = exp.importance_scores
        
        return shap_matrix, lime_matrix, hybrid_matrix
    
    def compute_coherence_scores(
        self,
        explanations: List[HybridExplanation]
    ) -> Dict[str, float]:
        """
        Compute temporal coherence scores for each method.
        """
        shap_matrix, lime_matrix, hybrid_matrix = self.get_temporal_importance_matrix(explanations)
        
        def coherence(matrix):
            diffs = np.diff(matrix, axis=0)
            avg_change = np.mean(np.abs(diffs))
            avg_magnitude = np.mean(np.abs(matrix))
            return 1.0 / (1.0 + avg_change / (avg_magnitude + 1e-8))
        
        return {
            'shap_coherence': coherence(shap_matrix),
            'lime_coherence': coherence(lime_matrix),
            'hybrid_coherence': coherence(hybrid_matrix)
        }


def create_hybrid_explainer(
    model,
    feature_names: List[str],
    training_data: np.ndarray,
    model_object=None,
    fusion_method: str = 'adaptive',
    **kwargs
) -> HybridSHAPLIMEExplainer:
    """
    Factory function to create hybrid explainer.
    """
    predict_func = model.predict if hasattr(model, 'predict') else model
    
    return HybridSHAPLIMEExplainer(
        model=predict_func,
        feature_names=feature_names,
        training_data=training_data,
        model_object=model_object,
        fusion_method=fusion_method,
        **kwargs
    )
