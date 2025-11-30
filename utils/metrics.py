"""
Evaluation metrics for forecasting and explanation quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ForecastMetrics:
    """Metrics for evaluating forecast quality."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Mean Absolute Scaled Error."""
        naive_mae = np.mean(np.abs(np.diff(y_train)))
        return mean_absolute_error(y_true, y_pred) / naive_mae
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient."""
        return r2_score(y_true, y_pred)
    
    @classmethod
    def compute_all(cls, y_true: np.ndarray, y_pred: np.ndarray, 
                    y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all forecast metrics."""
        metrics = {
            'RMSE': cls.rmse(y_true, y_pred),
            'MAE': cls.mae(y_true, y_pred),
            'MAPE': cls.mape(y_true, y_pred),
            'SMAPE': cls.smape(y_true, y_pred),
            'R2': cls.r2(y_true, y_pred)
        }
        
        if y_train is not None:
            metrics['MASE'] = cls.mase(y_true, y_pred, y_train)
            
        return metrics


class ExplanationMetrics:
    """Metrics for evaluating explanation quality."""
    
    @staticmethod
    def faithfulness_correlation(
        model_func,
        X: np.ndarray,
        explanations: np.ndarray,
        n_samples: int = 100
    ) -> float:
        """
        Measure correlation between explanation importance and actual impact on predictions.
        
        Higher is better - shows explanations reflect true model behavior.
        """
        if len(X) < n_samples:
            n_samples = len(X)
            
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        correlations = []
        
        for idx in indices:
            x = X[idx]
            exp = explanations[idx]
            
            # Baseline prediction
            base_pred = model_func(x.reshape(1, -1))[0]
            
            # Perturb each feature and measure impact
            impacts = []
            for i in range(len(x)):
                x_perturbed = x.copy()
                x_perturbed[i] = 0  # Zero out feature
                perturbed_pred = model_func(x_perturbed.reshape(1, -1))[0]
                impacts.append(np.abs(base_pred - perturbed_pred))
            
            # Correlation between explanation weights and actual impacts
            if np.std(impacts) > 0 and np.std(exp) > 0:
                corr, _ = stats.pearsonr(np.abs(exp), impacts)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    @staticmethod
    def temporal_coherence(
        explanations_over_time: np.ndarray,
        window_size: int = 5
    ) -> float:
        """
        Measure how smoothly explanations change over consecutive time steps.
        
        Lower variation = higher coherence (explanations don't jump erratically).
        Returns score between 0 and 1, where 1 is perfect coherence.
        """
        n_timesteps = len(explanations_over_time)
        
        if n_timesteps < 2:
            return 1.0
        
        # Calculate differences between consecutive explanations
        diffs = []
        for i in range(1, n_timesteps):
            diff = np.linalg.norm(explanations_over_time[i] - explanations_over_time[i-1])
            diffs.append(diff)
        
        # Normalize by average explanation magnitude
        avg_magnitude = np.mean([np.linalg.norm(e) for e in explanations_over_time])
        normalized_diffs = np.array(diffs) / (avg_magnitude + 1e-8)
        
        # Convert to coherence score (lower diff = higher coherence)
        coherence = 1.0 / (1.0 + np.mean(normalized_diffs))
        
        return coherence
    
    @staticmethod
    def explanation_stability(
        explainer_func,
        x: np.ndarray,
        n_iterations: int = 10
    ) -> float:
        """
        Measure consistency of explanations across multiple runs.
        
        Returns coefficient of variation (lower is more stable).
        """
        explanations = []
        
        for _ in range(n_iterations):
            exp = explainer_func(x)
            explanations.append(exp)
        
        explanations = np.array(explanations)
        
        # Calculate coefficient of variation for each feature
        mean_exp = np.mean(explanations, axis=0)
        std_exp = np.std(explanations, axis=0)
        
        cv = std_exp / (np.abs(mean_exp) + 1e-8)
        
        # Average stability (1 - mean CV, bounded to [0, 1])
        stability = 1.0 - np.clip(np.mean(cv), 0, 1)
        
        return stability
    
    @staticmethod
    def feature_agreement(
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        top_k: int = 5
    ) -> float:
        """
        Measure agreement between SHAP and LIME top-k important features.
        
        Returns Jaccard similarity of top-k features.
        """
        shap_top_k = set(np.argsort(np.abs(shap_values))[-top_k:])
        lime_top_k = set(np.argsort(np.abs(lime_values))[-top_k:])
        
        intersection = len(shap_top_k & lime_top_k)
        union = len(shap_top_k | lime_top_k)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def rank_correlation(
        shap_values: np.ndarray,
        lime_values: np.ndarray
    ) -> float:
        """
        Spearman rank correlation between SHAP and LIME importance rankings.
        """
        shap_ranks = stats.rankdata(np.abs(shap_values))
        lime_ranks = stats.rankdata(np.abs(lime_values))
        
        corr, _ = stats.spearmanr(shap_ranks, lime_ranks)
        
        return corr if not np.isnan(corr) else 0.0


class HybridExplainerMetrics:
    """Metrics specific to evaluating the SHAP-LIME hybrid approach."""
    
    def __init__(self):
        self.forecast_metrics = ForecastMetrics()
        self.explanation_metrics = ExplanationMetrics()
    
    def compute_temporal_leakage_score(
        self,
        explanations: np.ndarray,
        feature_timestamps: np.ndarray,
        prediction_timestamp: float
    ) -> float:
        """
        Check if explanations improperly weight future features.
        
        Returns percentage of importance assigned to future features (should be 0).
        """
        future_mask = feature_timestamps > prediction_timestamp
        
        total_importance = np.sum(np.abs(explanations))
        future_importance = np.sum(np.abs(explanations[future_mask]))
        
        if total_importance == 0:
            return 0.0
            
        leakage_score = future_importance / total_importance
        
        return leakage_score
    
    def compare_methods(
        self,
        y_true: np.ndarray,
        shap_explanations: np.ndarray,
        lime_explanations: np.ndarray,
        hybrid_explanations: np.ndarray,
        model_func
    ) -> pd.DataFrame:
        """
        Compare SHAP, LIME, and Hybrid methods across multiple metrics.
        """
        results = []
        
        for name, explanations in [
            ('SHAP', shap_explanations),
            ('LIME', lime_explanations),
            ('Hybrid', hybrid_explanations)
        ]:
            metrics = {
                'Method': name,
                'Temporal Coherence': self.explanation_metrics.temporal_coherence(explanations),
                'Avg Magnitude': np.mean(np.abs(explanations)),
                'Std Magnitude': np.std(np.abs(explanations))
            }
            results.append(metrics)
        
        # Add agreement metrics
        results.append({
            'Method': 'SHAP-LIME Agreement',
            'Temporal Coherence': np.nan,
            'Avg Magnitude': self.explanation_metrics.feature_agreement(
                shap_explanations.mean(axis=0),
                lime_explanations.mean(axis=0)
            ),
            'Std Magnitude': self.explanation_metrics.rank_correlation(
                shap_explanations.mean(axis=0),
                lime_explanations.mean(axis=0)
            )
        })
        
        return pd.DataFrame(results)


def evaluate_full_pipeline(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: Dict[str, np.ndarray],
    y_train: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation of both forecasting and explanation quality.
    """
    # Forecast metrics
    y_pred = model.predict(X_test)
    forecast_metrics = ForecastMetrics.compute_all(y_test, y_pred, y_train)
    
    # Explanation metrics
    exp_metrics = {}
    
    if 'shap' in explanations and 'lime' in explanations:
        exp_metrics['shap_lime_agreement'] = ExplanationMetrics.feature_agreement(
            explanations['shap'].mean(axis=0),
            explanations['lime'].mean(axis=0)
        )
        exp_metrics['rank_correlation'] = ExplanationMetrics.rank_correlation(
            explanations['shap'].mean(axis=0),
            explanations['lime'].mean(axis=0)
        )
    
    for method_name, exp in explanations.items():
        exp_metrics[f'{method_name}_coherence'] = ExplanationMetrics.temporal_coherence(exp)
    
    return {
        'forecast_metrics': forecast_metrics,
        'explanation_metrics': exp_metrics
    }


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    print("Forecast Metrics:")
    metrics = ForecastMetrics.compute_all(y_true, y_pred)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test explanation metrics
    explanations = np.random.randn(50, 10)  # 50 timesteps, 10 features
    coherence = ExplanationMetrics.temporal_coherence(explanations)
    print(f"\nTemporal Coherence: {coherence:.4f}")
    
    # Test feature agreement
    shap_vals = np.random.randn(10)
    lime_vals = shap_vals + np.random.randn(10) * 0.1  # Similar but not identical
    agreement = ExplanationMetrics.feature_agreement(shap_vals, lime_vals)
    print(f"Feature Agreement (top-5): {agreement:.4f}")
