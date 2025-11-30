"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest

from utils.metrics import ForecastMetrics, ExplanationMetrics


class TestForecastMetrics:
    """Test suite for ForecastMetrics."""
    
    def test_rmse(self):
        """Test RMSE calculation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        rmse = ForecastMetrics.rmse(actual, predicted)
        
        assert rmse > 0
        assert rmse < 1  # Small error expected
    
    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        
        rmse = ForecastMetrics.rmse(actual, predicted)
        
        assert rmse == 0.0
    
    def test_mae(self):
        """Test MAE calculation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        mae = ForecastMetrics.mae(actual, predicted)
        
        assert mae == 0.5
    
    def test_mape(self):
        """Test MAPE calculation."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 190.0, 330.0])
        
        mape = ForecastMetrics.mape(actual, predicted)
        
        assert mape > 0
        assert mape < 100  # Should be reasonable percentage
    
    def test_r2(self):
        """Test R² calculation."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.0, 3.1, 3.9, 5.0])
        
        r2 = ForecastMetrics.r2(actual, predicted)
        
        assert r2 > 0.9  # Good fit
        assert r2 <= 1.0
    
    def test_r2_perfect_prediction(self):
        """Test R² with perfect predictions."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        
        r2 = ForecastMetrics.r2(actual, predicted)
        
        assert r2 == 1.0
    
    def test_compute_all(self):
        """Test computing all metrics at once."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = ForecastMetrics.compute_all(actual, predicted)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestExplanationMetrics:
    """Test suite for ExplanationMetrics."""
    
    def test_temporal_coherence_stable(self):
        """Test temporal coherence with stable explanations."""
        # Create stable importance matrix (small changes over time)
        np.random.seed(42)
        base = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        importance_matrix = np.array([
            base + np.random.randn(5) * 0.01 for _ in range(10)
        ])
        
        coherence = ExplanationMetrics.temporal_coherence(importance_matrix)
        
        assert coherence > 0.5  # Should be high for stable explanations
    
    def test_temporal_coherence_unstable(self):
        """Test temporal coherence with unstable explanations."""
        # Create random importance matrix (high variance)
        np.random.seed(42)
        importance_matrix = np.random.randn(10, 5)
        
        coherence = ExplanationMetrics.temporal_coherence(importance_matrix)
        
        assert coherence < 0.5  # Should be lower for unstable explanations
    
    def test_rank_correlation_identical(self):
        """Test rank correlation with identical rankings."""
        scores1 = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        scores2 = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        
        corr = ExplanationMetrics.rank_correlation(scores1, scores2)
        
        assert corr == 1.0
    
    def test_rank_correlation_opposite(self):
        """Test rank correlation with opposite rankings."""
        scores1 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        scores2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        corr = ExplanationMetrics.rank_correlation(scores1, scores2)
        
        assert corr == -1.0
    
    def test_feature_agreement(self):
        """Test top-k feature agreement."""
        # Same top features but different scores
        scores1 = np.array([0.5, 0.4, 0.1, 0.0, 0.0])
        scores2 = np.array([0.6, 0.3, 0.05, 0.03, 0.02])
        
        agreement = ExplanationMetrics.feature_agreement(scores1, scores2, k=2)
        
        assert agreement == 1.0  # Top 2 features are the same
    
    def test_feature_agreement_partial(self):
        """Test partial feature agreement."""
        scores1 = np.array([0.5, 0.4, 0.1, 0.0, 0.0])
        scores2 = np.array([0.1, 0.5, 0.4, 0.0, 0.0])  # Different ordering
        
        agreement = ExplanationMetrics.feature_agreement(scores1, scores2, k=3)
        
        assert 0 < agreement < 1  # Partial overlap
