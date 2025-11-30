"""
Tests for explainer modules.
"""

import numpy as np
import pytest

from explainers.base_explainer import (
    BaseExplainer,
    Explanation,
    TemporalPerturbationStrategy
)
from explainers.hybrid_explainer import (
    HybridSHAPLIMEExplainer,
    HybridExplanation,
    create_hybrid_explainer
)


class TestExplanation:
    """Test suite for Explanation dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        exp = Explanation(
            feature_names=['a', 'b', 'c'],
            feature_values=np.array([1.0, 2.0, 3.0]),
            importance_scores=np.array([0.5, 0.3, 0.2]),
            base_value=1.0,
            prediction=2.5,
            method='test'
        )
        
        d = exp.to_dict()
        
        assert d['feature_names'] == ['a', 'b', 'c']
        assert d['importance_scores'] == [0.5, 0.3, 0.2]
        assert d['base_value'] == 1.0
        assert d['prediction'] == 2.5
        assert d['method'] == 'test'
    
    def test_top_k_features(self):
        """Test getting top-k important features."""
        exp = Explanation(
            feature_names=['a', 'b', 'c', 'd'],
            feature_values=np.array([1.0, 2.0, 3.0, 4.0]),
            importance_scores=np.array([0.1, 0.5, -0.8, 0.2]),
            base_value=1.0,
            prediction=2.5
        )
        
        top_2 = exp.top_k_features(k=2)
        
        assert len(top_2) == 2
        assert top_2[0][0] == 'c'  # Highest absolute value (-0.8)
        assert top_2[1][0] == 'b'  # Second highest (0.5)


class TestTemporalPerturbationStrategy:
    """Test suite for TemporalPerturbationStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create a perturbation strategy for testing."""
        np.random.seed(42)
        reference_data = np.random.randn(100, 5)
        feature_names = ['price_lag_1', 'price_lag_2', 'price_rolling_mean_7', 'volume', 'day_of_week']
        return TemporalPerturbationStrategy(reference_data, feature_names)
    
    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert len(strategy.feature_stats) == 5
        assert 'mean' in strategy.feature_stats['price_lag_1']
        assert 'std' in strategy.feature_stats['price_lag_1']
    
    def test_perturb_gaussian(self, strategy):
        """Test Gaussian perturbation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        perturbations = strategy.perturb_gaussian(x, n_samples=50)
        
        assert perturbations.shape == (50, 5)
        # Perturbations should be centered around original values
        assert np.abs(perturbations.mean(axis=0) - x).max() < 0.5
    
    def test_perturb_temporal_consistent(self, strategy):
        """Test temporal-consistent perturbation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        perturbations = strategy.perturb_temporal_consistent(x, n_samples=50)
        
        assert perturbations.shape == (50, 5)
    
    def test_perturb_by_replacement(self, strategy):
        """Test replacement-based perturbation."""
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        perturbations = strategy.perturb_by_replacement(x, n_samples=50)
        
        assert perturbations.shape == (50, 5)
        # Some values should be different from original
        assert not np.allclose(perturbations, x)
    
    def test_group_temporal_features(self, strategy):
        """Test temporal feature grouping."""
        groups = strategy._group_temporal_features()
        
        assert 'price' in groups
        assert len(groups['price']) == 3  # lag_1, lag_2, rolling_mean_7


class TestHybridExplainer:
    """Test suite for HybridSHAPLIMEExplainer."""
    
    def test_create_hybrid_explainer(self, trained_xgboost_model):
        """Test hybrid explainer creation via factory function."""
        model, X, y, feature_names = trained_xgboost_model
        
        explainer = create_hybrid_explainer(
            model=model,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model
        )
        
        assert isinstance(explainer, HybridSHAPLIMEExplainer)
        assert explainer.feature_names == feature_names
    
    def test_explain_instance(self, trained_xgboost_model):
        """Test single instance explanation."""
        model, X, y, feature_names = trained_xgboost_model
        
        explainer = create_hybrid_explainer(
            model=model,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model,
            fusion_method='adaptive'
        )
        
        exp = explainer.explain_instance(X[90])
        
        assert isinstance(exp, HybridExplanation)
        assert len(exp.importance_scores) == len(feature_names)
        assert exp.shap_scores is not None
        assert exp.lime_scores is not None
        assert exp.fusion_weights is not None
    
    def test_explain_temporal_window(self, trained_xgboost_model):
        """Test temporal window explanation."""
        model, X, y, feature_names = trained_xgboost_model
        
        explainer = create_hybrid_explainer(
            model=model,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model
        )
        
        explanations = explainer.explain_temporal_window(X[85:90])
        
        assert len(explanations) == 5
        assert all(isinstance(e, HybridExplanation) for e in explanations)
    
    def test_fusion_methods(self, trained_xgboost_model):
        """Test different fusion methods."""
        model, X, y, feature_names = trained_xgboost_model
        
        for method in ['average', 'weighted', 'adaptive', 'conflict_resolution']:
            explainer = create_hybrid_explainer(
                model=model,
                feature_names=feature_names,
                training_data=X[:80],
                model_object=model.model,
                fusion_method=method
            )
            
            exp = explainer.explain_instance(X[90])
            
            assert exp is not None
            assert len(exp.importance_scores) == len(feature_names)
    
    def test_compare_methods(self, trained_xgboost_model):
        """Test method comparison table."""
        model, X, y, feature_names = trained_xgboost_model
        
        explainer = create_hybrid_explainer(
            model=model,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model
        )
        
        comparison_df = explainer.compare_methods(X[90], top_k=5)
        
        assert len(comparison_df) == 5
        assert 'SHAP' in comparison_df.columns
        assert 'LIME' in comparison_df.columns
        assert 'Hybrid' in comparison_df.columns
    
    def test_temporal_importance_matrix(self, trained_xgboost_model):
        """Test temporal importance matrix generation."""
        model, X, y, feature_names = trained_xgboost_model
        
        explainer = create_hybrid_explainer(
            model=model,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model
        )
        
        explanations = explainer.explain_temporal_window(X[85:90])
        shap_matrix, lime_matrix, hybrid_matrix = explainer.get_temporal_importance_matrix(explanations)
        
        assert shap_matrix.shape == (5, len(feature_names))
        assert lime_matrix.shape == (5, len(feature_names))
        assert hybrid_matrix.shape == (5, len(feature_names))


class TestHybridExplanation:
    """Test suite for HybridExplanation dataclass."""
    
    def test_get_discrepancy_features(self):
        """Test discrepancy detection between SHAP and LIME."""
        exp = HybridExplanation(
            feature_names=['a', 'b', 'c'],
            feature_values=np.array([1.0, 2.0, 3.0]),
            importance_scores=np.array([0.5, 0.3, 0.2]),
            base_value=1.0,
            prediction=2.5,
            shap_scores=np.array([0.8, 0.1, 0.1]),  # SHAP says 'a' is important
            lime_scores=np.array([0.1, 0.8, 0.1]),  # LIME says 'b' is important
        )
        
        discrepancies = exp.get_discrepancy_features(threshold=0.3)
        
        assert len(discrepancies) > 0
        # 'a' and 'b' should have high discrepancy
        feature_names = [d[0] for d in discrepancies]
        assert 'a' in feature_names or 'b' in feature_names
