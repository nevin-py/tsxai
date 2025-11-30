"""
Tests for forecasting models.
"""

import numpy as np
import pytest

from models.xgboost_model import XGBoostForecaster, QuantileXGBoostForecaster
from models.base_model import ForecastResult


class TestXGBoostForecaster:
    """Test suite for XGBoostForecaster."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = XGBoostForecaster()
        
        assert model.model_name == "xgboost"
        assert model.is_fitted is False
        assert model.params['n_estimators'] == 100
        assert model.params['max_depth'] == 6
    
    def test_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = XGBoostForecaster(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05
        )
        
        assert model.params['n_estimators'] == 50
        assert model.params['max_depth'] == 4
        assert model.params['learning_rate'] == 0.05
    
    def test_fit(self, feature_matrix):
        """Test model training."""
        X, y, feature_names = feature_matrix
        model = XGBoostForecaster(n_estimators=10)
        
        model.fit(X[:80], y[:80], feature_names=feature_names)
        
        assert model.is_fitted is True
        assert model.feature_names == feature_names
        assert 'train_rmse' in model.training_history
    
    def test_fit_with_validation(self, feature_matrix):
        """Test model training with validation set."""
        X, y, feature_names = feature_matrix
        model = XGBoostForecaster(n_estimators=10)
        
        model.fit(X[:80], y[:80], X[80:], y[80:], feature_names=feature_names)
        
        assert model.is_fitted is True
        assert 'val_rmse' in model.training_history
    
    def test_predict(self, trained_xgboost_model):
        """Test model predictions."""
        model, X, y, _ = trained_xgboost_model
        
        predictions = model.predict(X[80:])
        
        assert predictions.shape == (20,)
        assert not np.isnan(predictions).any()
    
    def test_predict_not_fitted(self, feature_matrix):
        """Test prediction raises error when model not fitted."""
        X, y, _ = feature_matrix
        model = XGBoostForecaster()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)
    
    def test_predict_with_confidence(self, trained_xgboost_model):
        """Test predictions with confidence intervals."""
        model, X, y, _ = trained_xgboost_model
        
        result = model.predict_with_confidence(X[80:], confidence_level=0.95)
        
        assert isinstance(result, ForecastResult)
        assert result.predictions.shape == (20,)
        assert result.confidence_lower.shape == (20,)
        assert result.confidence_upper.shape == (20,)
        assert (result.confidence_lower <= result.predictions).all()
        assert (result.confidence_upper >= result.predictions).all()
    
    def test_feature_importance(self, trained_xgboost_model):
        """Test feature importance retrieval."""
        model, X, y, feature_names = trained_xgboost_model
        
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(feature_names)
        assert all(name in importance for name in feature_names)
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_feature_importance_not_fitted(self):
        """Test feature importance returns None when not fitted."""
        model = XGBoostForecaster()
        
        assert model.get_feature_importance() is None


class TestQuantileXGBoostForecaster:
    """Test suite for QuantileXGBoostForecaster."""
    
    def test_initialization(self):
        """Test quantile model initialization."""
        model = QuantileXGBoostForecaster()
        
        assert model.quantiles == (0.025, 0.5, 0.975)
    
    def test_fit_and_predict(self, feature_matrix):
        """Test quantile model training and prediction."""
        X, y, feature_names = feature_matrix
        model = QuantileXGBoostForecaster(n_estimators=10)
        
        model.fit(X[:80], y[:80], feature_names=feature_names)
        result = model.predict_with_confidence(X[80:])
        
        assert model.is_fitted is True
        assert len(model.models) == 3
        assert result.predictions.shape == (20,)
        # Lower quantile should be <= median <= upper quantile
        assert (result.confidence_lower <= result.predictions).all()
        assert (result.confidence_upper >= result.predictions).all()


class TestForecastResult:
    """Test suite for ForecastResult dataclass."""
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        result = ForecastResult(
            predictions=np.array([1.0, 2.0, 3.0]),
            confidence_lower=np.array([0.5, 1.5, 2.5]),
            confidence_upper=np.array([1.5, 2.5, 3.5])
        )
        
        df = result.to_dataframe()
        
        assert 'prediction' in df.columns
        assert 'lower' in df.columns
        assert 'upper' in df.columns
        assert len(df) == 3
