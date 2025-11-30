"""
Pytest configuration and fixtures for TSXAI tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic time series with trend and seasonality
    trend = np.linspace(100, 150, n_samples)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly pattern
    noise = np.random.randn(n_samples) * 5
    
    close = trend + seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n_samples),
        'high': close + np.abs(np.random.randn(n_samples) * 2),
        'low': close - np.abs(np.random.randn(n_samples) * 2),
    })
    
    return df


@pytest.fixture
def feature_matrix():
    """Generate sample feature matrix for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names


@pytest.fixture
def trained_xgboost_model(feature_matrix):
    """Return a trained XGBoost model for testing."""
    from models.xgboost_model import XGBoostForecaster
    
    X, y, feature_names = feature_matrix
    
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    model.fit(X[:80], y[:80], X[80:], y[80:], feature_names=feature_names)
    
    return model, X, y, feature_names
