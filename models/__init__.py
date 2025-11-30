"""Models module initialization."""

from models.base_model import BaseTimeSeriesModel, ForecastResult
from models.xgboost_model import XGBoostForecaster, QuantileXGBoostForecaster
from models.prophet_model import ProphetForecaster, SimpleProphetWrapper
from models.lstm_model import LSTMForecaster, SimpleLSTMWrapper


def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    models = {
        'xgboost': XGBoostForecaster,
        'prophet': SimpleProphetWrapper,
        'lstm': SimpleLSTMWrapper
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)
