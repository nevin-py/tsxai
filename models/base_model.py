"""
Base model interface for time series forecasters.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Container for forecast results."""
    predictions: np.ndarray
    timestamps: Optional[pd.DatetimeIndex] = None
    confidence_lower: Optional[np.ndarray] = None
    confidence_upper: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for visualization."""
        data = {'prediction': self.predictions}
        
        if self.timestamps is not None:
            data['timestamp'] = self.timestamps
        
        if self.confidence_lower is not None:
            data['lower'] = self.confidence_lower
        if self.confidence_upper is not None:
            data['upper'] = self.confidence_upper
            
        return pd.DataFrame(data)


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models.
    
    All models must implement these methods to work with the explainer framework.
    """
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.is_fitted = False
        self.feature_names = []
        self.training_history = {}
        
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'BaseTimeSeriesModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X: Input features
            confidence_level: Confidence level for intervals
            
        Returns:
            ForecastResult with predictions and intervals
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get built-in feature importance if available.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        return None
    
    def save(self, path: str):
        """Save model to disk."""
        import joblib
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'BaseTimeSeriesModel':
        """Load model from disk."""
        import joblib
        return joblib.load(path)
