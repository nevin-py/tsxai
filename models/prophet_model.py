"""
Prophet model for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from models.base_model import BaseTimeSeriesModel, ForecastResult


class ProphetForecaster(BaseTimeSeriesModel):
    """
    Facebook Prophet-based time series forecaster.
    
    Best for time series with strong seasonal patterns and trend changes.
    Handles missing data and outliers well.
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        **kwargs
    ):
        super().__init__(model_name="prophet")
        
        self.params = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            **kwargs
        }
        
        self.model = None
        self.target_col = 'y'
        self.date_col = 'ds'
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'ProphetForecaster':
        """
        Train Prophet model.
        
        Note: Prophet primarily uses date for forecasting.
        Additional features can be added as regressors.
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create Prophet-compatible DataFrame
        if dates is None:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(y_train), freq='D')
        
        train_df = pd.DataFrame({
            'ds': dates,
            'y': y_train
        })
        
        # Initialize Prophet
        self.model = Prophet(**self.params)
        
        # Add external regressors (features)
        for i, name in enumerate(self.feature_names[:5]):  # Limit to top 5 features
            train_df[name] = X_train[:, i]
            self.model.add_regressor(name)
        
        # Fit model
        self.model.fit(train_df)
        self.is_fitted = True
        
        # Store info for prediction
        self._train_end_date = dates[-1]
        self._n_regressors = min(5, len(self.feature_names))
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future DataFrame
        if dates is None:
            dates = pd.date_range(
                start=self._train_end_date + pd.Timedelta(days=1),
                periods=len(X),
                freq='D'
            )
        
        future_df = pd.DataFrame({'ds': dates})
        
        # Add regressors
        for i in range(self._n_regressors):
            future_df[self.feature_names[i]] = X[:, i]
        
        # Predict
        forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> ForecastResult:
        """Generate predictions with Prophet's built-in uncertainty."""
        if dates is None:
            dates = pd.date_range(
                start=self._train_end_date + pd.Timedelta(days=1),
                periods=len(X),
                freq='D'
            )
        
        future_df = pd.DataFrame({'ds': dates})
        
        for i in range(self._n_regressors):
            future_df[self.feature_names[i]] = X[:, i]
        
        forecast = self.model.predict(future_df)
        
        return ForecastResult(
            predictions=forecast['yhat'].values,
            timestamps=pd.DatetimeIndex(forecast['ds']),
            confidence_lower=forecast['yhat_lower'].values,
            confidence_upper=forecast['yhat_upper'].values,
            confidence_level=confidence_level
        )
    
    def get_components(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Get decomposed forecast components (trend, seasonality, etc.)."""
        if dates is None:
            dates = pd.date_range(
                start=self._train_end_date + pd.Timedelta(days=1),
                periods=len(X),
                freq='D'
            )
        
        future_df = pd.DataFrame({'ds': dates})
        
        for i in range(self._n_regressors):
            future_df[self.feature_names[i]] = X[:, i]
        
        return self.model.predict(future_df)


class SimpleProphetWrapper(BaseTimeSeriesModel):
    """
    Simplified Prophet wrapper that works with feature matrices.
    
    Converts features back to time series format for Prophet.
    """
    
    def __init__(self, **kwargs):
        super().__init__(model_name="simple_prophet")
        self.prophet_params = kwargs
        self.model = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'SimpleProphetWrapper':
        """Fit using just the target values."""
        try:
            from prophet import Prophet
        except ImportError:
            # Fall back to simple model if Prophet not available
            print("Prophet not available, using fallback")
            self.model = None
            self._mean = np.mean(y_train)
            self._std = np.std(y_train)
            self.is_fitted = True
            return self
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(y_train), freq='D')
        
        train_df = pd.DataFrame({
            'ds': dates,
            'y': y_train
        })
        
        self.model = Prophet(**self.prophet_params)
        self.model.fit(train_df)
        
        self._train_end_date = dates[-1]
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            # Fallback prediction
            return np.full(len(X), self._mean)
        
        dates = pd.date_range(
            start=self._train_end_date + pd.Timedelta(days=1),
            periods=len(X),
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': dates})
        forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with confidence intervals."""
        if self.model is None:
            preds = np.full(len(X), self._mean)
            margin = 1.96 * self._std
            return ForecastResult(
                predictions=preds,
                confidence_lower=preds - margin,
                confidence_upper=preds + margin,
                confidence_level=confidence_level
            )
        
        dates = pd.date_range(
            start=self._train_end_date + pd.Timedelta(days=1),
            periods=len(X),
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': dates})
        forecast = self.model.predict(future_df)
        
        return ForecastResult(
            predictions=forecast['yhat'].values,
            timestamps=pd.DatetimeIndex(forecast['ds']),
            confidence_lower=forecast['yhat_lower'].values,
            confidence_upper=forecast['yhat_upper'].values,
            confidence_level=confidence_level
        )
