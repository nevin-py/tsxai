"""
XGBoost model for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import xgboost as xgb

from models.base_model import BaseTimeSeriesModel, ForecastResult


class XGBoostForecaster(BaseTimeSeriesModel):
    """
    XGBoost-based time series forecaster.
    
    Suitable for tabular features derived from time series.
    Works well with engineered features like lags and rolling statistics.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(model_name="xgboost")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Fast histogram method
            **kwargs
        }
        
        self.model = None
        self.residual_std = None  # For confidence intervals
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = False
    ) -> 'XGBoostForecaster':
        """
        Train XGBoost model.
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create XGBoost model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Fit with optional early stopping
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=verbose
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
        
        # Compute residual std for confidence intervals
        train_preds = self.model.predict(X_train)
        residuals = y_train - train_preds
        self.residual_std = np.std(residuals)
        
        self.is_fitted = True
        
        # Store training metrics
        self.training_history['train_rmse'] = np.sqrt(np.mean(residuals ** 2))
        
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_residuals = y_val - val_preds
            self.training_history['val_rmse'] = np.sqrt(np.mean(val_residuals ** 2))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate predictions with confidence intervals.
        
        Uses residual-based uncertainty estimation.
        """
        predictions = self.predict(X)
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Simple confidence interval based on training residuals
        margin = z * self.residual_std
        
        return ForecastResult(
            predictions=predictions,
            confidence_lower=predictions - margin,
            confidence_upper=predictions + margin,
            confidence_level=confidence_level
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get XGBoost feature importance."""
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        
        return {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importance)
        }
    
    def get_booster(self):
        """Get underlying XGBoost booster for TreeSHAP."""
        return self.model


class QuantileXGBoostForecaster(XGBoostForecaster):
    """
    XGBoost with quantile regression for native uncertainty estimation.
    """
    
    def __init__(
        self,
        quantiles: Tuple[float, float, float] = (0.025, 0.5, 0.975),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.quantiles = quantiles
        self.models = {}
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'QuantileXGBoostForecaster':
        """Train separate models for each quantile."""
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        for q in self.quantiles:
            params = self.params.copy()
            params['objective'] = 'reg:quantileerror'
            params['quantile_alpha'] = q
            
            model = xgb.XGBRegressor(**params)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            
            self.models[q] = model
        
        # Use median model as primary
        self.model = self.models[self.quantiles[1]]
        self.is_fitted = True
        
        return self
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with quantile-based confidence intervals."""
        predictions = self.models[self.quantiles[1]].predict(X)
        lower = self.models[self.quantiles[0]].predict(X)
        upper = self.models[self.quantiles[2]].predict(X)
        
        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence_level=confidence_level
        )
