"""
Feature engineering utilities for time series forecasting.
Creates temporal features while respecting causal ordering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler


class TemporalFeatureEngineer:
    """
    Creates temporal features for time series forecasting.
    Critical: All features must only use past information to prevent leakage.
    """
    
    def __init__(self, 
                 target_col: str,
                 date_col: str = 'date',
                 lag_periods: List[int] = None,
                 rolling_windows: List[int] = None):
        self.target_col = target_col
        self.date_col = date_col
        self.lag_periods = lag_periods or [1, 2, 3, 5, 7, 14, 21, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of the target variable."""
        df = df.copy()
        
        for lag in self.lag_periods:
            col_name = f'{self.target_col}_lag_{lag}'
            df[col_name] = df[self.target_col].shift(lag)
            self.feature_names.append(col_name)
            
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        df = df.copy()
        
        for window in self.rolling_windows:
            # Rolling mean (shift by 1 to prevent leakage)
            col_mean = f'{self.target_col}_rolling_mean_{window}'
            df[col_mean] = df[self.target_col].shift(1).rolling(window=window).mean()
            self.feature_names.append(col_mean)
            
            # Rolling std
            col_std = f'{self.target_col}_rolling_std_{window}'
            df[col_std] = df[self.target_col].shift(1).rolling(window=window).std()
            self.feature_names.append(col_std)
            
            # Rolling min
            col_min = f'{self.target_col}_rolling_min_{window}'
            df[col_min] = df[self.target_col].shift(1).rolling(window=window).min()
            self.feature_names.append(col_min)
            
            # Rolling max
            col_max = f'{self.target_col}_rolling_max_{window}'
            df[col_max] = df[self.target_col].shift(1).rolling(window=window).max()
            self.feature_names.append(col_max)
            
        return df
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime components as features."""
        df = df.copy()
        
        if self.date_col in df.columns:
            dt = pd.to_datetime(df[self.date_col])
            
            # Basic datetime features
            df['day_of_week'] = dt.dt.dayofweek
            df['day_of_month'] = dt.dt.day
            df['day_of_year'] = dt.dt.dayofyear
            df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
            df['month'] = dt.dt.month
            df['quarter'] = dt.dt.quarter
            df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for periodic features
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            datetime_features = [
                'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year',
                'month', 'quarter', 'is_weekend',
                'day_of_week_sin', 'day_of_week_cos',
                'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos'
            ]
            self.feature_names.extend(datetime_features)
            
        return df
    
    def create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create return/change features (useful for financial data)."""
        df = df.copy()
        
        # Simple returns
        df[f'{self.target_col}_return_1'] = df[self.target_col].pct_change(1)
        df[f'{self.target_col}_return_7'] = df[self.target_col].pct_change(7)
        df[f'{self.target_col}_return_30'] = df[self.target_col].pct_change(30)
        
        # Log returns
        df[f'{self.target_col}_log_return_1'] = np.log(df[self.target_col] / df[self.target_col].shift(1))
        
        return_features = [
            f'{self.target_col}_return_1',
            f'{self.target_col}_return_7', 
            f'{self.target_col}_return_30',
            f'{self.target_col}_log_return_1'
        ]
        self.feature_names.extend(return_features)
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum/trend indicators."""
        df = df.copy()
        
        # Rate of change
        for period in [7, 14, 30]:
            col_name = f'{self.target_col}_roc_{period}'
            df[col_name] = (df[self.target_col] - df[self.target_col].shift(period)) / df[self.target_col].shift(period) * 100
            self.feature_names.append(col_name)
        
        # Moving average crossover signals
        df['ma_7'] = df[self.target_col].shift(1).rolling(7).mean()
        df['ma_30'] = df[self.target_col].shift(1).rolling(30).mean()
        df['ma_crossover'] = (df['ma_7'] > df['ma_30']).astype(int)
        df['ma_diff'] = df['ma_7'] - df['ma_30']
        
        self.feature_names.extend(['ma_7', 'ma_30', 'ma_crossover', 'ma_diff'])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, include_returns: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Apply all feature engineering transformations."""
        self.feature_names = []  # Reset
        
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_datetime_features(df)
        df = self.create_momentum_features(df)
        
        if include_returns:
            df = self.create_return_features(df)
        
        # Drop rows with NaN from lagging
        max_lag = max(self.lag_periods + self.rolling_windows)
        df = df.iloc[max_lag:].reset_index(drop=True)
        
        return df, self.feature_names
    
    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (features) and y (target) matrices."""
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        return X, y


class SequenceFeatureEngineer:
    """
    Creates sequence features for LSTM/RNN models.
    Transforms data into 3D arrays: (samples, timesteps, features)
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 forecast_horizon: int = 1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
    def create_sequences(self, 
                        data: np.ndarray, 
                        target_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training sequence models.
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            target_idx: Index of the target column
            
        Returns:
            X: 3D array of shape (n_sequences, sequence_length, n_features)
            y: 1D array of targets
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length])
            # Target (future value)
            y.append(data[i + self.sequence_length + self.forecast_horizon - 1, target_idx])
            
        return np.array(X), np.array(y)


def prepare_dataset_for_modeling(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = 'date',
    model_type: str = 'xgboost'
) -> Dict:
    """
    Full pipeline to prepare dataset for different model types.
    
    Returns dict with train/val/test splits and feature info.
    """
    from utils.data_loader import TimeSeriesDataLoader
    
    # Feature engineering
    fe = TemporalFeatureEngineer(target_col=target_col, date_col=date_col)
    df_features, feature_names = fe.fit_transform(df)
    
    # Temporal split
    loader = TimeSeriesDataLoader()
    train_df, val_df, test_df = loader.temporal_split(df_features, date_col)
    
    # Get feature matrix
    feature_cols = [col for col in feature_names if col in df_features.columns]
    
    if model_type in ['xgboost', 'prophet']:
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols,
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df
        }
    
    elif model_type == 'lstm':
        # Create sequences
        seq_fe = SequenceFeatureEngineer(sequence_length=30, forecast_horizon=1)
        
        train_data = train_df[feature_cols + [target_col]].values
        val_data = val_df[feature_cols + [target_col]].values
        test_data = test_df[feature_cols + [target_col]].values
        
        X_train, y_train = seq_fe.create_sequences(train_data, target_idx=-1)
        X_val, y_val = seq_fe.create_sequences(val_data, target_idx=-1)
        X_test, y_test = seq_fe.create_sequences(test_data, target_idx=-1)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols,
            'sequence_length': seq_fe.sequence_length,
            'train_df': train_df, 'val_df': val_df, 'test_df': test_df
        }


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import get_dataset
    
    df = get_dataset('stock')
    
    fe = TemporalFeatureEngineer(target_col='close', date_col='date')
    df_features, feature_names = fe.fit_transform(df)
    
    print(f"Original shape: {df.shape}")
    print(f"After feature engineering: {df_features.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nFeature names:\n{feature_names}")
