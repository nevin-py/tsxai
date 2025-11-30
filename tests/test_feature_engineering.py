"""
Tests for feature engineering utilities.
"""

import numpy as np
import pandas as pd
import pytest

from utils.feature_engineering import (
    TemporalFeatureEngineer,
    SequenceFeatureEngineer
)


class TestTemporalFeatureEngineer:
    """Test suite for TemporalFeatureEngineer."""
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        fe = TemporalFeatureEngineer(target_col='close', date_col='date')
        
        assert fe.target_col == 'close'
        assert fe.date_col == 'date'
        assert len(fe.lag_periods) > 0
        assert len(fe.rolling_windows) > 0
    
    def test_create_lag_features(self, sample_data):
        """Test lag feature creation."""
        fe = TemporalFeatureEngineer(
            target_col='close',
            lag_periods=[1, 2, 3]
        )
        
        df_result = fe.create_lag_features(sample_data)
        
        assert 'close_lag_1' in df_result.columns
        assert 'close_lag_2' in df_result.columns
        assert 'close_lag_3' in df_result.columns
        
        # Verify lag values are correct
        assert df_result['close_lag_1'].iloc[5] == sample_data['close'].iloc[4]
    
    def test_create_rolling_features(self, sample_data):
        """Test rolling feature creation."""
        fe = TemporalFeatureEngineer(
            target_col='close',
            rolling_windows=[7]
        )
        
        df_result = fe.create_rolling_features(sample_data)
        
        assert 'close_rolling_mean_7' in df_result.columns
        assert 'close_rolling_std_7' in df_result.columns
        assert 'close_rolling_min_7' in df_result.columns
        assert 'close_rolling_max_7' in df_result.columns
    
    def test_create_datetime_features(self, sample_data):
        """Test datetime feature creation."""
        fe = TemporalFeatureEngineer(target_col='close', date_col='date')
        
        df_result = fe.create_datetime_features(sample_data)
        
        assert 'day_of_week' in df_result.columns
        assert 'month' in df_result.columns
        assert 'is_weekend' in df_result.columns
        assert 'day_of_week_sin' in df_result.columns
        assert 'month_cos' in df_result.columns
        
        # Verify cyclical encoding bounds
        assert df_result['day_of_week_sin'].between(-1, 1).all()
        assert df_result['month_cos'].between(-1, 1).all()
    
    def test_create_return_features(self, sample_data):
        """Test return feature creation."""
        fe = TemporalFeatureEngineer(target_col='close')
        
        df_result = fe.create_return_features(sample_data)
        
        assert 'close_return_1' in df_result.columns
        assert 'close_return_7' in df_result.columns
        assert 'close_log_return_1' in df_result.columns
    
    def test_create_momentum_features(self, sample_data):
        """Test momentum feature creation."""
        fe = TemporalFeatureEngineer(target_col='close')
        
        df_result = fe.create_momentum_features(sample_data)
        
        assert 'close_roc_7' in df_result.columns
        assert 'ma_7' in df_result.columns
        assert 'ma_30' in df_result.columns
        assert 'ma_crossover' in df_result.columns
    
    def test_fit_transform(self, sample_data):
        """Test full feature engineering pipeline."""
        fe = TemporalFeatureEngineer(
            target_col='close',
            date_col='date',
            lag_periods=[1, 2, 3],
            rolling_windows=[7]
        )
        
        df_result, feature_names = fe.fit_transform(sample_data)
        
        # Should have removed rows with NaN from lagging
        assert len(df_result) < len(sample_data)
        
        # Should have created multiple features
        assert len(feature_names) > 10
        
        # No NaN in feature columns
        for col in feature_names:
            if col in df_result.columns:
                assert not df_result[col].isna().any(), f"NaN found in {col}"
    
    def test_no_data_leakage(self, sample_data):
        """Test that features don't leak future information."""
        fe = TemporalFeatureEngineer(
            target_col='close',
            lag_periods=[1],
            rolling_windows=[7]
        )
        
        df_result = fe.create_lag_features(sample_data)
        df_result = fe.create_rolling_features(df_result)
        
        # Lag features should be shifted (use past data only)
        # At index i, lag_1 should equal value at index i-1
        for i in range(1, 10):
            assert df_result['close_lag_1'].iloc[i] == sample_data['close'].iloc[i-1]
        
        # Rolling mean should be shifted (not include current value)
        # Rolling window uses shift(1) so it doesn't include current observation


class TestSequenceFeatureEngineer:
    """Test suite for SequenceFeatureEngineer."""
    
    def test_initialization(self):
        """Test sequence engineer initialization."""
        seq_fe = SequenceFeatureEngineer(sequence_length=30, forecast_horizon=1)
        
        assert seq_fe.sequence_length == 30
        assert seq_fe.forecast_horizon == 1
    
    def test_create_sequences(self):
        """Test sequence creation for LSTM."""
        seq_fe = SequenceFeatureEngineer(sequence_length=10, forecast_horizon=1)
        
        # Create sample data: 50 samples, 5 features
        data = np.random.randn(50, 5)
        
        X, y = seq_fe.create_sequences(data, target_idx=0)
        
        # Should create sequences
        assert X.shape[0] == 50 - 10 - 1 + 1  # n - seq_len - horizon + 1
        assert X.shape[1] == 10  # sequence length
        assert X.shape[2] == 5  # features
        assert len(y) == X.shape[0]
    
    def test_sequence_values(self):
        """Test that sequence values are correct."""
        seq_fe = SequenceFeatureEngineer(sequence_length=3, forecast_horizon=1)
        
        # Simple sequential data
        data = np.arange(20).reshape(-1, 1)  # [[0], [1], [2], ...]
        
        X, y = seq_fe.create_sequences(data, target_idx=0)
        
        # First sequence should be [0, 1, 2], target should be 3
        np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2])
        assert y[0] == 3
        
        # Second sequence should be [1, 2, 3], target should be 4
        np.testing.assert_array_equal(X[1].flatten(), [1, 2, 3])
        assert y[1] == 4
    
    def test_forecast_horizon(self):
        """Test different forecast horizons."""
        data = np.arange(30).reshape(-1, 1)
        
        # Horizon 1
        seq_fe_1 = SequenceFeatureEngineer(sequence_length=5, forecast_horizon=1)
        X1, y1 = seq_fe_1.create_sequences(data, target_idx=0)
        assert y1[0] == 5  # Predict next step
        
        # Horizon 3
        seq_fe_3 = SequenceFeatureEngineer(sequence_length=5, forecast_horizon=3)
        X3, y3 = seq_fe_3.create_sequences(data, target_idx=0)
        assert y3[0] == 7  # Predict 3 steps ahead
