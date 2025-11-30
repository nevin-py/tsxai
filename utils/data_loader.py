"""
Data loading utilities for time series datasets.
Handles stock prices, weather data, and energy consumption.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataLoader:
    """Base class for loading time series data with proper temporal splits."""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
    def temporal_split(self, df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally to prevent data leakage.
        Critical: Never shuffle time series data!
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df


class StockDataLoader(TimeSeriesDataLoader):
    """Load stock market data (synthetic for demo, real via yfinance)."""
    
    def generate_synthetic_stock_data(
        self, 
        n_days: int = 1000, 
        start_price: float = 100.0,
        ticker: str = 'SYNTH'
    ) -> pd.DataFrame:
        """Generate synthetic stock data with realistic patterns."""
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Generate price with trend, seasonality, and noise
        trend = np.linspace(0, 50, n_days)
        seasonality = 10 * np.sin(np.linspace(0, 8 * np.pi, n_days))
        noise = np.random.randn(n_days) * 5
        
        # Random walk component
        random_walk = np.cumsum(np.random.randn(n_days) * 0.5)
        
        close = start_price + trend + seasonality + noise + random_walk
        close = np.maximum(close, 1)  # Prevent negative prices
        
        # Generate OHLCV
        high = close * (1 + np.abs(np.random.randn(n_days) * 0.02))
        low = close * (1 - np.abs(np.random.randn(n_days) * 0.02))
        open_price = low + (high - low) * np.random.rand(n_days)
        volume = np.random.lognormal(15, 1, n_days).astype(int)
        
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return df
    
    def load_stock_data(self, ticker: str = 'SPY', use_synthetic: bool = True) -> pd.DataFrame:
        """Load stock data - synthetic or real."""
        if use_synthetic:
            return self.generate_synthetic_stock_data(ticker=ticker)
        
        try:
            import yfinance as yf
            data = yf.download(ticker, period='5y', progress=False)
            df = data.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df['ticker'] = ticker
            return df
        except Exception as e:
            print(f"Failed to load real data: {e}. Using synthetic data.")
            return self.generate_synthetic_stock_data(ticker=ticker)


class WeatherDataLoader(TimeSeriesDataLoader):
    """Load weather data (synthetic for demo)."""
    
    def generate_synthetic_weather_data(self, n_days: int = 1000) -> pd.DataFrame:
        """Generate synthetic weather data with seasonal patterns."""
        np.random.seed(43)
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Day of year for seasonality
        day_of_year = dates.dayofyear
        
        # Temperature with seasonal pattern
        base_temp = 15  # Base temperature in Celsius
        seasonal_amplitude = 15
        temp = base_temp + seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp += np.random.randn(n_days) * 3  # Daily variation
        
        # Humidity (inverse correlation with temperature)
        humidity = 70 - 0.5 * (temp - base_temp) + np.random.randn(n_days) * 10
        humidity = np.clip(humidity, 20, 100)
        
        # Precipitation (more in certain seasons)
        precip_base = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 20) / 365)
        precipitation = np.maximum(0, precip_base + np.random.exponential(3, n_days) - 5)
        precipitation *= (np.random.rand(n_days) > 0.6).astype(float)  # Many dry days
        
        # Wind speed
        wind_speed = np.random.weibull(2, n_days) * 10
        
        # Pressure
        pressure = 1013 + np.random.randn(n_days) * 10
        
        df = pd.DataFrame({
            'date': dates,
            'temperature': temp,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'pressure': pressure
        })
        
        return df


class EnergyDataLoader(TimeSeriesDataLoader):
    """Load energy consumption data (synthetic for demo)."""
    
    def generate_synthetic_energy_data(self, n_hours: int = 24000) -> pd.DataFrame:
        """Generate synthetic hourly energy consumption data."""
        np.random.seed(44)
        
        dates = pd.date_range(end=datetime.now(), periods=n_hours, freq='H')
        
        # Base load
        base_load = 1000  # MW
        
        # Daily pattern (peak during day, low at night)
        hour_of_day = dates.hour
        daily_pattern = 200 * np.sin(np.pi * (hour_of_day - 6) / 12)
        daily_pattern = np.where(hour_of_day < 6, -150, daily_pattern)
        daily_pattern = np.where(hour_of_day > 22, -100, daily_pattern)
        
        # Weekly pattern (lower on weekends)
        day_of_week = dates.dayofweek
        weekend_effect = np.where(day_of_week >= 5, -200, 0)
        
        # Seasonal pattern
        day_of_year = dates.dayofyear
        seasonal = 150 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # Peak in summer/winter
        
        # Temperature effect (synthetic temperature)
        temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n_hours) * 3
        temp_effect = 20 * np.abs(temp - 20)  # More energy when hot or cold
        
        # Combine
        consumption = base_load + daily_pattern + weekend_effect + seasonal + temp_effect
        consumption += np.random.randn(n_hours) * 50  # Noise
        consumption = np.maximum(consumption, 200)  # Minimum load
        
        df = pd.DataFrame({
            'date': dates,
            'consumption': consumption,
            'temperature': temp,
            'hour': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': (day_of_week >= 5).astype(int)
        })
        
        return df


def get_dataset(name: str, **kwargs) -> pd.DataFrame:
    """Factory function to get datasets by name."""
    loaders = {
        'stock': StockDataLoader(),
        'weather': WeatherDataLoader(),
        'energy': EnergyDataLoader()
    }
    
    if name == 'stock':
        return loaders['stock'].load_stock_data(**kwargs)
    elif name == 'weather':
        return loaders['weather'].generate_synthetic_weather_data(**kwargs)
    elif name == 'energy':
        return loaders['energy'].generate_synthetic_energy_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(loaders.keys())}")


if __name__ == "__main__":
    # Test data loaders
    for dataset_name in ['stock', 'weather', 'energy']:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_name} data loader...")
        df = get_dataset(dataset_name)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(df.head())
