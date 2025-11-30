"""
LSTM model for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from models.base_model import BaseTimeSeriesModel, ForecastResult


class LSTMForecaster(BaseTimeSeriesModel):
    """
    LSTM-based time series forecaster using PyTorch.
    
    Best for capturing long-term dependencies in time series.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'cpu'
    ):
        super().__init__(model_name="lstm")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.model = None
        self.scaler = None
        
    def _build_model(self, input_size: int):
        """Build LSTM model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out.squeeze()
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> 'LSTMForecaster':
        """
        Train LSTM model.
        
        X_train expected shape: (n_samples, sequence_length, n_features)
        or (n_samples, n_features) which will be reshaped
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("PyTorch not available, using fallback")
            self._mean = np.mean(y_train)
            self._std = np.std(y_train)
            self.is_fitted = True
            return self
        
        # Handle input shape
        if len(X_train.shape) == 2:
            # Reshape to (samples, seq_len=1, features)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        self.input_size = X_train.shape[2]
        self.feature_names = feature_names or [f'feature_{i}' for i in range(self.input_size)]
        
        # Build model
        self.model = self._build_model(self.input_size)
        self.model.to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:  # Early stopping
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        # Store residual std for confidence intervals
        self.model.eval()
        with torch.no_grad():
            train_preds = self.model(X_tensor).cpu().numpy()
        self.residual_std = np.std(y_train - train_preds)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model is None:
            return np.full(len(X), self._mean)
        
        try:
            import torch
        except ImportError:
            return np.full(len(X), self._mean)
        
        # Handle input shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with confidence intervals."""
        predictions = self.predict(X)
        
        # Simple confidence interval based on training residuals
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        margin = z * self.residual_std if hasattr(self, 'residual_std') else predictions * 0.1
        
        return ForecastResult(
            predictions=predictions,
            confidence_lower=predictions - margin,
            confidence_upper=predictions + margin,
            confidence_level=confidence_level
        )


class SimpleLSTMWrapper(BaseTimeSeriesModel):
    """
    Simplified LSTM wrapper for the dashboard.
    Falls back to simple model if PyTorch not available.
    """
    
    def __init__(self, **kwargs):
        super().__init__(model_name="simple_lstm")
        self.lstm = LSTMForecaster(**kwargs)
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'SimpleLSTMWrapper':
        """Fit the LSTM model."""
        try:
            self.lstm.fit(X_train, y_train, X_val, y_val, verbose=False, **kwargs)
        except Exception as e:
            print(f"LSTM training failed: {e}. Using fallback.")
            self._mean = np.mean(y_train)
            self._std = np.std(y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        try:
            return self.lstm.predict(X)
        except:
            return np.full(len(X), self._mean if hasattr(self, '_mean') else 0)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate predictions with confidence intervals."""
        try:
            return self.lstm.predict_with_confidence(X, confidence_level)
        except:
            preds = np.full(len(X), self._mean if hasattr(self, '_mean') else 0)
            std = self._std if hasattr(self, '_std') else 1
            margin = 1.96 * std
            return ForecastResult(
                predictions=preds,
                confidence_lower=preds - margin,
                confidence_upper=preds + margin,
                confidence_level=confidence_level
            )
