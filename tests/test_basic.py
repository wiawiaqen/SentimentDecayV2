"""
Unit tests for sentiment decay analysis modules.
Run with: pytest tests/test_basic.py
"""

import sys
sys.path.append('src')

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_extract_ticker_from_headline(self):
        from data_loader import DataLoader
        
        loader = DataLoader("", "", "")
        
        # Test AAPL
        assert loader.extract_ticker_from_headline("Apple announces new iPhone") == "AAPL"
        assert loader.extract_ticker_from_headline("AAPL stock rises") == "AAPL"
        
        # Test MSFT
        assert loader.extract_ticker_from_headline("Microsoft Azure growth") == "MSFT"
        assert loader.extract_ticker_from_headline("MSFT earnings beat") == "MSFT"
        
        # Test no match
        assert loader.extract_ticker_from_headline("Market news today") is None


class TestDecayFunctions:
    """Test decay function implementations."""
    
    def test_exponential_decay(self):
        from decay_functions import ExponentialDecay
        
        decay = ExponentialDecay(lambda_param=1.0)
        times = np.array([0.0, 0.5, 1.0])
        weights = decay.compute_weights(times)
        
        # Check properties
        assert len(weights) == 3
        assert all(weights >= 0)
        assert all(weights <= 1)
        # Most recent should have highest weight (time=1.0)
        assert weights[2] > weights[0]
    
    def test_linear_decay(self):
        from decay_functions import LinearDecay
        
        decay = LinearDecay(beta=0.5)
        times = np.array([0.0, 0.5, 1.0])
        weights = decay.compute_weights(times)
        
        assert len(weights) == 3
        assert all(weights >= 0)
        assert all(weights <= 1)
    
    def test_half_life_decay(self):
        from decay_functions import HalfLifeDecay
        
        decay = HalfLifeDecay(half_life_days=3.0)
        
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        sentiment = pd.Series(np.ones(10))
        
        memory = decay.compute_memory(sentiment, pd.Series(dates))
        
        assert len(memory) == 10
        assert memory.iloc[0] == 1.0


class TestTechnicalIndicators:
    """Test technical indicator calculations."""
    
    def test_add_returns(self):
        from feature_engineering import TechnicalIndicators
        
        df = pd.DataFrame({
            'Close': [100, 101, 99, 102, 105],
            'Ticker': ['AAPL'] * 5
        })
        
        result = TechnicalIndicators.add_returns(df, periods=[1])
        
        assert 'return_1d' in result.columns
        assert 'log_return_1d' in result.columns
        assert not result['return_1d'].iloc[0] == result['return_1d'].iloc[0]  # NaN check
    
    def test_add_ema(self):
        from feature_engineering import TechnicalIndicators
        
        df = pd.DataFrame({
            'Close': np.arange(1, 101),
            'Ticker': ['AAPL'] * 100
        })
        
        result = TechnicalIndicators.add_ema(df, windows=[12, 26])
        
        assert 'EMA_12' in result.columns
        assert 'EMA_26' in result.columns


class TestModels:
    """Test model training functionality."""
    
    def test_base_model_logistic(self):
        from models import BaseModel
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = BaseModel(model_type='logistic_regression')
        model.fit(X, y, feature_names=[f'f{i}' for i in range(5)])
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
    
    def test_rolling_time_series_split(self):
        from models import RollingTimeSeriesSplit
        
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        df = pd.DataFrame({'Date': dates})
        
        splitter = RollingTimeSeriesSplit(train_window=252, test_window=21, n_splits=3)
        splits = splitter.split(df)
        
        assert len(splits) > 0
        assert len(splits) <= 3


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        from evaluation import ClassificationMetrics
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])
        
        metrics = ClassificationMetrics.compute_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_sharpe_ratio(self):
        from evaluation import TradingMetrics
        
        returns = np.array([0.01, -0.005, 0.02, 0.01, -0.01])
        sharpe = TradingMetrics.compute_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
    
    def test_trading_metrics(self):
        from evaluation import TradingMetrics
        
        y_pred = np.array([1, 1, 0, 1, 0])
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        
        metrics = TradingMetrics.compute_trading_metrics(y_pred, returns)
        
        assert 'sharpe_ratio' in metrics
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
