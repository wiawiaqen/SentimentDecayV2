"""
Sentiment Decay Analysis Package
"""

__version__ = "1.0.0"
__author__ = "Sentiment Decay Research Team"

from .data_loader import DataLoader, load_and_prepare_data
from .sentiment_extractor import SentimentExtractor, SentimentAggregator
from .decay_functions import (
    ExponentialDecay,
    HalfLifeDecay,
    LinearDecay,
    DecayAggregator,
    apply_decay_function
)
from .feature_engineering import FeatureEngineer, TechnicalIndicators
from .models import BaseModel, ModelTrainer
from .evaluation import ClassificationMetrics, TradingMetrics, ModelEvaluator
from .reporting import ReportGenerator

__all__ = [
    'DataLoader',
    'load_and_prepare_data',
    'SentimentExtractor',
    'SentimentAggregator',
    'ExponentialDecay',
    'HalfLifeDecay',
    'LinearDecay',
    'DecayAggregator',
    'apply_decay_function',
    'FeatureEngineer',
    'TechnicalIndicators',
    'BaseModel',
    'ModelTrainer',
    'ClassificationMetrics',
    'TradingMetrics',
    'ModelEvaluator',
    'ReportGenerator'
]
