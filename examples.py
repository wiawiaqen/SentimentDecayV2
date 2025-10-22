"""
Simple example script demonstrating individual components.
Run this to test each module separately before running the full pipeline.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_data_loading():
    """Example: Load and align data."""
    logger.info("=" * 60)
    logger.info("Example 1: Data Loading")
    logger.info("=" * 60)
    
    from data_loader import DataLoader
    
    # Initialize loader
    loader = DataLoader(
        aapl_path="AAPL_cleaned.csv",
        msft_path="MSFT_cleaned.csv",
        reuters_path="reuters_headlines.csv"
    )
    
    # Load market data
    aapl_data = loader.load_market_data("AAPL")
    logger.info(f"AAPL data shape: {aapl_data.shape}")
    logger.info(f"AAPL columns: {aapl_data.columns.tolist()}")
    logger.info(f"AAPL date range: {aapl_data['Date'].min()} to {aapl_data['Date'].max()}")
    
    # Load news
    news_data = loader.load_news_headlines()
    logger.info(f"News data shape: {news_data.shape}")
    
    return aapl_data, news_data


def example_technical_indicators():
    """Example: Compute technical indicators."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Technical Indicators")
    logger.info("=" * 60)
    
    from feature_engineering import TechnicalIndicators
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Ticker': 'AAPL'
    })
    
    # Add indicators
    df = TechnicalIndicators.add_returns(df)
    df = TechnicalIndicators.add_volatility(df)
    df = TechnicalIndicators.add_ema(df)
    
    logger.info(f"Features added: {[col for col in df.columns if col not in ['Date', 'Close', 'Ticker']]}")
    logger.info(f"Sample data:\n{df[['Date', 'Close', 'return_1d', 'volatility', 'EMA_12']].head()}")
    
    return df


def example_decay_functions():
    """Example: Apply decay functions."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Decay Functions")
    logger.info("=" * 60)
    
    from decay_functions import ExponentialDecay, HalfLifeDecay, LinearDecay
    
    # Sample normalized times [0, 1]
    times = np.linspace(0, 1, 10)
    
    # Exponential decay
    exp_decay = ExponentialDecay(lambda_param=1.0)
    exp_weights = exp_decay.compute_weights(times)
    logger.info(f"Exponential decay weights (λ=1.0): {exp_weights}")
    
    # Linear decay
    lin_decay = LinearDecay(beta=0.5)
    lin_weights = lin_decay.compute_weights(times)
    logger.info(f"Linear decay weights (β=0.5): {lin_weights}")
    
    # Half-life decay
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    sentiment = np.random.randn(10)
    
    half_decay = HalfLifeDecay(half_life_days=3)
    sentiment_series = pd.Series(sentiment)
    date_series = pd.Series(dates)
    
    memory = half_decay.compute_memory(sentiment_series, date_series)
    logger.info(f"Half-life memory (h=3): {memory.values}")


def example_model_training():
    """Example: Train a simple model."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Model Training")
    logger.info("=" * 60)
    
    from models import BaseModel
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y)
    
    # Train model
    model = BaseModel(model_type='logistic_regression')
    model.fit(X_df, y_series)
    
    # Predict
    predictions = model.predict(X_df[:5])
    probabilities = model.predict_proba(X_df[:5])
    
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Probabilities:\n{probabilities}")
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        logger.info(f"Feature importance: {importance}")


def example_evaluation_metrics():
    """Example: Compute evaluation metrics."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Evaluation Metrics")
    logger.info("=" * 60)
    
    from evaluation import ClassificationMetrics, TradingMetrics
    
    # Sample predictions
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.3, 0.8, 0.4, 0.2, 0.9, 0.7, 0.1, 0.6, 0.8, 0.3])
    
    # Classification metrics
    class_metrics = ClassificationMetrics.compute_metrics(y_true, y_pred, y_pred_proba)
    logger.info(f"Classification metrics: {class_metrics}")
    
    # Trading metrics
    actual_returns = np.random.randn(10) * 0.01  # 1% daily returns
    trading_metrics = TradingMetrics.compute_trading_metrics(y_pred, actual_returns)
    logger.info(f"Trading metrics: {trading_metrics}")


def example_complete_workflow():
    """Example: Simplified end-to-end workflow."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Complete Workflow (Simplified)")
    logger.info("=" * 60)
    
    # This demonstrates the flow without actual data
    logger.info("1. Load data → DataLoader")
    logger.info("2. Extract sentiment → SentimentExtractor (FinBERT)")
    logger.info("3. Apply decay → ExponentialDecay/HalfLifeDecay/LinearDecay")
    logger.info("4. Engineer features → FeatureEngineer (technical + sentiment)")
    logger.info("5. Train models → ModelTrainer (baseline, flat, decay)")
    logger.info("6. Grid search → DecayGridSearch (optimize parameters)")
    logger.info("7. Evaluate → ModelEvaluator (metrics, SHAP)")
    logger.info("8. Report → ReportGenerator (plots, tables)")
    logger.info("\nRun src/main.py for the full pipeline!")


def main():
    """Run all examples."""
    logger.info("SENTIMENT DECAY ANALYSIS - EXAMPLES")
    logger.info("=" * 60)
    
    try:
        # Check if data files exist
        if Path("AAPL_cleaned.csv").exists():
            example_data_loading()
        else:
            logger.warning("Data files not found. Skipping data loading example.")
        
        example_technical_indicators()
        example_decay_functions()
        example_model_training()
        example_evaluation_metrics()
        example_complete_workflow()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed!")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
