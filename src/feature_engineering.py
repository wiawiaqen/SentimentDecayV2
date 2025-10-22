"""
Feature engineering module for sentiment decay analysis.
Includes technical indicators and sentiment-based features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# Technical indicators
try:
    import ta
except ImportError:
    ta = None
    logging.warning("ta library not installed. Install with: pip install ta")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Compute technical indicators for stock price data."""
    
    @staticmethod
    def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Add log returns for multiple periods.
        
        Args:
            df: DataFrame with 'Close' column
            periods: List of periods for computing returns
            
        Returns:
            DataFrame with return columns added
        """
        result_df = df.copy()
        
        for period in periods:
            result_df[f'return_{period}d'] = result_df.groupby('Ticker')['Close'].pct_change(period)
            result_df[f'log_return_{period}d'] = np.log1p(result_df[f'return_{period}d'])
        
        return result_df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add rolling volatility (std of returns).
        
        Args:
            df: DataFrame with 'Close' column
            window: Rolling window size
            
        Returns:
            DataFrame with volatility column added
        """
        result_df = df.copy()
        
        # Compute daily returns if not present
        if 'return_1d' not in result_df.columns:
            result_df['return_1d'] = result_df.groupby('Ticker')['Close'].pct_change()
        
        # Rolling volatility per ticker
        result_df['volatility'] = result_df.groupby('Ticker')['return_1d'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        return result_df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with 'Close' column
            window: RSI period
            
        Returns:
            DataFrame with RSI column added
        """
        result_df = df.copy()
        
        if ta is None:
            logger.warning("ta library not available. Skipping RSI calculation.")
            result_df['RSI'] = np.nan
            return result_df
        
        # Calculate RSI per ticker
        rsi_values = []
        
        for ticker in result_df['Ticker'].unique():
            ticker_mask = result_df['Ticker'] == ticker
            ticker_data = result_df[ticker_mask].copy()
            
            rsi = ta.momentum.RSIIndicator(
                close=ticker_data['Close'],
                window=window
            ).rsi()
            
            rsi_values.extend(rsi.values)
        
        result_df['RSI'] = rsi_values
        
        return result_df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicators.
        
        Args:
            df: DataFrame with 'Close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD columns added
        """
        result_df = df.copy()
        
        if ta is None:
            logger.warning("ta library not available. Skipping MACD calculation.")
            result_df['MACD'] = np.nan
            result_df['MACD_signal'] = np.nan
            result_df['MACD_diff'] = np.nan
            return result_df
        
        # Calculate MACD per ticker
        macd_values = []
        signal_values = []
        diff_values = []
        
        for ticker in result_df['Ticker'].unique():
            ticker_mask = result_df['Ticker'] == ticker
            ticker_data = result_df[ticker_mask].copy()
            
            macd_indicator = ta.trend.MACD(
                close=ticker_data['Close'],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal
            )
            
            macd_values.extend(macd_indicator.macd().values)
            signal_values.extend(macd_indicator.macd_signal().values)
            diff_values.extend(macd_indicator.macd_diff().values)
        
        result_df['MACD'] = macd_values
        result_df['MACD_signal'] = signal_values
        result_df['MACD_diff'] = diff_values
        
        return result_df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """
        Add Exponential Moving Averages.
        
        Args:
            df: DataFrame with 'Close' column
            windows: List of EMA periods
            
        Returns:
            DataFrame with EMA columns added
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f'EMA_{window}'] = result_df.groupby('Ticker')['Close'].transform(
                lambda x: x.ewm(span=window, adjust=False).mean()
            )
        
        return result_df
    
    @staticmethod
    def add_momentum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Add momentum indicator (rate of change).
        
        Args:
            df: DataFrame with 'Close' column
            window: Momentum period
            
        Returns:
            DataFrame with momentum column added
        """
        result_df = df.copy()
        
        result_df['Momentum'] = result_df.groupby('Ticker')['Close'].transform(
            lambda x: x.pct_change(periods=window)
        )
        
        return result_df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Computing technical indicators...")
        
        result_df = df.copy()
        
        # Sort by ticker and date
        result_df = result_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Add indicators
        result_df = TechnicalIndicators.add_returns(result_df)
        result_df = TechnicalIndicators.add_volatility(result_df)
        result_df = TechnicalIndicators.add_rsi(result_df)
        result_df = TechnicalIndicators.add_macd(result_df)
        result_df = TechnicalIndicators.add_ema(result_df)
        result_df = TechnicalIndicators.add_momentum(result_df)
        
        logger.info("Technical indicators computed")
        
        return result_df


class SentimentFeatures:
    """Create sentiment-based features."""
    
    @staticmethod
    def add_lagged_sentiment(
        df: pd.DataFrame,
        sentiment_cols: List[str],
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Add lagged sentiment features.
        
        Args:
            df: DataFrame with sentiment columns
            sentiment_cols: List of sentiment column names
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged sentiment features
        """
        result_df = df.copy()
        
        for col in sentiment_cols:
            if col not in result_df.columns:
                continue
            
            for lag in lags:
                result_df[f'{col}_lag{lag}'] = result_df.groupby('Ticker')[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def add_sentiment_momentum(
        df: pd.DataFrame,
        sentiment_col: str = 'sent_mean',
        window: int = 3
    ) -> pd.DataFrame:
        """
        Add sentiment momentum (change over time).
        
        Args:
            df: DataFrame with sentiment column
            sentiment_col: Name of sentiment column
            window: Period for computing momentum
            
        Returns:
            DataFrame with sentiment momentum
        """
        result_df = df.copy()
        
        if sentiment_col in result_df.columns:
            result_df['sent_momentum'] = result_df.groupby('Ticker')[sentiment_col].transform(
                lambda x: x.diff(window)
            )
        
        return result_df
    
    @staticmethod
    def add_sentiment_volatility(
        df: pd.DataFrame,
        sentiment_col: str = 'sent_mean',
        window: int = 5
    ) -> pd.DataFrame:
        """
        Add rolling sentiment volatility.
        
        Args:
            df: DataFrame with sentiment column
            sentiment_col: Name of sentiment column
            window: Rolling window size
            
        Returns:
            DataFrame with sentiment volatility
        """
        result_df = df.copy()
        
        if sentiment_col in result_df.columns:
            result_df['sent_volatility'] = result_df.groupby('Ticker')[sentiment_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return result_df


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def merge_market_and_sentiment(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge market data with sentiment features.
        
        Args:
            market_df: DataFrame with market data and target
            sentiment_df: DataFrame with sentiment features
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging market data with sentiment features...")
        
        # Ensure date columns are datetime
        market_df['Date'] = pd.to_datetime(market_df['Date'])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        
        # Merge on ticker and date
        merged_df = market_df.merge(
            sentiment_df,
            on=['Ticker', 'Date'],
            how='left'
        )
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Rows with sentiment: {merged_df['sent_mean'].notna().sum()}" if 'sent_mean' in merged_df.columns else "No sentiment column found")
        
        return merged_df
    
    def create_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create baseline features (technical indicators only).
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with baseline features
        """
        logger.info("Creating baseline features...")
        
        result_df = TechnicalIndicators.add_all_indicators(df)
        
        return result_df
    
    def create_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_type: str = 'flat'
    ) -> pd.DataFrame:
        """
        Create sentiment-enhanced features.
        
        Args:
            df: DataFrame with market data and sentiment
            sentiment_type: Type of sentiment ('flat' or 'decay')
            
        Returns:
            DataFrame with sentiment features
        """
        logger.info(f"Creating {sentiment_type} sentiment features...")
        
        result_df = df.copy()
        
        # Identify sentiment columns
        if sentiment_type == 'flat':
            sentiment_cols = ['sent_mean', 'sent_std', 'sent_min', 'sent_max', 'sent_count']
        elif sentiment_type == 'decay':
            sentiment_cols = ['sent_decay_mean', 'sent_decay_var', 'sentiment_memory']
        else:
            raise ValueError(f"Unknown sentiment type: {sentiment_type}")
        
        # Filter existing columns
        sentiment_cols = [col for col in sentiment_cols if col in result_df.columns]
        
        if not sentiment_cols:
            logger.warning(f"No {sentiment_type} sentiment columns found")
            return result_df
        
        # Add lagged sentiment
        result_df = SentimentFeatures.add_lagged_sentiment(result_df, sentiment_cols)
        
        # Add sentiment momentum and volatility
        for col in sentiment_cols:
            if 'mean' in col:
                result_df = SentimentFeatures.add_sentiment_momentum(result_df, col)
                result_df = SentimentFeatures.add_sentiment_volatility(result_df, col)
        
        return result_df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        sentiment_type: str = 'flat'
    ) -> pd.DataFrame:
        """
        Prepare all features for modeling.
        
        Args:
            df: Input DataFrame
            include_sentiment: Whether to include sentiment features
            sentiment_type: Type of sentiment ('flat' or 'decay')
            
        Returns:
            DataFrame with all features
        """
        logger.info("Preparing features for modeling...")
        
        result_df = df.copy()
        
        # Add technical indicators
        result_df = self.create_baseline_features(result_df)
        
        # Add sentiment features if requested
        if include_sentiment:
            result_df = self.create_sentiment_features(result_df, sentiment_type)
        
        # Fill missing values
        result_df = self.handle_missing_values(result_df)
        
        logger.info(f"Final feature set shape: {result_df.shape}")
        
        return result_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        result_df = df.copy()
        
        # Forward fill sentiment features (use previous day's sentiment if missing)
        sentiment_cols = [col for col in result_df.columns if 'sent' in col.lower()]
        for col in sentiment_cols:
            result_df[col] = result_df.groupby('Ticker')[col].fillna(method='ffill')
        
        # Fill remaining NaN with 0 for sentiment (no news = neutral)
        result_df[sentiment_cols] = result_df[sentiment_cols].fillna(0)
        
        # Forward fill technical indicators
        tech_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'EMA_12', 'EMA_26', 
                     'Momentum', 'volatility', 'return_1d', 'return_5d', 'return_10d']
        existing_tech_cols = [col for col in tech_cols if col in result_df.columns]
        
        for col in existing_tech_cols:
            result_df[col] = result_df.groupby('Ticker')[col].fillna(method='ffill')
        
        # Fill any remaining with 0
        result_df[existing_tech_cols] = result_df[existing_tech_cols].fillna(0)
        
        return result_df
    
    def get_feature_names(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True
    ) -> List[str]:
        """
        Get list of feature column names for modeling.
        
        Args:
            df: DataFrame with features
            include_sentiment: Whether sentiment features are included
            
        Returns:
            List of feature column names
        """
        # Exclude non-feature columns
        exclude_cols = [
            'Date', 'Ticker', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Headlines', 'Time', 'Description', 'Date_only', 'Next_Close',
            'negative', 'neutral', 'positive', 'sentiment_score', 'decay_weight',
            'normalized_time', 'combined_text'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter out sentiment columns if not included
        if not include_sentiment:
            feature_cols = [col for col in feature_cols if 'sent' not in col.lower()]
        
        return feature_cols
