"""
Sentiment decay functions module.
Implements exponential, half-life, and linear decay mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecayFunction:
    """Base class for decay functions."""
    
    def __init__(self, name: str):
        """Initialize decay function with a name."""
        self.name = name
    
    def compute_weights(self, timestamps: np.ndarray, reference_time: float) -> np.ndarray:
        """
        Compute decay weights for timestamps relative to a reference time.
        
        Args:
            timestamps: Array of timestamps (normalized)
            reference_time: Reference time point
            
        Returns:
            Array of decay weights
        """
        raise NotImplementedError("Subclasses must implement compute_weights")


class ExponentialDecay(DecayFunction):
    """
    Exponential intraday decay: w_j = exp(-λ * t_j)
    where t_j ∈ [0,1] is normalized time of day.
    """
    
    def __init__(self, lambda_param: float):
        """
        Initialize exponential decay function.
        
        Args:
            lambda_param: Decay rate parameter (λ)
        """
        super().__init__(f"exponential_lambda_{lambda_param}")
        self.lambda_param = lambda_param
    
    def compute_weights(self, normalized_times: np.ndarray) -> np.ndarray:
        """
        Compute exponential decay weights.
        
        Args:
            normalized_times: Time normalized to [0, 1] within the day
            
        Returns:
            Decay weights
        """
        # Invert times so more recent = lower value, then apply decay
        # If t=0 is start of day and t=1 is end, we want end-of-day news to have higher weight
        # So we use (1 - normalized_times) to invert
        elapsed = 1.0 - normalized_times
        weights = np.exp(-self.lambda_param * elapsed)
        return weights
    
    def apply_to_dataframe(self, df: pd.DataFrame, time_col: str = 'normalized_time') -> pd.DataFrame:
        """
        Apply exponential decay to a DataFrame with timestamps.
        
        Args:
            df: DataFrame with sentiment scores and normalized times
            time_col: Column name containing normalized times [0, 1]
            
        Returns:
            DataFrame with 'decay_weight' column added
        """
        result_df = df.copy()
        result_df['decay_weight'] = self.compute_weights(result_df[time_col].values)
        return result_df


class HalfLifeDecay(DecayFunction):
    """
    Half-life decay (cross-day): M_t = α * S_t + (1 - α) * M_{t-1}
    where α = 1 - exp(ln(0.5) / h), h = half-life in days.
    """
    
    def __init__(self, half_life_days: float):
        """
        Initialize half-life decay function.
        
        Args:
            half_life_days: Half-life period in days
        """
        super().__init__(f"halflife_{half_life_days}d")
        self.half_life_days = half_life_days
        self.alpha = 1 - np.exp(np.log(0.5) / half_life_days)
    
    def compute_memory(
        self, 
        sentiment_series: pd.Series, 
        dates: pd.Series
    ) -> pd.Series:
        """
        Compute exponentially weighted moving average (memory) across days.
        
        Args:
            sentiment_series: Daily sentiment scores
            dates: Corresponding dates (must be sorted)
            
        Returns:
            Series with memory values
        """
        memory = np.zeros(len(sentiment_series))
        memory[0] = sentiment_series.iloc[0]
        
        for i in range(1, len(sentiment_series)):
            # Days since last observation
            days_elapsed = (dates.iloc[i] - dates.iloc[i-1]).days
            
            # Decay factor based on elapsed days
            decay_factor = (1 - self.alpha) ** days_elapsed
            
            # Update memory
            memory[i] = self.alpha * sentiment_series.iloc[i] + decay_factor * memory[i-1]
        
        return pd.Series(memory, index=sentiment_series.index)
    
    def apply_to_dataframe(
        self, 
        df: pd.DataFrame, 
        sentiment_col: str = 'sent_mean',
        date_col: str = 'Date',
        ticker_col: str = 'Ticker'
    ) -> pd.DataFrame:
        """
        Apply half-life decay to create memory feature.
        
        Args:
            df: DataFrame with daily sentiment scores
            sentiment_col: Column name with sentiment scores
            date_col: Column name with dates
            ticker_col: Column name with ticker symbols
            
        Returns:
            DataFrame with 'sentiment_memory' column added
        """
        result_df = df.copy()
        
        # Sort by ticker and date
        result_df = result_df.sort_values([ticker_col, date_col]).reset_index(drop=True)
        
        # Apply memory computation per ticker
        memory_values = []
        
        for ticker in result_df[ticker_col].unique():
            ticker_mask = result_df[ticker_col] == ticker
            ticker_data = result_df[ticker_mask]
            
            memory = self.compute_memory(
                ticker_data[sentiment_col],
                ticker_data[date_col]
            )
            memory_values.extend(memory.values)
        
        result_df['sentiment_memory'] = memory_values
        
        return result_df


class LinearDecay(DecayFunction):
    """
    Linear intraday decay: w_j = 1 - β * t_j
    where t_j ∈ [0,1] is normalized time of day.
    """
    
    def __init__(self, beta: float):
        """
        Initialize linear decay function.
        
        Args:
            beta: Linear decay slope parameter
        """
        super().__init__(f"linear_beta_{beta}")
        self.beta = beta
    
    def compute_weights(self, normalized_times: np.ndarray) -> np.ndarray:
        """
        Compute linear decay weights.
        
        Args:
            normalized_times: Time normalized to [0, 1] within the day
            
        Returns:
            Decay weights (clipped to [0, 1])
        """
        # Invert times so more recent = higher weight
        elapsed = 1.0 - normalized_times
        weights = 1.0 - self.beta * elapsed
        weights = np.clip(weights, 0, 1)  # Ensure weights stay in [0, 1]
        return weights
    
    def apply_to_dataframe(self, df: pd.DataFrame, time_col: str = 'normalized_time') -> pd.DataFrame:
        """
        Apply linear decay to a DataFrame with timestamps.
        
        Args:
            df: DataFrame with sentiment scores and normalized times
            time_col: Column name containing normalized times [0, 1]
            
        Returns:
            DataFrame with 'decay_weight' column added
        """
        result_df = df.copy()
        result_df['decay_weight'] = self.compute_weights(result_df[time_col].values)
        return result_df


class DecayAggregator:
    """Aggregate sentiment with decay weights."""
    
    @staticmethod
    def aggregate_with_decay(
        df: pd.DataFrame,
        groupby_cols: List[str] = ['Ticker', 'Date'],
        sentiment_col: str = 'sentiment_score',
        weight_col: str = 'decay_weight'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores with decay weights.
        
        Args:
            df: DataFrame with sentiment scores and decay weights
            groupby_cols: Columns to group by
            sentiment_col: Column with sentiment scores
            weight_col: Column with decay weights
            
        Returns:
            Aggregated DataFrame with weighted sentiment
        """
        result_df = df.copy()
        
        # Compute weighted sentiment
        result_df['weighted_sentiment'] = result_df[sentiment_col] * result_df[weight_col]
        
        # Group and aggregate
        agg_dict = {
            'weighted_sentiment': 'sum',
            weight_col: 'sum',
            sentiment_col: ['mean', 'std', 'count']
        }
        
        agg_df = result_df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        # Calculate weighted mean
        agg_df['sent_decay_mean'] = (
            agg_df[('weighted_sentiment', 'sum')] / agg_df[(weight_col, 'sum')]
        )
        
        # Flatten columns
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_df.columns.values]
        
        # Rename for clarity
        rename_dict = {
            'sentiment_score_mean': 'sent_flat_mean',
            'sentiment_score_std': 'sent_std',
            'sentiment_score_count': 'sent_count'
        }
        agg_df = agg_df.rename(columns=rename_dict)
        
        # Drop temporary columns
        cols_to_drop = [col for col in agg_df.columns if col.startswith('weighted_sentiment') or 'decay_weight_sum' in col]
        agg_df = agg_df.drop(columns=cols_to_drop)
        
        return agg_df
    
    @staticmethod
    def compute_weighted_variance(
        df: pd.DataFrame,
        groupby_cols: List[str] = ['Ticker', 'Date'],
        sentiment_col: str = 'sentiment_score',
        weight_col: str = 'decay_weight'
    ) -> pd.DataFrame:
        """
        Compute weighted variance of sentiment scores.
        
        Args:
            df: DataFrame with sentiment scores and decay weights
            groupby_cols: Columns to group by
            sentiment_col: Column with sentiment scores
            weight_col: Column with decay weights
            
        Returns:
            DataFrame with weighted variance
        """
        result_df = df.copy()
        
        # First compute weighted mean per group
        result_df['weighted_sent'] = result_df[sentiment_col] * result_df[weight_col]
        
        grouped = result_df.groupby(groupby_cols)
        weighted_mean = grouped['weighted_sent'].sum() / grouped[weight_col].sum()
        
        # Merge back weighted mean
        result_df = result_df.merge(
            weighted_mean.rename('weighted_mean').reset_index(),
            on=groupby_cols,
            how='left'
        )
        
        # Compute squared deviations
        result_df['squared_dev'] = (result_df[sentiment_col] - result_df['weighted_mean']) ** 2
        result_df['weighted_squared_dev'] = result_df['squared_dev'] * result_df[weight_col]
        
        # Aggregate
        variance_df = result_df.groupby(groupby_cols).agg({
            'weighted_squared_dev': 'sum',
            weight_col: 'sum'
        }).reset_index()
        
        variance_df['sent_decay_var'] = (
            variance_df['weighted_squared_dev'] / variance_df[weight_col]
        )
        
        return variance_df[groupby_cols + ['sent_decay_var']]


def apply_decay_function(
    sentiment_df: pd.DataFrame,
    decay_type: str,
    param_value: float,
    **kwargs
) -> pd.DataFrame:
    """
    Apply specified decay function to sentiment data.
    
    Args:
        sentiment_df: DataFrame with sentiment scores
        decay_type: Type of decay ('exponential', 'half_life', 'linear')
        param_value: Parameter value (lambda, half-life days, or beta)
        **kwargs: Additional arguments for specific decay types
        
    Returns:
        DataFrame with decay-weighted sentiment features
    """
    df = sentiment_df.copy()
    
    # For intraday decay (exponential, linear), compute normalized time if needed
    if decay_type in ['exponential', 'linear']:
        if 'normalized_time' not in df.columns and 'Time' in df.columns:
            # Compute normalized times
            df['hour'] = pd.to_datetime(df['Time']).dt.hour
            df['minute'] = pd.to_datetime(df['Time']).dt.minute
            df['time_of_day'] = df['hour'] + df['minute'] / 60.0
            trading_start = 9.5  # 9:30 AM
            trading_end = 16.0   # 4:00 PM
            df['normalized_time'] = (
                (df['time_of_day'] - trading_start) / (trading_end - trading_start)
            ).clip(0, 1)
    
    if decay_type == 'exponential':
        decay_fn = ExponentialDecay(lambda_param=param_value)
        df_with_weights = decay_fn.apply_to_dataframe(df)
        aggregator = DecayAggregator()
        return aggregator.aggregate_with_decay(df_with_weights)
    
    elif decay_type == 'half_life':
        # Half-life operates on daily aggregated data
        if 'sentiment_score' in df.columns and 'Date' in df.columns and 'Ticker' in df.columns:
            # Aggregate to daily first if not already
            daily_sent = df.groupby(['Ticker', 'Date']).agg({
                'sentiment_score': 'mean'
            }).reset_index()
            daily_sent.columns = ['Ticker', 'Date', 'sent_mean']
            df = daily_sent
        
        decay_fn = HalfLifeDecay(half_life_days=param_value)
        return decay_fn.apply_to_dataframe(df, sentiment_col='sent_mean')
    
    elif decay_type == 'linear':
        decay_fn = LinearDecay(beta=param_value)
        df_with_weights = decay_fn.apply_to_dataframe(df)
        aggregator = DecayAggregator()
        return aggregator.aggregate_with_decay(df_with_weights)
    
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
