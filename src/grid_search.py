"""
Grid search module for optimizing decay parameters.
Evaluates different decay functions and parameters to find optimal configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from itertools import product

from decay_functions import ExponentialDecay, HalfLifeDecay, LinearDecay, DecayAggregator
from models import ModelTrainer
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecayGridSearch:
    """Grid search for optimal decay parameters."""
    
    def __init__(self, config: Dict):
        """
        Initialize grid search.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_metric = config['grid_search']['optimization_metric']
        self.results = []
    
    def create_decay_features(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        decay_type: str,
        param_value: float
    ) -> pd.DataFrame:
        """
        Create features with specific decay configuration.
        
        Args:
            sentiment_df: DataFrame with sentiment scores and timestamps
            market_df: DataFrame with market data
            decay_type: Type of decay function
            param_value: Parameter value for decay
            
        Returns:
            DataFrame with decay-weighted features merged with market data
        """
        # Apply decay function
        if decay_type == 'exponential':
            decay_fn = ExponentialDecay(lambda_param=param_value)
            
            # Need normalized times for intraday decay
            if 'normalized_time' not in sentiment_df.columns:
                # Compute normalized times
                sentiment_df = sentiment_df.copy()
                sentiment_df['hour'] = pd.to_datetime(sentiment_df['Time']).dt.hour
                sentiment_df['minute'] = pd.to_datetime(sentiment_df['Time']).dt.minute
                sentiment_df['time_of_day'] = sentiment_df['hour'] + sentiment_df['minute'] / 60.0
                trading_start = 9.5
                trading_end = 16.0
                sentiment_df['normalized_time'] = (
                    (sentiment_df['time_of_day'] - trading_start) / (trading_end - trading_start)
                ).clip(0, 1)
            
            # Apply decay
            df_with_weights = decay_fn.apply_to_dataframe(sentiment_df)
            aggregator = DecayAggregator()
            decay_features = aggregator.aggregate_with_decay(df_with_weights)
        
        elif decay_type == 'half_life':
            # First aggregate daily sentiment
            daily_sent = sentiment_df.groupby(['Ticker', 'Date']).agg({
                'sentiment_score': 'mean'
            }).reset_index()
            daily_sent.columns = ['Ticker', 'Date', 'sent_mean']
            
            # Apply half-life decay
            decay_fn = HalfLifeDecay(half_life_days=param_value)
            decay_features = decay_fn.apply_to_dataframe(
                daily_sent,
                sentiment_col='sent_mean',
                date_col='Date',
                ticker_col='Ticker'
            )
        
        elif decay_type == 'linear':
            decay_fn = LinearDecay(beta=param_value)
            
            # Need normalized times
            if 'normalized_time' not in sentiment_df.columns:
                sentiment_df = sentiment_df.copy()
                sentiment_df['hour'] = pd.to_datetime(sentiment_df['Time']).dt.hour
                sentiment_df['minute'] = pd.to_datetime(sentiment_df['Time']).dt.minute
                sentiment_df['time_of_day'] = sentiment_df['hour'] + sentiment_df['minute'] / 60.0
                trading_start = 9.5
                trading_end = 16.0
                sentiment_df['normalized_time'] = (
                    (sentiment_df['time_of_day'] - trading_start) / (trading_end - trading_start)
                ).clip(0, 1)
            
            df_with_weights = decay_fn.apply_to_dataframe(sentiment_df)
            aggregator = DecayAggregator()
            decay_features = aggregator.aggregate_with_decay(df_with_weights)
        
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
        
        # Merge with market data
        market_df['Date'] = pd.to_datetime(market_df['Date'])
        decay_features['Date'] = pd.to_datetime(decay_features['Date'])
        
        merged_df = market_df.merge(
            decay_features,
            on=['Ticker', 'Date'],
            how='left'
        )
        
        return merged_df
    
    def evaluate_decay_config(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        decay_type: str,
        param_value: float,
        model_type: str = 'xgboost'
    ) -> Dict:
        """
        Evaluate a specific decay configuration.
        
        Args:
            sentiment_df: Sentiment data
            market_df: Market data
            decay_type: Type of decay
            param_value: Parameter value
            model_type: Type of model to use
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {decay_type} decay with param={param_value}")
        
        # Create features with this decay configuration
        df = self.create_decay_features(sentiment_df, market_df, decay_type, param_value)
        
        # Prepare features
        engineer = FeatureEngineer(self.config)
        df = engineer.prepare_features(df, include_sentiment=True, sentiment_type='decay')
        
        # Get feature columns
        feature_cols = engineer.get_feature_names(df, include_sentiment=True)
        
        # Train model with CV
        trainer = ModelTrainer(self.config)
        try:
            cv_results = trainer.train_with_cv(df, feature_cols, model_type)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
        
        return {
            'decay_type': decay_type,
            'param_value': param_value,
            'param_name': self._get_param_name(decay_type),
            'cv_results': cv_results
        }
    
    def _get_param_name(self, decay_type: str) -> str:
        """Get parameter name for decay type."""
        if decay_type == 'exponential':
            return 'lambda'
        elif decay_type == 'half_life':
            return 'half_life_days'
        elif decay_type == 'linear':
            return 'beta'
        return 'param'
    
    def grid_search_exponential(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> List[Dict]:
        """
        Grid search over exponential decay lambda values.
        
        Args:
            sentiment_df: Sentiment data
            market_df: Market data
            model_type: Type of model to use
            
        Returns:
            List of evaluation results
        """
        lambda_values = self.config['decay']['exponential']['lambda_values']
        
        logger.info(f"Starting exponential decay grid search with λ = {lambda_values}")
        
        results = []
        for lambda_val in lambda_values:
            result = self.evaluate_decay_config(
                sentiment_df, market_df, 'exponential', lambda_val, model_type
            )
            if result is not None:
                results.append(result)
        
        return results
    
    def grid_search_half_life(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> List[Dict]:
        """
        Grid search over half-life decay values.
        
        Args:
            sentiment_df: Sentiment data
            market_df: Market data
            model_type: Type of model to use
            
        Returns:
            List of evaluation results
        """
        h_values = self.config['decay']['half_life']['h_values']
        
        logger.info(f"Starting half-life decay grid search with h = {h_values} days")
        
        results = []
        for h_val in h_values:
            result = self.evaluate_decay_config(
                sentiment_df, market_df, 'half_life', h_val, model_type
            )
            if result is not None:
                results.append(result)
        
        return results
    
    def grid_search_linear(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> List[Dict]:
        """
        Grid search over linear decay beta values.
        
        Args:
            sentiment_df: Sentiment data
            market_df: Market data
            model_type: Type of model to use
            
        Returns:
            List of evaluation results
        """
        beta_values = self.config['decay']['linear']['beta_values']
        
        logger.info(f"Starting linear decay grid search with β = {beta_values}")
        
        results = []
        for beta_val in beta_values:
            result = self.evaluate_decay_config(
                sentiment_df, market_df, 'linear', beta_val, model_type
            )
            if result is not None:
                results.append(result)
        
        return results
    
    def search_all_decay_types(
        self,
        sentiment_df: pd.DataFrame,
        market_df: pd.DataFrame,
        model_type: str = 'xgboost'
    ) -> Dict[str, List[Dict]]:
        """
        Perform grid search for all decay types.
        
        Args:
            sentiment_df: Sentiment data
            market_df: Market data
            model_type: Type of model to use
            
        Returns:
            Dictionary with results for each decay type
        """
        results = {
            'exponential': self.grid_search_exponential(sentiment_df, market_df, model_type),
            'half_life': self.grid_search_half_life(sentiment_df, market_df, model_type),
            'linear': self.grid_search_linear(sentiment_df, market_df, model_type)
        }
        
        self.results = results
        return results
    
    def find_best_config(self, results: Dict[str, List[Dict]], metric: str = None) -> Dict:
        """
        Find the best decay configuration across all types.
        
        Args:
            results: Grid search results
            metric: Metric to optimize (uses config default if None)
            
        Returns:
            Best configuration dictionary
        """
        if metric is None:
            metric = self.optimization_metric
        
        logger.info(f"Finding best configuration based on {metric}")
        
        best_config = None
        best_score = -np.inf
        
        for decay_type, decay_results in results.items():
            for result in decay_results:
                # Calculate average metric across folds
                fold_results = result['cv_results']['fold_results']
                
                # Metric will be computed in evaluation module
                # For now, we'll store the config and results
                # This is a placeholder - actual metric computation happens in evaluation
                
                if best_config is None:
                    best_config = result
                    # Will be properly computed later
        
        return best_config
    
    def create_summary_dataframe(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Create summary DataFrame of grid search results.
        
        Args:
            results: Grid search results
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for decay_type, decay_results in results.items():
            for result in decay_results:
                summary_data.append({
                    'decay_type': decay_type,
                    'param_name': result['param_name'],
                    'param_value': result['param_value'],
                    'n_folds': len(result['cv_results']['fold_results'])
                })
        
        return pd.DataFrame(summary_data)


def run_grid_search(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: Dict,
    model_type: str = 'xgboost'
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run complete grid search for decay optimization.
    
    Args:
        sentiment_df: Sentiment data with scores
        market_df: Market data
        config: Configuration dictionary
        model_type: Type of model to use
        
    Returns:
        Tuple of (results dict, summary DataFrame)
    """
    searcher = DecayGridSearch(config)
    
    # Run grid search for all decay types
    results = searcher.search_all_decay_types(sentiment_df, market_df, model_type)
    
    # Create summary
    summary = searcher.create_summary_dataframe(results)
    
    logger.info("Grid search completed")
    logger.info(f"Total configurations evaluated: {len(summary)}")
    
    return results, summary
