"""
Main orchestrator for sentiment decay analysis.
Coordinates the entire research pipeline.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import hashlib
import json

from data_loader import load_and_prepare_data
from sentiment_extractor import extract_and_aggregate_sentiment
from decay_functions import apply_decay_function
from feature_engineering import FeatureEngineer
from models import ModelTrainer, train_all_models
from grid_search import run_grid_search
from evaluation import evaluate_all_models, SHAPAnalyzer
from reporting import generate_complete_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentDecayPipeline:
    """Main pipeline for sentiment decay analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {}
        
        # Setup cache directory
        self.cache_dir = Path(self.config.get('data', {}).get('cache_dir', '../cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_cache_hash(self, data_df: pd.DataFrame) -> str:
        """
        Generate hash for caching based on input data.
        
        Args:
            data_df: Input dataframe
            
        Returns:
            Hash string
        """
        # Create hash from dataframe shape and first/last few rows
        cache_str = f"{len(data_df)}_{data_df.shape[1]}"
        if len(data_df) > 0:
            cache_str += f"_{data_df.index[0]}_{data_df.index[-1]}"
        
        return hashlib.md5(cache_str.encode()).hexdigest()[:8]
    
    def _save_sentiment_cache(self, sentiment_df: pd.DataFrame, daily_agg: pd.DataFrame, cache_hash: str):
        """
        Save sentiment results to cache.
        
        Args:
            sentiment_df: Sentiment dataframe
            daily_agg: Daily aggregated sentiment
            cache_hash: Cache identifier
        """
        try:
            cache_file = self.cache_dir / f"sentiment_{cache_hash}.pkl"
            cache_data = {
                'sentiment_df': sentiment_df,
                'daily_agg': daily_agg
            }
            pd.to_pickle(cache_data, cache_file)
            logger.info(f"Saved sentiment cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_sentiment_cache(self, cache_hash: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load sentiment results from cache.
        
        Args:
            cache_hash: Cache identifier
            
        Returns:
            Tuple of (sentiment_df, daily_agg) or None if not found
        """
        try:
            cache_file = self.cache_dir / f"sentiment_{cache_hash}.pkl"
            if cache_file.exists():
                cache_data = pd.read_pickle(cache_file)
                logger.info(f"Loaded sentiment from cache: {cache_file}")
                return cache_data['sentiment_df'], cache_data['daily_agg']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def run_data_loading(self):
        """Step 1: Load and prepare data."""
        logger.info("=" * 80)
        logger.info("STEP 1: DATA LOADING AND PREPARATION")
        logger.info("=" * 80)
        
        market_df, news_df, aligned_df = load_and_prepare_data(self.config)
        
        self.results['market_df'] = market_df
        self.results['news_df'] = news_df
        self.results['aligned_df'] = aligned_df
        
        logger.info(f"Market data: {len(market_df)} rows")
        logger.info(f"News data: {len(news_df)} rows")
        logger.info(f"Aligned data: {len(aligned_df)} rows")
    
    def run_sentiment_extraction(self):
        """Step 2: Extract sentiment scores with caching."""
        logger.info("=" * 80)
        logger.info("STEP 2: SENTIMENT EXTRACTION (with caching)")
        logger.info("=" * 80)
        
        aligned_df = self.results['aligned_df']
        
        # Check cache settings
        cache_enabled = self.config.get('cache', {}).get('sentiment_cache', True)
        force_refresh = self.config.get('cache', {}).get('force_refresh', False)
        
        sentiment_df = None
        daily_agg = None
        
        # Try to load from cache
        if cache_enabled and not force_refresh:
            cache_hash = self._get_cache_hash(aligned_df)
            cached_result = self._load_sentiment_cache(cache_hash)
            
            if cached_result is not None:
                sentiment_df, daily_agg = cached_result
                logger.info("Using cached sentiment results - SKIPPING FinBERT computation!")
        
        # If not cached, compute sentiment
        if sentiment_df is None:
            logger.info("Computing sentiment with FinBERT (this may take a while)...")
            sentiment_df, daily_agg = extract_and_aggregate_sentiment(
                aligned_df,
                self.config
            )
            
            # Save to cache
            if cache_enabled:
                cache_hash = self._get_cache_hash(aligned_df)
                self._save_sentiment_cache(sentiment_df, daily_agg, cache_hash)
        
        self.results['sentiment_df'] = sentiment_df
        self.results['daily_sentiment'] = daily_agg
        
        logger.info(f"Extracted sentiment for {len(sentiment_df)} news items")
        logger.info(f"Daily aggregated sentiment: {len(daily_agg)} days")
    
    def run_baseline_modeling(self):
        """Step 3: Train baseline models (technical indicators only)."""
        logger.info("=" * 80)
        logger.info("STEP 3: BASELINE MODELING")
        logger.info("=" * 80)
        
        market_df = self.results['market_df']
        
        # Prepare features
        engineer = FeatureEngineer(self.config)
        baseline_df = engineer.prepare_features(
            market_df,
            include_sentiment=False
        )
        
        # Get feature names
        feature_cols = engineer.get_feature_names(baseline_df, include_sentiment=False)
        
        # Train models
        trainer = ModelTrainer(self.config)
        baseline_results = trainer.train_baseline(baseline_df, feature_cols)
        
        self.results['baseline_df'] = baseline_df
        self.results['baseline_results'] = baseline_results
        
        logger.info("Baseline modeling complete")
    
    def run_flat_sentiment_modeling(self):
        """Step 4: Train models with flat sentiment features."""
        logger.info("=" * 80)
        logger.info("STEP 4: FLAT SENTIMENT MODELING")
        logger.info("=" * 80)
        
        market_df = self.results['market_df']
        daily_sentiment = self.results['daily_sentiment']
        
        # Merge market data with sentiment
        engineer = FeatureEngineer(self.config)
        merged_df = engineer.merge_market_and_sentiment(market_df, daily_sentiment)
        
        # Prepare features
        flat_sent_df = engineer.prepare_features(
            merged_df,
            include_sentiment=True,
            sentiment_type='flat'
        )
        
        # Get feature names
        feature_cols = engineer.get_feature_names(flat_sent_df, include_sentiment=True)
        
        # Train models
        trainer = ModelTrainer(self.config)
        flat_sent_results = trainer.train_with_sentiment(
            flat_sent_df,
            feature_cols,
            sentiment_type='flat'
        )
        
        self.results['flat_sent_df'] = flat_sent_df
        self.results['flat_sent_results'] = flat_sent_results
        
        logger.info("Flat sentiment modeling complete")
    
    def run_grid_search(self):
        """Step 5: Grid search for optimal decay parameters."""
        logger.info("=" * 80)
        logger.info("STEP 5: DECAY PARAMETER GRID SEARCH")
        logger.info("=" * 80)
        
        sentiment_df = self.results['sentiment_df']
        market_df = self.results['market_df']
        
        # Run grid search
        grid_results, grid_summary = run_grid_search(
            sentiment_df,
            market_df,
            self.config,
            model_type='xgboost'
        )
        
        self.results['grid_results'] = grid_results
        self.results['grid_summary'] = grid_summary
        
        logger.info(f"Grid search complete: {len(grid_summary)} configurations tested")
    
    def run_optimal_decay_modeling(self, decay_type: str = 'exponential', param_value: float = 1.0):
        """Step 6: Train model with optimal decay configuration."""
        logger.info("=" * 80)
        logger.info("STEP 6: OPTIMAL DECAY MODELING")
        logger.info("=" * 80)
        logger.info(f"Using {decay_type} decay with parameter = {param_value}")
        
        sentiment_df = self.results['sentiment_df']
        market_df = self.results['market_df']
        
        # Apply decay function
        decay_df = apply_decay_function(
            sentiment_df,
            decay_type=decay_type,
            param_value=param_value
        )
        
        # Merge with market data
        engineer = FeatureEngineer(self.config)
        merged_df = engineer.merge_market_and_sentiment(market_df, decay_df)
        
        # Prepare features
        decay_sent_df = engineer.prepare_features(
            merged_df,
            include_sentiment=True,
            sentiment_type='decay'
        )
        
        # Get feature names
        feature_cols = engineer.get_feature_names(decay_sent_df, include_sentiment=True)
        
        # Train models
        trainer = ModelTrainer(self.config)
        decay_sent_results = trainer.train_with_sentiment(
            decay_sent_df,
            feature_cols,
            sentiment_type='decay'
        )
        
        self.results['decay_sent_df'] = decay_sent_df
        self.results['decay_sent_results'] = decay_sent_results
        
        logger.info("Optimal decay modeling complete")
    
    def run_evaluation(self):
        """Step 7: Comprehensive evaluation and comparison."""
        logger.info("=" * 80)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Collect all results
        all_results = {
            'baseline': self.results.get('baseline_results', {}),
            'flat_sentiment': self.results.get('flat_sent_results', {}),
            'decay_sentiment': self.results.get('decay_sent_results', {})
        }
        
        # Evaluate
        comparison_df, detailed_metrics = evaluate_all_models(all_results, self.config)
        
        self.results['comparison_df'] = comparison_df
        self.results['detailed_metrics'] = detailed_metrics
        
        logger.info("Evaluation complete")
        logger.info("\n" + comparison_df.to_string())
    
    def run_shap_analysis(self):
        """Step 8: SHAP feature importance analysis."""
        logger.info("=" * 80)
        logger.info("STEP 8: SHAP ANALYSIS")
        logger.info("=" * 80)
        
        # This would require trained models and data
        # Placeholder for now
        logger.info("SHAP analysis would be performed here")
        # Implementation would use SHAPAnalyzer class
    
    def generate_report(self):
        """Step 9: Generate final report."""
        logger.info("=" * 80)
        logger.info("STEP 9: REPORT GENERATION")
        logger.info("=" * 80)
        
        comparison_df = self.results.get('comparison_df', pd.DataFrame())
        grid_results = self.results.get('grid_results', {})
        grid_summary = self.results.get('grid_summary', pd.DataFrame())
        sentiment_df = self.results.get('sentiment_df', pd.DataFrame())
        
        # Best config (placeholder)
        best_config = {
            'decay_type': 'exponential',
            'param_name': 'lambda',
            'param_value': 1.0
        }
        
        generate_complete_report(
            comparison_df,
            grid_results,
            grid_summary,
            best_config,
            sentiment_df,
            output_dir=self.config['data']['output_dir']
        )
        
        logger.info("Report generation complete")
    
    def run_full_pipeline(
        self,
        skip_steps: Optional[list] = None,
        optimal_decay_type: str = 'exponential',
        optimal_param: float = 1.0
    ):
        """
        Run the complete analysis pipeline.
        
        Args:
            skip_steps: List of step numbers to skip
            optimal_decay_type: Type of decay for optimal modeling
            optimal_param: Parameter value for optimal modeling
        """
        if skip_steps is None:
            skip_steps = []
        
        logger.info("=" * 80)
        logger.info("SENTIMENT DECAY ANALYSIS PIPELINE")
        logger.info("=" * 80)
        
        try:
            if 1 not in skip_steps:
                self.run_data_loading()
            
            if 2 not in skip_steps:
                self.run_sentiment_extraction()
            
            if 3 not in skip_steps:
                self.run_baseline_modeling()
            
            if 4 not in skip_steps:
                self.run_flat_sentiment_modeling()
            
            if 5 not in skip_steps:
                self.run_grid_search()
            
            if 6 not in skip_steps:
                self.run_optimal_decay_modeling(optimal_decay_type, optimal_param)
            
            if 7 not in skip_steps:
                self.run_evaluation()
            
            if 8 not in skip_steps:
                self.run_shap_analysis()
            
            if 9 not in skip_steps:
                self.generate_report()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    # Create pipeline
    pipeline = SentimentDecayPipeline(config_path="config.yaml")
    
    # Run full pipeline
    # Note: Steps 2 (sentiment extraction) requires GPU/time, so you may want to skip in testing
    pipeline.run_full_pipeline(
        skip_steps=[],  # Add step numbers to skip, e.g., [2, 5]
        optimal_decay_type='exponential',
        optimal_param=1.0
    )


if __name__ == "__main__":
    main()
