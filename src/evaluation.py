"""
Evaluation module with classification metrics, trading metrics, and SHAP analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

# SHAP for explainability
try:
    import shap
except ImportError:
    shap = None
    logging.warning("shap library not installed. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Compute classification performance metrics."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        return metrics
    
    @staticmethod
    def aggregate_cv_metrics(fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across CV folds.
        
        Args:
            fold_results: List of fold results
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Compute metrics for each fold
        fold_metrics = []
        for fold in fold_results:
            metrics = ClassificationMetrics.compute_metrics(
                fold['y_true'],
                fold['y_pred'],
                fold['y_pred_proba']
            )
            fold_metrics.append(metrics)
        
        # Aggregate
        agg_metrics = {}
        metric_names = ['accuracy', 'f1', 'auc', 'precision', 'recall']
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in fold_metrics if metric_name in m]
            if values:
                agg_metrics[f'{metric_name}_mean'] = np.mean(values)
                agg_metrics[f'{metric_name}_std'] = np.std(values)
        
        return agg_metrics


class TradingMetrics:
    """Compute trading performance metrics."""
    
    @staticmethod
    def compute_returns(
        y_pred: np.ndarray,
        actual_returns: np.ndarray,
        transaction_cost: float = 0.001
    ) -> np.ndarray:
        """
        Compute strategy returns.
        
        Args:
            y_pred: Predicted signals (1 = buy, 0 = sell/hold)
            actual_returns: Actual market returns
            transaction_cost: Transaction cost as fraction
            
        Returns:
            Array of strategy returns
        """
        # Convert predictions to positions (1 = long, -1 = short, 0 = no position)
        positions = np.where(y_pred == 1, 1, -1)
        
        # Strategy returns before costs
        strategy_returns = positions * actual_returns
        
        # Apply transaction costs when position changes
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * transaction_cost
        
        # Net returns
        net_returns = strategy_returns - costs
        
        return net_returns
    
    @staticmethod
    def compute_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Compute annualized Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Daily risk-free rate
        daily_rf = risk_free_rate / periods_per_year
        
        # Excess returns
        excess_returns = returns - daily_rf
        
        # Annualized Sharpe ratio
        sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
        
        return sharpe
    
    @staticmethod
    def compute_cumulative_returns(returns: np.ndarray) -> np.ndarray:
        """
        Compute cumulative returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Array of cumulative returns
        """
        return np.cumprod(1 + returns) - 1
    
    @staticmethod
    def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            cumulative_returns: Array of cumulative returns
            
        Returns:
            Maximum drawdown as fraction
        """
        if len(cumulative_returns) == 0:
            return 0.0
        
        # Compute running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Compute drawdown
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        
        return np.min(drawdown)
    
    @staticmethod
    def compute_trading_metrics(
        y_pred: np.ndarray,
        actual_returns: np.ndarray,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Compute comprehensive trading metrics.
        
        Args:
            y_pred: Predicted signals
            actual_returns: Actual market returns
            transaction_cost: Transaction cost
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with trading metrics
        """
        # Compute strategy returns
        strategy_returns = TradingMetrics.compute_returns(y_pred, actual_returns, transaction_cost)
        
        # Cumulative returns
        cum_returns = TradingMetrics.compute_cumulative_returns(strategy_returns)
        
        # Metrics
        metrics = {
            'total_return': cum_returns[-1] if len(cum_returns) > 0 else 0.0,
            'sharpe_ratio': TradingMetrics.compute_sharpe_ratio(strategy_returns, risk_free_rate),
            'max_drawdown': TradingMetrics.compute_max_drawdown(cum_returns),
            'mean_return': np.mean(strategy_returns),
            'std_return': np.std(strategy_returns),
            'win_rate': np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
        }
        
        return metrics


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.transaction_cost = config['evaluation']['trading_costs']
        self.risk_free_rate = config['evaluation']['risk_free_rate']
    
    def evaluate_cv_results(
        self,
        cv_results: Dict,
        returns_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Evaluate cross-validation results.
        
        Args:
            cv_results: CV results from model training
            returns_data: DataFrame with actual returns (optional)
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        fold_results = cv_results['fold_results']
        
        # Classification metrics
        classification_metrics = ClassificationMetrics.aggregate_cv_metrics(fold_results)
        
        # Trading metrics (if returns data available)
        trading_metrics = {}
        if returns_data is not None:
            all_trading_metrics = []
            
            for fold in fold_results:
                # Extract returns for test period
                # This is simplified - in practice would need to match indices
                if len(fold['y_true']) <= len(returns_data):
                    actual_returns = returns_data.iloc[:len(fold['y_true'])].values
                    
                    fold_trading = TradingMetrics.compute_trading_metrics(
                        fold['y_pred'],
                        actual_returns,
                        self.transaction_cost,
                        self.risk_free_rate
                    )
                    all_trading_metrics.append(fold_trading)
            
            # Aggregate trading metrics
            if all_trading_metrics:
                for key in all_trading_metrics[0].keys():
                    values = [m[key] for m in all_trading_metrics]
                    trading_metrics[f'{key}_mean'] = np.mean(values)
                    trading_metrics[f'{key}_std'] = np.std(values)
        
        return {
            'classification': classification_metrics,
            'trading': trading_metrics,
            'model_type': cv_results['model_type']
        }
    
    def compare_models(
        self,
        baseline_results: Dict,
        flat_sent_results: Dict,
        decay_sent_results: Dict
    ) -> pd.DataFrame:
        """
        Compare different model configurations.
        
        Args:
            baseline_results: Baseline model results
            flat_sent_results: Flat sentiment model results
            decay_sent_results: Decay sentiment model results
            
        Returns:
            Comparison DataFrame
        """
        comparisons = []
        
        for name, results in [
            ('Baseline', baseline_results),
            ('Flat Sentiment', flat_sent_results),
            ('Decay Sentiment', decay_sent_results)
        ]:
            eval_metrics = self.evaluate_cv_results(results)
            
            comparison = {
                'Model': name,
                **eval_metrics['classification'],
                **eval_metrics['trading']
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)


class SHAPAnalyzer:
    """SHAP-based feature importance analysis."""
    
    def __init__(self, config: Dict):
        """
        Initialize SHAP analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_samples = config['evaluation']['shap_samples']
    
    def analyze_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict:
        """
        Perform SHAP analysis on trained model.
        
        Args:
            model: Trained model instance
            X_train: Training data
            X_test: Test data
            feature_names: List of feature names
            
        Returns:
            Dictionary with SHAP values and analysis
        """
        if shap is None:
            logger.warning("SHAP not available. Skipping analysis.")
            return {}
        
        logger.info("Computing SHAP values...")
        
        try:
            # Sample data if too large
            if len(X_train) > self.n_samples:
                X_train_sample = X_train.sample(n=self.n_samples, random_state=42)
            else:
                X_train_sample = X_train
            
            # Create explainer
            explainer = shap.Explainer(model.model, X_train_sample)
            
            # Compute SHAP values for test set
            if len(X_test) > self.n_samples:
                X_test_sample = X_test.sample(n=self.n_samples, random_state=42)
            else:
                X_test_sample = X_test
            
            shap_values = explainer(X_test_sample)
            
            # Feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'explainer': explainer
            }
        
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {}
    
    def compare_feature_importance(
        self,
        baseline_shap: Dict,
        sentiment_shap: Dict
    ) -> pd.DataFrame:
        """
        Compare feature importance between models.
        
        Args:
            baseline_shap: SHAP results for baseline model
            sentiment_shap: SHAP results for sentiment model
            
        Returns:
            Comparison DataFrame
        """
        if not baseline_shap or not sentiment_shap:
            return pd.DataFrame()
        
        baseline_imp = baseline_shap['feature_importance'].set_index('feature')['importance']
        sentiment_imp = sentiment_shap['feature_importance'].set_index('feature')['importance']
        
        # Combine
        comparison = pd.DataFrame({
            'baseline_importance': baseline_imp,
            'sentiment_importance': sentiment_imp
        }).fillna(0)
        
        comparison['importance_change'] = (
            comparison['sentiment_importance'] - comparison['baseline_importance']
        )
        
        return comparison.sort_values('sentiment_importance', ascending=False)


def evaluate_all_models(
    results: Dict,
    config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate all model variants and return comprehensive analysis.
    
    Args:
        results: Dictionary with all model results
        config: Configuration dictionary
        
    Returns:
        Tuple of (comparison DataFrame, detailed metrics dict)
    """
    evaluator = ModelEvaluator(config)
    
    # Extract results for each configuration
    baseline_results = results.get('baseline', {})
    flat_sent_results = results.get('flat_sentiment', {})
    decay_sent_results = results.get('decay_sentiment', {})
    
    # Get model results (assuming xgboost)
    model_type = config['modeling']['models'][0]
    
    baseline_cv = baseline_results.get(model_type, {})
    flat_cv = flat_sent_results.get(model_type, {})
    decay_cv = decay_sent_results.get(model_type, {})
    
    # Evaluate each
    detailed_metrics = {
        'baseline': evaluator.evaluate_cv_results(baseline_cv) if baseline_cv else {},
        'flat_sentiment': evaluator.evaluate_cv_results(flat_cv) if flat_cv else {},
        'decay_sentiment': evaluator.evaluate_cv_results(decay_cv) if decay_cv else {}
    }
    
    # Create comparison DataFrame
    comparison_df = evaluator.compare_models(baseline_cv, flat_cv, decay_cv)
    
    return comparison_df, detailed_metrics
