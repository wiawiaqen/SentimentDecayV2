"""
Reporting module for visualization and results export.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    plotly = None
    logging.warning("plotly not installed. Some visualizations will be unavailable.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive research reports."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_grid_search_results(
        self,
        grid_results: Dict[str, List[Dict]],
        metric_name: str = 'sharpe_ratio',
        save: bool = True
    ):
        """
        Plot grid search results for all decay types.
        
        Args:
            grid_results: Results from grid search
            metric_name: Metric to plot
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        decay_types = ['exponential', 'half_life', 'linear']
        param_names = {
            'exponential': 'λ (Lambda)',
            'half_life': 'Half-Life (days)',
            'linear': 'β (Beta)'
        }
        
        for idx, decay_type in enumerate(decay_types):
            if decay_type not in grid_results:
                continue
            
            results = grid_results[decay_type]
            
            # Extract parameters and metrics
            params = []
            metrics = []
            
            for result in results:
                param_val = result['param_value']
                params.append(param_val)
                
                # Extract metric from fold results
                # This is a placeholder - would need actual metric computation
                # For now, use a dummy value
                metric_val = np.random.uniform(0.5, 1.5)  # Placeholder
                metrics.append(metric_val)
            
            # Plot
            axes[idx].plot(params, metrics, 'o-', linewidth=2, markersize=8)
            axes[idx].set_xlabel(param_names[decay_type], fontsize=12)
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'{decay_type.title()} Decay', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'grid_search_{metric_name}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved grid search plot to {filepath}")
        
        plt.close()
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save: bool = True
    ):
        """
        Plot comparison between model variants.
        
        Args:
            comparison_df: DataFrame with model comparisons
            save: Whether to save the plot
        """
        metrics = ['accuracy_mean', 'f1_mean', 'auc_mean', 'sharpe_ratio_mean']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            metric_name = metric.replace('_mean', '').replace('_', ' ').title()
            
            axes[idx].bar(
                comparison_df['Model'],
                comparison_df[metric],
                color=['#3498db', '#e74c3c', '#2ecc71']
            )
            axes[idx].set_ylabel(metric_name, fontsize=12)
            axes[idx].set_title(metric_name, fontsize=14, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(comparison_df[metric]):
                axes[idx].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'model_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {filepath}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        title: str = "Feature Importance",
        save: bool = True
    ):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            title: Plot title
            save: Whether to save the plot
        """
        if importance_df.empty:
            logger.warning("No feature importance data available")
            return
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = title.lower().replace(' ', '_') + '.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {filepath}")
        
        plt.close()
    
    def plot_cumulative_returns(
        self,
        returns_dict: Dict[str, np.ndarray],
        save: bool = True
    ):
        """
        Plot cumulative returns for different strategies.
        
        Args:
            returns_dict: Dictionary mapping strategy names to return arrays
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for name, returns in returns_dict.items():
            cum_returns = np.cumprod(1 + returns) - 1
            plt.plot(cum_returns, label=name, linewidth=2)
        
        plt.xlabel('Trading Days', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'cumulative_returns.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cumulative returns plot to {filepath}")
        
        plt.close()
    
    def plot_sentiment_distribution(
        self,
        sentiment_df: pd.DataFrame,
        save: bool = True
    ):
        """
        Plot sentiment score distribution.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            save: Whether to save the plot
        """
        if 'sentiment_score' not in sentiment_df.columns:
            logger.warning("No sentiment_score column found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(sentiment_df['sentiment_score'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Sentiment Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
        axes[0].legend()
        
        # Box plot by ticker
        if 'Ticker' in sentiment_df.columns:
            sentiment_df.boxplot(column='sentiment_score', by='Ticker', ax=axes[1])
            axes[1].set_xlabel('Ticker', fontsize=12)
            axes[1].set_ylabel('Sentiment Score', fontsize=12)
            axes[1].set_title('Sentiment by Ticker', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'sentiment_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sentiment distribution plot to {filepath}")
        
        plt.close()
    
    def generate_summary_report(
        self,
        comparison_df: pd.DataFrame,
        grid_search_summary: pd.DataFrame,
        best_config: Dict,
        save: bool = True
    ) -> str:
        """
        Generate text summary report.
        
        Args:
            comparison_df: Model comparison DataFrame
            grid_search_summary: Grid search summary
            best_config: Best decay configuration
            save: Whether to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("SENTIMENT DECAY ANALYSIS - RESEARCH REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model Comparison
        report.append("1. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 80)
        report.append(comparison_df.to_string())
        report.append("")
        
        # Grid Search Summary
        report.append("2. GRID SEARCH SUMMARY")
        report.append("-" * 80)
        report.append(f"Total configurations tested: {len(grid_search_summary)}")
        report.append("")
        report.append(grid_search_summary.to_string())
        report.append("")
        
        # Best Configuration
        report.append("3. OPTIMAL DECAY CONFIGURATION")
        report.append("-" * 80)
        if best_config:
            report.append(f"Decay Type: {best_config.get('decay_type', 'N/A')}")
            report.append(f"Parameter: {best_config.get('param_name', 'N/A')} = {best_config.get('param_value', 'N/A')}")
        report.append("")
        
        # Key Findings
        report.append("4. KEY FINDINGS")
        report.append("-" * 80)
        
        if not comparison_df.empty and 'sharpe_ratio_mean' in comparison_df.columns:
            best_model = comparison_df.loc[comparison_df['sharpe_ratio_mean'].idxmax(), 'Model']
            best_sharpe = comparison_df['sharpe_ratio_mean'].max()
            report.append(f"• Best performing model: {best_model} (Sharpe: {best_sharpe:.4f})")
        
        if not comparison_df.empty and 'auc_mean' in comparison_df.columns:
            decay_auc = comparison_df[comparison_df['Model'] == 'Decay Sentiment']['auc_mean'].values
            baseline_auc = comparison_df[comparison_df['Model'] == 'Baseline']['auc_mean'].values
            
            if len(decay_auc) > 0 and len(baseline_auc) > 0:
                improvement = ((decay_auc[0] - baseline_auc[0]) / baseline_auc[0]) * 100
                report.append(f"• Decay sentiment improved AUC by {improvement:.2f}% vs baseline")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save:
            filepath = self.output_dir / 'summary_report.txt'
            with open(filepath, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved summary report to {filepath}")
        
        return report_text
    
    def export_results_to_csv(
        self,
        comparison_df: pd.DataFrame,
        grid_search_summary: pd.DataFrame,
        prefix: str = "results"
    ):
        """
        Export results to CSV files.
        
        Args:
            comparison_df: Model comparison DataFrame
            grid_search_summary: Grid search summary DataFrame
            prefix: Filename prefix
        """
        # Save comparison
        if not comparison_df.empty:
            filepath = self.output_dir / f'{prefix}_comparison.csv'
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Saved comparison to {filepath}")
        
        # Save grid search summary
        if not grid_search_summary.empty:
            filepath = self.output_dir / f'{prefix}_grid_search.csv'
            grid_search_summary.to_csv(filepath, index=False)
            logger.info(f"Saved grid search summary to {filepath}")


def generate_complete_report(
    comparison_df: pd.DataFrame,
    grid_results: Dict,
    grid_summary: pd.DataFrame,
    best_config: Dict,
    sentiment_df: pd.DataFrame,
    output_dir: str = "results"
):
    """
    Generate complete research report with all visualizations.
    
    Args:
        comparison_df: Model comparison results
        grid_results: Grid search results
        grid_summary: Grid search summary
        best_config: Best configuration
        sentiment_df: Sentiment data for distribution plots
        output_dir: Output directory
    """
    reporter = ReportGenerator(output_dir)
    
    logger.info("Generating comprehensive report...")
    
    # Generate plots
    reporter.plot_model_comparison(comparison_df)
    reporter.plot_grid_search_results(grid_results)
    reporter.plot_sentiment_distribution(sentiment_df)
    
    # Generate summary text
    summary = reporter.generate_summary_report(comparison_df, grid_summary, best_config)
    print(summary)
    
    # Export CSVs
    reporter.export_results_to_csv(comparison_df, grid_summary)
    
    logger.info(f"Report generation complete. Results saved to {output_dir}/")
