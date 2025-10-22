"""
Modeling framework with baseline and sentiment-enhanced models.
Includes rolling time-series cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RollingTimeSeriesSplit:
    """Custom rolling time-series cross-validator."""
    
    def __init__(self, train_window: int = 252, test_window: int = 21, n_splits: int = 10):
        """
        Initialize rolling time-series splitter.
        
        Args:
            train_window: Number of days in training window
            test_window: Number of days in test window
            n_splits: Number of splits to generate
        """
        self.train_window = train_window
        self.test_window = test_window
        self.n_splits = n_splits
    
    def split(self, df: pd.DataFrame, date_col: str = 'Date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = df[date_col].sort_values().unique()
        total_days = len(dates)
        
        if total_days < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data: {total_days} days available, "
                f"but need at least {self.train_window + self.test_window}"
            )
        
        splits = []
        step_size = max(1, (total_days - self.train_window - self.test_window) // (self.n_splits - 1))
        
        for i in range(self.n_splits):
            start_idx = i * step_size
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window
            
            if test_end_idx > total_days:
                break
            
            train_dates = dates[start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            train_mask = df[date_col].isin(train_dates)
            test_mask = df[date_col].isin(test_dates)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            splits.append((train_indices, test_indices))
            
            logger.info(
                f"Split {len(splits)}: Train {train_dates[0]} to {train_dates[-1]}, "
                f"Test {test_dates[0]} to {test_dates[-1]}"
            )
        
        return splits


class BaseModel:
    """Base class for predictive models."""
    
    def __init__(self, model_type: str = 'logistic_regression', **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ('logistic_regression' or 'xgboost')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.kwargs = kwargs
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **self.kwargs
            )
        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            default_params.update(self.kwargs)
            self.model = XGBClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None):
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if feature_names is not None:
            self.feature_names = feature_names
            X = X[feature_names]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importances or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        return None


class ModelTrainer:
    """Train and evaluate models with rolling time-series CV."""
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.train_window = config['modeling']['train_window']
        self.test_window = config['modeling']['test_window']
        self.n_splits = config['modeling']['rolling_splits']
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'Target'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y)
        """
        # Remove rows with missing target
        df = df[df[target_col].notna()].copy()
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with any missing features
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        
        return X, y
    
    def train_with_cv(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'logistic_regression',
        **model_kwargs
    ) -> Dict:
        """
        Train model with rolling time-series cross-validation.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            model_type: Type of model to train
            **model_kwargs: Additional model parameters
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Training {model_type} with rolling CV...")
        
        # Prepare data
        X, y = self.prepare_data(df, feature_cols)
        
        # Add date column back for splitting
        X_with_date = X.copy()
        X_with_date['Date'] = df.loc[X.index, 'Date']
        
        # Create rolling splits
        splitter = RollingTimeSeriesSplit(
            train_window=self.train_window,
            test_window=self.test_window,
            n_splits=self.n_splits
        )
        
        splits = splitter.split(X_with_date, date_col='Date')
        
        # Store results for each fold
        fold_results = []
        models = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(splits)}...")
            
            # Use loc instead of iloc to work with actual indices
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Initialize and train model
            model = BaseModel(model_type=model_type, **model_kwargs)
            model.fit(X_train, y_train, feature_names=feature_cols)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Store results
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            
            models.append(model)
        
        return {
            'model_type': model_type,
            'feature_cols': feature_cols,
            'fold_results': fold_results,
            'models': models
        }
    
    def train_baseline(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        Train baseline model (technical indicators only).
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            
        Returns:
            Training results dictionary
        """
        # Filter to only technical features (exclude sentiment)
        tech_features = [col for col in feature_cols if 'sent' not in col.lower()]
        
        logger.info(f"Training baseline with {len(tech_features)} technical features")
        
        results = {}
        for model_type in self.config['modeling']['models']:
            results[model_type] = self.train_with_cv(df, tech_features, model_type)
        
        return results
    
    def train_with_sentiment(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sentiment_type: str = 'flat'
    ) -> Dict:
        """
        Train model with sentiment features.
        
        Args:
            df: DataFrame with features including sentiment
            feature_cols: List of feature column names
            sentiment_type: Type of sentiment ('flat' or 'decay')
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training with {sentiment_type} sentiment features")
        
        results = {}
        for model_type in self.config['modeling']['models']:
            results[model_type] = self.train_with_cv(df, feature_cols, model_type)
        
        return results


def train_all_models(
    df: pd.DataFrame,
    feature_cols_baseline: List[str],
    feature_cols_flat_sent: List[str],
    feature_cols_decay_sent: List[str],
    config: Dict
) -> Dict:
    """
    Train all model variants (baseline, flat sentiment, decay sentiment).
    
    Args:
        df: DataFrame with all features
        feature_cols_baseline: Baseline feature columns
        feature_cols_flat_sent: Flat sentiment feature columns
        feature_cols_decay_sent: Decay sentiment feature columns
        config: Configuration dictionary
        
    Returns:
        Dictionary with all training results
    """
    trainer = ModelTrainer(config)
    
    results = {
        'baseline': trainer.train_baseline(df, feature_cols_baseline),
        'flat_sentiment': trainer.train_with_sentiment(df, feature_cols_flat_sent, 'flat'),
        'decay_sentiment': trainer.train_with_sentiment(df, feature_cols_decay_sent, 'decay')
    }
    
    return results
