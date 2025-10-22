"""
Sentiment extraction module using FinBERT for financial news analysis.
Computes sentiment scores as P(positive) - P(negative).
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentExtractor:
    """Extract sentiment scores from financial news using FinBERT."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, max_length: int = 512):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def predict_sentiment_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of dictionaries with sentiment scores
        """
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to CPU and numpy
        predictions = predictions.cpu().numpy()
        
        # Format results
        # FinBERT outputs: [negative, neutral, positive]
        results = []
        for pred in predictions:
            results.append({
                'negative': float(pred[0]),
                'neutral': float(pred[1]),
                'positive': float(pred[2]),
                'sentiment_score': float(pred[2] - pred[0])  # P(pos) - P(neg)
            })
        
        return results
    
    def extract_sentiment(self, df: pd.DataFrame, text_column: str = 'Headlines') -> pd.DataFrame:
        """
        Extract sentiment scores for all texts in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text
            
        Returns:
            DataFrame with added sentiment columns
        """
        result_df = df.copy()
        
        logger.info(f"Extracting sentiment for {len(df)} texts...")
        
        # Process in batches
        all_sentiments = []
        texts = result_df[text_column].tolist()
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_sentiments = self.predict_sentiment_batch(batch_texts)
            all_sentiments.extend(batch_sentiments)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        # Add sentiment columns
        sentiment_df = pd.DataFrame(all_sentiments)
        result_df = pd.concat([result_df, sentiment_df], axis=1)
        
        logger.info("Sentiment extraction completed")
        logger.info(f"Mean sentiment score: {result_df['sentiment_score'].mean():.4f}")
        logger.info(f"Sentiment score range: [{result_df['sentiment_score'].min():.4f}, {result_df['sentiment_score'].max():.4f}]")
        
        return result_df
    
    def extract_with_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sentiment using both headlines and descriptions.
        
        Args:
            df: DataFrame with 'Headlines' and 'Description' columns
            
        Returns:
            DataFrame with sentiment scores
        """
        result_df = df.copy()
        
        # Combine headlines and descriptions
        result_df['combined_text'] = result_df.apply(
            lambda row: f"{row['Headlines']} {row.get('Description', '')}".strip(),
            axis=1
        )
        
        # Extract sentiment
        result_df = self.extract_sentiment(result_df, text_column='combined_text')
        
        # Drop temporary column
        result_df = result_df.drop(columns=['combined_text'])
        
        return result_df


class SentimentAggregator:
    """Aggregate sentiment scores by ticker and date."""
    
    @staticmethod
    def aggregate_daily_sentiment(
        sentiment_df: pd.DataFrame,
        groupby_cols: List[str] = ['Ticker', 'Date']
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by ticker and date.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            groupby_cols: Columns to group by
            
        Returns:
            Aggregated DataFrame with statistics
        """
        agg_dict = {
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }
        
        agg_df = sentiment_df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            '_'.join(col).strip('_') if col[1] else col[0] 
            for col in agg_df.columns.values
        ]
        
        # Rename for clarity
        rename_dict = {
            'sentiment_score_mean': 'sent_mean',
            'sentiment_score_std': 'sent_std',
            'sentiment_score_min': 'sent_min',
            'sentiment_score_max': 'sent_max',
            'sentiment_score_count': 'sent_count',
            'positive_mean': 'positive_mean',
            'negative_mean': 'negative_mean',
            'neutral_mean': 'neutral_mean'
        }
        agg_df = agg_df.rename(columns=rename_dict)
        
        # Fill NaN std with 0 (when only one news item)
        agg_df['sent_std'] = agg_df['sent_std'].fillna(0)
        
        return agg_df
    
    @staticmethod
    def aggregate_intraday_sentiment(
        sentiment_df: pd.DataFrame,
        groupby_cols: List[str] = ['Ticker', 'Date']
    ) -> pd.DataFrame:
        """
        Aggregate intraday sentiment with time-of-day information.
        
        Args:
            sentiment_df: DataFrame with sentiment scores and Time column
            groupby_cols: Columns to group by
            
        Returns:
            Aggregated DataFrame with intraday statistics
        """
        df = sentiment_df.copy()
        
        # Extract time features
        df['hour'] = pd.to_datetime(df['Time']).dt.hour
        df['minute'] = pd.to_datetime(df['Time']).dt.minute
        df['time_of_day'] = df['hour'] + df['minute'] / 60.0  # Fractional hours
        
        # Normalize time to [0, 1] (assuming trading hours 9:30 AM - 4:00 PM EST)
        trading_start = 9.5  # 9:30 AM
        trading_end = 16.0   # 4:00 PM
        df['normalized_time'] = (df['time_of_day'] - trading_start) / (trading_end - trading_start)
        df['normalized_time'] = df['normalized_time'].clip(0, 1)
        
        # Group and aggregate
        agg_df = df.groupby(groupby_cols).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'normalized_time': ['mean', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_df.columns.values]
        
        return agg_df


def extract_and_aggregate_sentiment(
    aligned_df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to extract and aggregate sentiment.
    
    Args:
        aligned_df: DataFrame with aligned news and market data
        config: Configuration dictionary
        
    Returns:
        Tuple of (sentiment_df with scores, aggregated_df)
    """
    # Extract sentiment
    extractor = SentimentExtractor(
        model_name=config['sentiment']['model_name'],
        batch_size=config['sentiment']['batch_size'],
        max_length=config['sentiment']['max_length']
    )
    
    sentiment_df = extractor.extract_with_description(aligned_df)
    
    # Aggregate daily sentiment
    aggregator = SentimentAggregator()
    daily_agg = aggregator.aggregate_daily_sentiment(sentiment_df)
    
    return sentiment_df, daily_agg
