"""
Data loading and preprocessing module for sentiment decay analysis.
Handles market data (AAPL, MSFT) and news headlines (Reuters).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and align market data with news headlines."""
    
    def __init__(self, aapl_path: str, msft_path: str, reuters_path: str):
        """
        Initialize data loader with file paths.
        
        Args:
            aapl_path: Path to AAPL_cleaned.csv
            msft_path: Path to MSFT_cleaned.csv
            reuters_path: Path to reuters_headlines.csv
        """
        self.aapl_path = aapl_path
        self.msft_path = msft_path
        self.reuters_path = reuters_path
        
    def load_market_data(self, ticker: str) -> pd.DataFrame:
        """
        Load market data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (AAPL or MSFT)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        path = self.aapl_path if ticker == "AAPL" else self.msft_path
        
        logger.info(f"Loading market data for {ticker} from {path}")
        df = pd.read_csv(path)
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add ticker column
        df['Ticker'] = ticker
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} rows for {ticker}")
        return df
    
    def load_all_market_data(self) -> pd.DataFrame:
        """
        Load and combine market data for all tickers.
        
        Returns:
            Combined DataFrame with all market data
        """
        aapl_df = self.load_market_data("AAPL")
        msft_df = self.load_market_data("MSFT")
        
        combined_df = pd.concat([aapl_df, msft_df], ignore_index=True)
        combined_df = combined_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        return combined_df
    
    def load_news_headlines(self) -> pd.DataFrame:
        """
        Load Reuters news headlines.
        
        Returns:
            DataFrame with columns: Headlines, Time, Description
        """
        logger.info(f"Loading news headlines from {self.reuters_path}")
        df = pd.read_csv(self.reuters_path)
        
        # Validate required columns
        required_cols = ['Headlines', 'Time', 'Description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert Time to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Headlines', 'Time'])
        
        logger.info(f"Loaded {len(df)} unique headlines")
        return df
    
    def extract_ticker_from_headline(self, headline: str, description: str = "") -> Optional[str]:
        """
        Extract ticker symbol from headline or description.
        
        Args:
            headline: News headline text
            description: News description text
            
        Returns:
            Ticker symbol (AAPL or MSFT) or None if not found
        """
        text = (headline + " " + description).upper()
        
        # Simple keyword matching (can be enhanced with NER)
        if any(keyword in text for keyword in ["APPLE", "AAPL", "IPHONE", "IPAD", "MAC"]):
            return "AAPL"
        elif any(keyword in text for keyword in ["MICROSOFT", "MSFT", "WINDOWS", "AZURE", "XBOX"]):
            return "MSFT"
        
        return None
    
    def align_news_with_market(
        self, 
        market_df: pd.DataFrame, 
        news_df: pd.DataFrame,
        use_previous_day: bool = True
    ) -> pd.DataFrame:
        """
        Align news headlines with market data by ticker and trading day.
        
        Args:
            market_df: Market data DataFrame
            news_df: News headlines DataFrame
            use_previous_day: If True, use previous day's sentiment when markets are closed
            
        Returns:
            DataFrame with aligned news and market data
        """
        logger.info("Aligning news with market data...")
        
        # Extract tickers from headlines
        news_df['Ticker'] = news_df.apply(
            lambda row: self.extract_ticker_from_headline(
                row['Headlines'], 
                row.get('Description', '')
            ),
            axis=1
        )
        
        # Filter out headlines without ticker match
        news_df = news_df[news_df['Ticker'].notna()].copy()
        logger.info(f"Found {len(news_df)} headlines with ticker matches")
        
        # Extract date from time
        news_df['Date'] = news_df['Time'].dt.date
        market_df['Date_only'] = market_df['Date'].dt.date
        
        # Create list to store aligned data
        aligned_data = []
        
        for ticker in market_df['Ticker'].unique():
            ticker_market = market_df[market_df['Ticker'] == ticker].copy()
            ticker_news = news_df[news_df['Ticker'] == ticker].copy()
            
            # Get trading days
            trading_days = set(ticker_market['Date_only'].values)
            
            # Assign each news item to a trading day
            for _, news_row in ticker_news.iterrows():
                news_date = news_row['Date']
                
                # Find the appropriate trading day
                if news_date in trading_days:
                    trading_day = news_date
                elif use_previous_day:
                    # Find the next trading day
                    future_days = [d for d in trading_days if d > news_date]
                    if future_days:
                        trading_day = min(future_days)
                    else:
                        continue  # Skip if no future trading day
                else:
                    continue
                
                # Find market data for this trading day
                market_row = ticker_market[ticker_market['Date_only'] == trading_day]
                if not market_row.empty:
                    market_row = market_row.iloc[0]
                    aligned_data.append({
                        'Ticker': ticker,
                        'Date': market_row['Date'],
                        'Time': news_row['Time'],
                        'Headlines': news_row['Headlines'],
                        'Description': news_row['Description'],
                        'Open': market_row['Open'],
                        'High': market_row['High'],
                        'Low': market_row['Low'],
                        'Close': market_row['Close'],
                        'Volume': market_row['Volume']
                    })
        
        aligned_df = pd.DataFrame(aligned_data)
        logger.info(f"Created {len(aligned_df)} aligned news-market records")
        
        return aligned_df
    
    def create_target_variable(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable: 1 if Close[t+1] > Close[t], else 0.
        
        Args:
            market_df: Market data DataFrame
            
        Returns:
            DataFrame with 'Target' column added
        """
        result_df = market_df.copy()
        
        # Sort by ticker and date
        result_df = result_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Calculate next day's close for each ticker
        result_df['Next_Close'] = result_df.groupby('Ticker')['Close'].shift(-1)
        
        # Create binary target
        result_df['Target'] = (result_df['Next_Close'] > result_df['Close']).astype(int)
        
        # Remove last row for each ticker (no next day data)
        result_df = result_df[result_df['Next_Close'].notna()].copy()
        
        logger.info(f"Created target variable for {len(result_df)} samples")
        logger.info(f"Target distribution: {result_df['Target'].value_counts().to_dict()}")
        
        return result_df.drop(columns=['Next_Close'])
    
    def get_trading_days(self, market_df: pd.DataFrame, ticker: str) -> List[datetime]:
        """
        Get list of trading days for a specific ticker.
        
        Args:
            market_df: Market data DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            List of trading days as datetime objects
        """
        ticker_data = market_df[market_df['Ticker'] == ticker]
        return sorted(ticker_data['Date'].unique())


def load_and_prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to load and prepare all data.
    
    Args:
        config: Configuration dictionary with data paths
        
    Returns:
        Tuple of (market_df, news_df, aligned_df)
    """
    loader = DataLoader(
        aapl_path=config['data']['aapl_path'],
        msft_path=config['data']['msft_path'],
        reuters_path=config['data']['reuters_path']
    )
    
    # Load data
    market_df = loader.load_all_market_data()
    news_df = loader.load_news_headlines()
    
    # Align news with market
    aligned_df = loader.align_news_with_market(market_df, news_df)
    
    # Create target variable
    market_df = loader.create_target_variable(market_df)
    
    return market_df, news_df, aligned_df
