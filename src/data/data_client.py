"""
Data client for fetching stock market data from various providers.
Since TradingView doesn't have an official public API, we use alternative providers.
"""

import pandas as pd
import yfinance as yf
import requests
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from functools import wraps
import random

# Set up optimized logging for data operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO for performance


def rate_limit(max_calls_per_minute: int = 60):
    """
    Decorator to rate limit API calls.
    
    Args:
        max_calls_per_minute: Maximum number of calls allowed per minute
    """
    min_interval = 60.0 / max_calls_per_minute
    last_called = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            if func_name in last_called:
                elapsed = now - last_called[func_name]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    logger.debug(f"Rate limiting {func_name}, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            last_called[func_name] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator to retry failed API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get historical price data."""
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time price data."""
        pass


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance library with cryptocurrency support."""
    
    def __init__(self):
        self.name = "yfinance"
        # Supported crypto symbols mapping from common trading pairs to yfinance format
        self.crypto_symbols = {
            'BTC/USD': 'BTC-USD',
            'BTC/USDT': 'BTC-USD',  # yfinance uses BTC-USD for USDT pairs
            'ETH/USD': 'ETH-USD',
            'ETH/USDT': 'ETH-USD',
            'ADA/USD': 'ADA-USD',
            'ADA/USDT': 'ADA-USD',
            'DOT/USD': 'DOT-USD',
            'DOT/USDT': 'DOT-USD',
            'LINK/USD': 'LINK-USD',
            'LINK/USDT': 'LINK-USD',
            'XRP/USD': 'XRP-USD',
            'XRP/USDT': 'XRP-USD',
            'LTC/USD': 'LTC-USD',
            'LTC/USDT': 'LTC-USD',
            'BCH/USD': 'BCH-USD',
            'BCH/USDT': 'BCH-USD',
            'MATIC/USD': 'MATIC-USD',
            'MATIC/USDT': 'MATIC-USD',
            'SOL/USD': 'SOL-USD',
            'SOL/USDT': 'SOL-USD',
            'AVAX/USD': 'AVAX-USD',
            'AVAX/USDT': 'AVAX-USD',
            'ATOM/USD': 'ATOM-USD',
            'ATOM/USDT': 'ATOM-USD',
            'DOGE/USD': 'DOGE-USD',
            'DOGE/USDT': 'DOGE-USD'
        }
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for yfinance (convert crypto pairs if needed)."""
        # Check if it's a crypto pair that needs conversion
        if symbol in self.crypto_symbols:
            normalized = self.crypto_symbols[symbol]
            logger.info(f"Converting crypto symbol {symbol} to {normalized}")
            return normalized
        
        # Check if it's already a yfinance crypto format (e.g., BTC-USD)
        if '-USD' in symbol:
            return symbol
            
        # For regular stocks, return as-is
        return symbol
    
    @rate_limit(max_calls_per_minute=60)  # Yahoo Finance rate limiting
    @retry_on_failure(max_retries=3, backoff_factor=1.0)
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """
        Get historical price data from Yahoo Finance.
        Supports both stocks (e.g., 'AAPL') and cryptocurrencies (e.g., 'BTC/USDT', 'ETH/USD').
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL') or crypto pair (e.g., 'BTC/USDT', 'ETH/USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Normalize the symbol (convert crypto pairs to yfinance format)
            normalized_symbol = self._normalize_symbol(symbol)
            
            ticker = yf.Ticker(normalized_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol} (normalized: {normalized_symbol})")
                return pd.DataFrame()
            
            # Handle different possible column names from yfinance
            original_columns = list(data.columns)
            
            # Standard expected columns
            standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Only rename if we have the expected number of columns
            if len(original_columns) >= 4:  # At least OHLC
                # Create mapping for existing columns
                column_mapping = {}
                for i, std_col in enumerate(standard_columns):
                    if i < len(original_columns):
                        column_mapping[original_columns[i]] = std_col
                
                # Rename columns
                data = data.rename(columns=column_mapping)
                
                # Ensure we have Volume column
                if 'Volume' not in data.columns and len(original_columns) < 5:
                    data['Volume'] = 0
                    logger.warning(f"Volume data not available for {symbol}, using zeros")
                
                # Select only the standard columns that exist
                available_columns = [col for col in standard_columns if col in data.columns]
                data = data[available_columns]
            
            data.index.name = 'datetime'
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            logger.info(f"Successfully fetched {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @rate_limit(max_calls_per_minute=60)
    @retry_on_failure(max_retries=2, backoff_factor=0.5)
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time price data for stocks and cryptocurrencies."""
        try:
            # Normalize the symbol (convert crypto pairs to yfinance format)
            normalized_symbol = self._normalize_symbol(symbol)
            
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,  # Return original symbol for display
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return {}


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.name = "alphavantage"
    
    @rate_limit(max_calls_per_minute=5)  # Alpha Vantage has stricter limits
    @retry_on_failure(max_retries=3, backoff_factor=2.0)
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "60min") -> pd.DataFrame:
        """Get historical data from Alpha Vantage."""
        try:
            # Map interval to Alpha Vantage format
            interval_map = {
                "1m": "1min", "5m": "5min", "15m": "15min", 
                "30m": "30min", "1h": "60min", "1d": "daily"
            }
            av_interval = interval_map.get(interval, "60min")
            
            if av_interval == "daily":
                function = "TIME_SERIES_DAILY"
            else:
                function = "TIME_SERIES_INTRADAY"
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            if function == "TIME_SERIES_INTRADAY":
                params['interval'] = av_interval
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            # Parse the time series data
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f'Time Series ({av_interval})'
            else:
                time_series_key = 'Time Series (Daily)'
            
            time_series = data.get(time_series_key, {})
            
            if not time_series:
                logger.warning(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            quote = data.get('Global Quote', {})
            
            if not quote:
                return {}
            
            return {
                'symbol': symbol,
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                'volume': int(quote.get('06. volume', 0)),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage real-time data for {symbol}: {e}")
            return {}


class DataClient:
    """Main data client that manages different providers."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> DataProvider:
        """Initialize the data provider based on configuration."""
        provider_name = self.config.get('tradingview', {}).get('provider', 'yfinance')
        
        if provider_name == 'yfinance':
            return YFinanceProvider()
        elif provider_name == 'alphavantage':
            api_key = self.config.get('data_providers', {}).get('alpha_vantage', {}).get('api_key')
            if not api_key:
                raise ValueError("Alpha Vantage API key is required")
            return AlphaVantageProvider(api_key)
        else:
            raise ValueError(f"Unsupported data provider: {provider_name}")
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1h"
    ) -> pd.DataFrame:
        """Get historical price data."""
        logger.info(f"Fetching historical data for {symbol} (period={period}, interval={interval})")
        return self.provider.get_historical_data(symbol, period, interval)
    
    def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        interval: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols."""
        data = {}
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.get_historical_data(symbol, period, interval)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)  # Rate limiting
        return data
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time price data."""
        return self.provider.get_realtime_data(symbol)
    
    def get_realtime_multiple(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time data for multiple symbols."""
        data = {}
        for symbol in symbols:
            quote = self.get_realtime_data(symbol)
            if quote:
                data[symbol] = quote
            time.sleep(0.1)  # Rate limiting
        return data