"""
Data client for fetching stock market data from various providers with performance optimizations.
Supports YFinance, Alpha Vantage, and Alpaca Markets for comprehensive market data access.
"""

import pandas as pd
import yfinance as yf
import requests
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os

# Import Alpaca clients
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca SDK not available. Install alpaca-py to use Alpaca data provider.")

# Import Binance client
try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("Binance SDK not available. Install python-binance to use Binance data provider.")

# Set up optimized logging for data operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Reduced to WARNING for performance

# Global cache for data
_data_cache = {}
_cache_lock = threading.Lock()


def cache_data(ttl_hours: int = 1):
    """
    Decorator to cache data with TTL (time to live).
    
    Args:
        ttl_hours: Time to live in hours for cached data
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            with _cache_lock:
                if cache_key in _data_cache:
                    data, timestamp = _data_cache[cache_key]
                    if time.time() - timestamp < ttl_hours * 3600:
                        logger.debug(f"Cache hit for {cache_key}")
                        return data
                    else:
                        # Remove expired cache entry
                        del _data_cache[cache_key]
            
            # Cache miss - fetch data
            result = func(*args, **kwargs)
            
            with _cache_lock:
                _data_cache[cache_key] = (result, time.time())
                logger.debug(f"Cached data for {cache_key}")
            
            return result
        return wrapper
    return decorator


def rate_limit(max_calls_per_minute: int = 120):  # Increased limit for faster processing
    """
    Decorator to rate limit API calls with optimized timing.
    
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
    
    @cache_data(ttl_hours=1)  # Cache data for 1 hour
    @rate_limit(max_calls_per_minute=120)  # Increased rate limit
    @retry_on_failure(max_retries=3, backoff_factor=0.5)  # Faster backoff
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """
        Get historical price data from Yahoo Finance with caching and performance optimizations.
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
            
            # Use faster download method for better performance
            data = ticker.history(period=period, interval=interval, 
                                auto_adjust=True,  # Faster processing
                                prepost=False)     # Exclude pre/post market for speed
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol} (normalized: {normalized_symbol})")
                return pd.DataFrame()
            
            # Optimize DataFrame operations
            data = data.dropna()  # Remove NaN values
            
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


class AlpacaProvider(DataProvider):
    """Alpaca Markets data provider for stocks with real-time capabilities."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not available. Install alpaca-py to use Alpaca data provider.")
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.name = "alpaca"
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        
        # Verify connection
        try:
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca account: {account.id} (Paper: {paper})")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    @rate_limit(max_calls_per_minute=200)  # Alpaca has generous rate limits
    @retry_on_failure(max_retries=3, backoff_factor=0.5)
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """Get historical data from Alpaca Markets."""
        try:
            # Map period to start date
            end_date = datetime.now()
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "5d":
                start_date = end_date - timedelta(days=5)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
            
            # Map interval to Alpaca TimeFrame
            timeframe_map = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame(5, "Min"),
                "15m": TimeFrame(15, "Min"),
                "30m": TimeFrame(30, "Min"),
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day,
                "1wk": TimeFrame.Week,
                "1mo": TimeFrame.Month
            }
            
            timeframe = timeframe_map.get(interval, TimeFrame.Hour)
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            # Get data
            bars = self.data_client.get_stock_bars(request)
            
            # Check if we have data
            if not bars or not hasattr(bars, 'df'):
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Get the DataFrame from the BarSet
            df = bars.df
            
            if df.empty:
                logger.warning(f"Empty DataFrame returned for symbol {symbol}")
                return pd.DataFrame()
            
            # Filter for the requested symbol (DataFrame has MultiIndex with symbol)
            if symbol in df.index.get_level_values(0):
                symbol_data = df.loc[symbol]
            else:
                logger.warning(f"Symbol {symbol} not found in returned data")
                return pd.DataFrame()
            
            # Rename columns to match expected format
            if 'close' in symbol_data.columns:
                symbol_data = symbol_data.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
            
            # Ensure index is named correctly
            symbol_data.index.name = 'datetime'
            
            # Remove any rows with NaN values
            symbol_data = symbol_data.dropna()
            
            logger.info(f"Successfully fetched {len(symbol_data)} rows of data for {symbol} from Alpaca")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error fetching Alpaca historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @rate_limit(max_calls_per_minute=200)
    @retry_on_failure(max_retries=2, backoff_factor=0.5)
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time quote data from Alpaca."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol not in quotes:
                logger.warning(f"No quote data found for symbol {symbol}")
                return {}
            
            quote = quotes[symbol]
            
            # Calculate change (we don't have previous close directly, so we estimate)
            current_price = quote.ask_price if quote.ask_price > 0 else quote.bid_price
            
            return {
                'symbol': symbol,
                'price': current_price,
                'bid_price': quote.bid_price,
                'ask_price': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp,
                'exchange': quote.ask_exchange
            }
            
        except Exception as e:
            logger.error(f"Error fetching Alpaca real-time data for {symbol}: {e}")
            return {}


class BinanceProvider(DataProvider):
    """Binance.US data provider for cryptocurrency data."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.name = "binance_us"
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Binance.US client."""
        try:
            # Always use Binance.US (tld='us')
            self.client = BinanceClient(
                self.api_key, 
                self.secret_key,
                tld='us'  # Use Binance.US
            )
            logger.info("Initialized Binance.US client")
                
        except Exception as e:
            logger.error(f"Failed to initialize Binance.US client: {e}")
            raise
    
    @rate_limit(max_calls_per_minute=1200)  # Binance has generous limits
    @retry_on_failure(max_retries=3, backoff_factor=1.0)
    def get_historical_data(self, symbol: str, period: str = '1y', interval: str = '5m') -> pd.DataFrame:
        """Get historical data from Binance."""
        try:
            if not self.client:
                self._initialize_client()
            
            # Map interval to Binance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h',
                '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'
            }
            binance_interval = interval_map.get(interval, '1d')
            
            # Calculate start time based on period
            period_map = {
                '1d': '1 day ago UTC',
                '5d': '5 days ago UTC', 
                '1mo': '1 month ago UTC',
                '3mo': '3 months ago UTC',
                '6mo': '6 months ago UTC',
                '1y': '1 year ago UTC',
                '2y': '2 years ago UTC',
                '5y': '5 years ago UTC'
            }
            start_str = period_map.get(period, '1 year ago UTC')
            
            # Get klines (candlestick data) from Binance
            # DEBUG: Let's test with different strategies based on interval and period
            print(f"🔍 [BINANCE DEBUG] Requesting: {symbol}, {binance_interval}, {start_str}")
            
            try:
                klines = self.client.get_historical_klines(
                    symbol, binance_interval, start_str
                )
                print(f"🔍 [BINANCE DEBUG] Binance API returned {len(klines) if klines else 0} raw klines")
            except Exception as e:
                print(f"🔍 [BINANCE DEBUG] API Error: {e}")
                klines = []
            
            if not klines:
                logger.warning(f"No data returned from Binance for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns to float
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only the essential columns
            df = df[price_columns]
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            logger.info(f"Successfully fetched {len(df)} rows of data for {symbol} from Binance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @rate_limit(max_calls_per_minute=1200)
    @retry_on_failure(max_retries=2, backoff_factor=0.5)
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time ticker data from Binance."""
        try:
            if not self.client:
                self._initialize_client()
                
            ticker = self.client.get_ticker(symbol=symbol)
            
            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'bid_size': float(ticker['bidQty']),
                'ask_size': float(ticker['askQty']),
                'volume': float(ticker['volume']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'change_24h': float(ticker['priceChange']),
                'change_percent_24h': float(ticker['priceChangePercent']),
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching Binance real-time data for {symbol}: {e}")
            return {}


class DataClient:
    """Main data client that manages different providers."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> DataProvider:
        """Initialize the data provider based on configuration."""
        provider_name = self.config.get('data_source', {}).get('provider', 'yfinance')
        
        if provider_name == 'yfinance':
            return YFinanceProvider()
        elif provider_name == 'alphavantage':
            api_key = self.config.get('data_providers', {}).get('alpha_vantage', {}).get('api_key')
            if not api_key:
                raise ValueError("Alpha Vantage API key is required")
            return AlphaVantageProvider(api_key)
        elif provider_name == 'alpaca':
            # Try to get Alpaca credentials from config first, then environment
            alpaca_config = self.config.get('data_providers', {}).get('alpaca', {})
            api_key = alpaca_config.get('api_key') or os.getenv('ALPACA_API_KEY')
            secret_key = alpaca_config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
            paper = alpaca_config.get('paper', True)
            
            if not api_key or not secret_key:
                # Load from .env file if not found
                from dotenv import load_dotenv
                load_dotenv()
                api_key = api_key or os.getenv('ALPACA_API_KEY')
                secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("Alpaca API key and secret key are required. Add them to .env file or configuration.")
            
            logger.info(f"Using Alpaca credentials from {'config' if alpaca_config.get('api_key') else 'environment'}")
            return AlpacaProvider(api_key, secret_key, paper)
        elif provider_name == 'binance':
            # Get Binance.US credentials from config or environment
            binance_config = self.config.get('data_providers', {}).get('binance', {})
            api_key = binance_config.get('api_key') or os.getenv('BINANCE_API_KEY')
            secret_key = binance_config.get('secret_key') or os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key:
                # Load from .env file if not found
                from dotenv import load_dotenv
                load_dotenv()
                api_key = api_key or os.getenv('BINANCE_API_KEY')
                secret_key = secret_key or os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key or api_key == 'your_binance_api_key_here':
                raise ValueError("Binance.US API key and secret key are required. Add them to .env file or configuration.")
            
            logger.info(f"Using Binance.US credentials from {'config' if binance_config.get('api_key') else 'environment'}")
            return BinanceProvider(api_key, secret_key)
        else:
            raise ValueError(f"Unsupported data provider: {provider_name}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available data providers."""
        providers = ['yfinance', 'alphavantage']
        if ALPACA_AVAILABLE:
            providers.append('alpaca')
        if BINANCE_AVAILABLE:
            providers.append('binance')
        return providers
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "5m"
    ) -> pd.DataFrame:
        """Get historical price data."""
        logger.info(f"Fetching historical data for {symbol} (period={period}, interval={interval})")
        return self.provider.get_historical_data(symbol, period, interval)
    
    def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        interval: str = "5m",
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols with parallel processing."""
        data = {}
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.get_historical_data, symbol, period, interval): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        data[symbol] = df
                        logger.debug(f"Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"No data returned for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """Get real-time price data."""
        return self.provider.get_realtime_data(symbol)
    
    def get_realtime_multiple(self, symbols: List[str], max_workers: int = 4) -> Dict[str, Dict]:
        """Get real-time data for multiple symbols with parallel processing."""
        data = {}
        
        # Use ThreadPoolExecutor for parallel real-time data fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.get_realtime_data, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    quote = future.result()
                    if quote:
                        data[symbol] = quote
                except Exception as e:
                    logger.error(f"Error fetching real-time data for {symbol}: {e}")
        
        return data