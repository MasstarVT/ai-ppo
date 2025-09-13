"""
Technical indicators for trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate various technical indicators for trading."""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Calculate all technical indicators for a given DataFrame."""
        try:
            result_df = df.copy()
            
            # Simple Moving Averages
            for period in config.get('indicators', {}).get('sma_periods', [10, 20, 50]):
                result_df[f'SMA_{period}'] = TechnicalIndicators.sma(df['Close'], period)
            
            # Exponential Moving Averages
            for period in config.get('indicators', {}).get('ema_periods', [12, 26]):
                result_df[f'EMA_{period}'] = TechnicalIndicators.ema(df['Close'], period)
            
            # RSI
            rsi_period = config.get('indicators', {}).get('rsi_period', 14)
            result_df['RSI'] = TechnicalIndicators.rsi(df['Close'], rsi_period)
            
            # MACD
            macd_periods = config.get('indicators', {}).get('macd_periods', [12, 26, 9])
            macd_line, signal_line, histogram = TechnicalIndicators.macd(
                df['Close'], macd_periods[0], macd_periods[1], macd_periods[2]
            )
            result_df['MACD'] = macd_line
            result_df['MACD_Signal'] = signal_line
            result_df['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_period = config.get('indicators', {}).get('bollinger_period', 20)
            bb_std = config.get('indicators', {}).get('bollinger_std', 2)
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                df['Close'], bb_period, bb_std
            )
            result_df['BB_Upper'] = bb_upper
            result_df['BB_Middle'] = bb_middle
            result_df['BB_Lower'] = bb_lower
            
            # Stochastic Oscillator
            stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(
                df['High'], df['Low'], df['Close']
            )
            result_df['Stoch_K'] = stoch_k
            result_df['Stoch_D'] = stoch_d
            
            # ATR
            result_df['ATR'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
            
            # Additional features
            result_df['Returns'] = df['Close'].pct_change()
            result_df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            result_df['Volatility'] = result_df['Returns'].rolling(window=20).std()
            
            # Price position relative to Bollinger Bands
            result_df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume features
            result_df['Volume_SMA'] = TechnicalIndicators.sma(df['Volume'], 20)
            result_df['Volume_Ratio'] = df['Volume'] / result_df['Volume_SMA']
            
            logger.info(f"Calculated technical indicators. Shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df


def normalize_features(df: pd.DataFrame, features: List[str], method: str = 'minmax') -> pd.DataFrame:
    """Normalize features for ML model input."""
    result_df = df.copy()
    
    for feature in features:
        if feature in df.columns:
            if method == 'minmax':
                # Min-Max normalization
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val != min_val:
                    result_df[feature] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    result_df[feature] = 0
            elif method == 'zscore':
                # Z-score normalization
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                if std_val != 0:
                    result_df[feature] = (df[feature] - mean_val) / std_val
                else:
                    result_df[feature] = 0
    
    return result_df


def prepare_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Prepare features for the RL environment."""
    try:
        # Calculate technical indicators
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, config)
        
        # Define feature columns for normalization
        price_features = ['Open', 'High', 'Low', 'Close']
        volume_features = ['Volume', 'Volume_SMA', 'Volume_Ratio']
        technical_features = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position',
            'Stoch_K', 'Stoch_D', 'ATR', 'Volatility'
        ]
        
        # Normalize features if specified in config
        if config.get('environment', {}).get('normalize_observations', True):
            # Normalize price-based features using min-max
            df_with_indicators = normalize_features(
                df_with_indicators, price_features + technical_features, 'minmax'
            )
            # Normalize volume features separately
            df_with_indicators = normalize_features(
                df_with_indicators, volume_features, 'minmax'
            )
        
        # Remove rows with NaN values
        df_with_indicators = df_with_indicators.dropna()
        
        logger.info(f"Prepared features. Final shape: {df_with_indicators.shape}")
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return df