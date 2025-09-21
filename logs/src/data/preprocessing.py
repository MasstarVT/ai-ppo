"""
Data preprocessing pipeline for trading ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses trading data for ML models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        self.is_fitted = False
        
    def prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare training data from multiple symbols.
        
        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV + indicators
            
        Returns:
            Combined preprocessed DataFrame and metadata
        """
        logger.info("Preparing training data...")
        
        combined_data = []
        metadata = {
            'symbols': list(data.keys()),
            'date_ranges': {},
            'feature_stats': {}
        }
        
        for symbol, df in data.items():
            if df.empty:
                logger.warning(f"Empty data for {symbol}, skipping")
                continue
                
            # Add symbol identifier
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            
            # Store date range
            metadata['date_ranges'][symbol] = {
                'start': df.index.min(),
                'end': df.index.max(),
                'length': len(df)
            }
            
            combined_data.append(df_copy)
        
        if not combined_data:
            raise ValueError("No valid data found for any symbol")
        
        # Combine all data
        full_data = pd.concat(combined_data, axis=0)
        full_data = full_data.sort_index()
        
        logger.info(f"Combined data shape: {full_data.shape}")
        
        # Clean and prepare features
        processed_data = self._prepare_features(full_data)
        
        # Store feature statistics
        metadata['feature_stats'] = self._calculate_feature_stats(processed_data)
        
        return processed_data, metadata
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        df_processed = df.copy()
        
        # Remove rows with NaN values
        initial_length = len(df_processed)
        df_processed = df_processed.dropna()
        final_length = len(df_processed)
        
        if final_length < initial_length:
            logger.info(f"Removed {initial_length - final_length} rows with NaN values")
        
        # Define feature groups
        price_features = ['Open', 'High', 'Low', 'Close']
        volume_features = ['Volume']
        technical_features = [col for col in df_processed.columns 
                            if col not in price_features + volume_features + ['symbol']]
        
        # Store feature columns
        self.feature_columns = price_features + volume_features + technical_features
        
        # Create additional features
        df_processed = self._create_additional_features(df_processed)
        
        # Handle outliers
        df_processed = self._handle_outliers(df_processed)
        
        return df_processed
    
    def _create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features."""
        df_features = df.copy()
        
        try:
            # Price-based features
            df_features['price_change'] = df['Close'].pct_change()
            df_features['price_change_abs'] = df_features['price_change'].abs()
            
            # High-Low spread
            df_features['hl_spread'] = (df['High'] - df['Low']) / df['Close']
            
            # Open-Close spread
            df_features['oc_spread'] = (df['Close'] - df['Open']) / df['Open']
            
            # Volume features
            if 'Volume' in df.columns:
                df_features['volume_change'] = df['Volume'].pct_change()
                df_features['volume_price_ratio'] = df['Volume'] / df['Close']
            
            # Momentum features
            for window in [5, 10, 20]:
                df_features[f'momentum_{window}'] = df['Close'].pct_change(window)
                df_features[f'volatility_{window}'] = df['Close'].rolling(window).std() / df['Close'].rolling(window).mean()
            
            # Time-based features
            if hasattr(df.index, 'hour'):
                df_features['hour'] = df.index.hour
                df_features['day_of_week'] = df.index.dayofweek
                df_features['month'] = df.index.month
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                df_features[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df_features[f'volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
            
            # Rolling statistics
            for window in [10, 20, 50]:
                if len(df) > window:
                    df_features[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
                    df_features[f'close_std_{window}'] = df['Close'].rolling(window).std()
                    df_features[f'close_min_{window}'] = df['Close'].rolling(window).min()
                    df_features[f'close_max_{window}'] = df['Close'].rolling(window).max()
            
            logger.info(f"Created additional features. New shape: {df_features.shape}")
            
        except Exception as e:
            logger.error(f"Error creating additional features: {e}")
            return df
        
        return df_features
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in the data."""
        df_clean = df.copy()
        
        try:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['symbol']:  # Skip non-feature columns
                    continue
                    
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    
                elif method == 'zscore':
                    mean = df_clean[col].mean()
                    std = df_clean[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    # Cap outliers
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            return df
        
        return df_clean
    
    def _calculate_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate feature statistics."""
        stats = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['symbol']:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'quantiles': {
                        '25%': float(df[col].quantile(0.25)),
                        '50%': float(df[col].quantile(0.50)),
                        '75%': float(df[col].quantile(0.75))
                    }
                }
        
        return stats
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training.
        
        Args:
            df: Input DataFrame
            sequence_length: Length of input sequences
            target_col: Target column for prediction
            
        Returns:
            Features (X) and targets (y) arrays
        """
        try:
            # Prepare feature columns
            feature_cols = [col for col in df.columns if col not in ['symbol', target_col]]
            
            X, y = [], []
            
            # Group by symbol to maintain temporal order within each symbol
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_index()
                
                feature_data = symbol_data[feature_cols].values
                target_data = symbol_data[target_col].values
                
                # Create sequences
                for i in range(len(symbol_data) - sequence_length):
                    X.append(feature_data[i:i + sequence_length])
                    y.append(target_data[i + sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def fit_scalers(self, df: pd.DataFrame) -> Dict[str, object]:
        """Fit scalers on training data."""
        try:
            scaling_method = self.config.get('preprocessing', {}).get('scaling_method', 'standard')
            
            # Define feature groups for different scaling
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume']
            technical_features = [col for col in df.columns 
                                if col not in price_features + volume_features + ['symbol']]
            
            # Initialize scalers
            if scaling_method == 'standard':
                scaler_class = StandardScaler
            elif scaling_method == 'minmax':
                scaler_class = MinMaxScaler
            elif scaling_method == 'robust':
                scaler_class = RobustScaler
            else:
                scaler_class = StandardScaler
            
            # Fit scalers for each feature group
            for feature_group, features in [
                ('price', price_features),
                ('volume', volume_features), 
                ('technical', technical_features)
            ]:
                valid_features = [f for f in features if f in df.columns]
                if valid_features:
                    scaler = scaler_class()
                    scaler.fit(df[valid_features])
                    self.scalers[feature_group] = scaler
                    logger.info(f"Fitted {feature_group} scaler for {len(valid_features)} features")
            
            self.is_fitted = True
            return self.scalers
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")
            return {}
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        
        df_transformed = df.copy()
        
        try:
            # Define feature groups
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume']
            technical_features = [col for col in df.columns 
                                if col not in price_features + volume_features + ['symbol']]
            
            # Transform each feature group
            for feature_group, features in [
                ('price', price_features),
                ('volume', volume_features),
                ('technical', technical_features)
            ]:
                if feature_group in self.scalers:
                    valid_features = [f for f in features if f in df.columns]
                    if valid_features:
                        scaler = self.scalers[feature_group]
                        df_transformed[valid_features] = scaler.transform(df[valid_features])
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform data."""
        self.fit_scalers(df)
        return self.transform(df)
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers."""
        if not self.is_fitted:
            logger.warning("No fitted scalers to save")
            return
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'scalers': self.scalers,
                    'feature_columns': self.feature_columns,
                    'is_fitted': self.is_fitted
                }, f)
            logger.info(f"Scalers saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving scalers: {e}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.scalers = data['scalers']
            self.feature_columns = data['feature_columns']
            self.is_fitted = data['is_fitted']
            
            logger.info(f"Scalers loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")


def split_data(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets maintaining temporal order.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        
    Returns:
        Training, validation, and test DataFrames
    """
    try:
        df_sorted = df.sort_index()
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = df_sorted.iloc[:n_train]
        val_data = df_sorted.iloc[n_train:n_train + n_val]
        test_data = df_sorted.iloc[n_train + n_val:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return df, pd.DataFrame(), pd.DataFrame()


def create_features_for_env(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create features specifically for the trading environment.
    
    Args:
        df: Input DataFrame with OHLCV data
        config: Configuration dictionary
        
    Returns:
        DataFrame with features for RL environment
    """
    try:
        from .indicators import TechnicalIndicators
        
        # Calculate technical indicators
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, config)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Prepare features
        df_features = preprocessor._prepare_features(df_with_indicators)
        
        # Normalize if specified
        if config.get('environment', {}).get('normalize_observations', True):
            df_features = preprocessor.fit_transform(df_features)
        
        return df_features
        
    except Exception as e:
        logger.error(f"Error creating features for environment: {e}")
        return df