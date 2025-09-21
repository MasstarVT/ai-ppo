"""
Utility functions and helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    if returns.std() == 0:
        return 0
    
    excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns / returns.std() * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio."""
    max_dd = calculate_max_drawdown(returns)
    if max_dd == 0:
        return np.inf
    
    annual_return = (1 + returns.mean()) ** 252 - 1
    return annual_return / abs(max_dd)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf
    
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk."""
    return returns.quantile(confidence_level)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk."""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if len(returns) == 0:
        return {}
    
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annual_return': (1 + returns.mean()) ** 252 - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'var_5': calculate_var(returns, 0.05),
        'cvar_5': calculate_cvar(returns, 0.05),
        'win_rate': len(returns[returns > 0]) / len(returns),
        'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
        'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf
    }
    
    return metrics


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"{self.name} completed in {duration:.2f} seconds")


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, name: str = "Progress", log_interval: int = 100):
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "N/A"
        
        logger.info(f"{self.name}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta_str}")


def create_date_range(start_date: str, end_date: str, freq: str = 'D') -> pd.DatetimeIndex:
    """Create date range."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def filter_trading_hours(df: pd.DataFrame, start_hour: int = 9, end_hour: int = 16) -> pd.DataFrame:
    """Filter data to trading hours only."""
    if not hasattr(df.index, 'hour'):
        return df
    
    return df.between_time(f"{start_hour:02d}:00", f"{end_hour:02d}:00")


def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV data to different frequency."""
    try:
        resampled = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
        
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        return df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    if denominator == 0:
        return default
    return numerator / denominator


def rolling_apply(series: pd.Series, window: int, func: callable, **kwargs) -> pd.Series:
    """Apply function to rolling window."""
    try:
        return series.rolling(window=window).apply(func, **kwargs)
    except Exception as e:
        logger.error(f"Error applying rolling function: {e}")
        return series


def load_pickle(filepath: str) -> Any:
    """Load pickle file safely."""
    try:
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {filepath}: {e}")
        return None


def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file."""
    try:
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        
        logger.info(f"Object saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving pickle file {filepath}: {e}")


def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2  # MB
            }
    except ImportError:
        pass
    
    return {'allocated': 0, 'cached': 0}


def log_system_info():
    """Log system information."""
    try:
        import platform
        import psutil
        
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        try:
            import torch
            logger.info(f"PyTorch: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        except ImportError:
            logger.info("PyTorch not available")
            
    except Exception as e:
        logger.error(f"Error logging system info: {e}")


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return report."""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': 0,
        'date_gaps': [],
        'outliers': {}
    }
    
    try:
        # Missing values
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percentage': float(missing / len(df) * 100)
                }
        
        # Duplicate rows
        report['duplicate_rows'] = int(df.duplicated().sum())
        
        # Date gaps (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
                missing_dates = full_range.difference(df.index)
                report['date_gaps'] = [str(date) for date in missing_dates[:10]]  # First 10 gaps
        
        # Outliers using IQR method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': float(len(outliers) / len(df) * 100)
                }
        
        logger.info(f"Data quality report: {report['total_rows']} rows, {len(report['missing_values'])} columns with missing values")
        
    except Exception as e:
        logger.error(f"Error validating data quality: {e}")
    
    return report