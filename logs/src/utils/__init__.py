"""
Utils module initialization.
"""

from .config import ConfigManager, setup_logging, create_default_config, validate_paths, merge_configs
from .helpers import (
    set_seed, format_currency, format_percentage, calculate_performance_metrics,
    Timer, ProgressTracker, memory_usage_mb, log_system_info, validate_data_quality
)

try:
    from .training_manager import BackgroundTrainingManager, TrainingProgress, NetworkAnalyzer, training_manager
    _training_manager_available = True
except ImportError:
    # Training manager requires torch/numpy, might not be available
    _training_manager_available = False

__all__ = [
    'ConfigManager',
    'setup_logging', 
    'create_default_config',
    'validate_paths',
    'merge_configs',
    'set_seed',
    'format_currency',
    'format_percentage', 
    'calculate_performance_metrics',
    'Timer',
    'ProgressTracker',
    'memory_usage_mb',
    'log_system_info',
    'validate_data_quality'
]

if _training_manager_available:
    __all__.extend([
        'BackgroundTrainingManager',
        'TrainingProgress', 
        'NetworkAnalyzer',
        'training_manager'
    ])