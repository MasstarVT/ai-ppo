"""
Evaluation module initialization.
"""

from .backtesting import Backtester, WalkForwardAnalysis

__all__ = [
    'Backtester',
    'WalkForwardAnalysis'
]