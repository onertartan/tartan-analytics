"""
Utility package initializer.

This module exposes commonly used utility classes and functions
so consumers can import them directly via:

    from utils import SessionAdapter
"""

from .session_adapter import SessionAdapter

__all__ = [
    "SessionAdapter",
]
