"""Utility functions for text processing and chunking."""

try:
    from .text_splitter import TextSplitter
    TEXT_SPLITTER_AVAILABLE = True
except ImportError as e:
    TEXT_SPLITTER_AVAILABLE = False
    print(f"Warning: TextSplitter not available: {e}")

__all__ = []

if TEXT_SPLITTER_AVAILABLE:
    __all__.append("TextSplitter")
