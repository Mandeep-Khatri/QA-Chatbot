"""Cache manager for the Q&A chatbot."""

import hashlib
import json
import os
import pickle
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for embeddings, responses, and other data."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.embedding_cache = EmbeddingCache(cache_dir)
        self.response_cache = ResponseCache(cache_dir)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache configuration
        self.max_cache_size = 1000  # Maximum number of cached items
        self.cache_ttl = timedelta(hours=24)  # Time to live for cache items
        
    def get_cache_key(self, data: str) -> str:
        """Generate a cache key from data.
        
        Args:
            data: Data to generate key from
            
        Returns:
            MD5 hash of the data
        """
        return hashlib.md5(data.encode('utf-8')).hexdigest()
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file is still valid.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_file):
            return False
            
        # Check file modification time
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - file_time < self.cache_ttl
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache files."""
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath) and not self.is_cache_valid(filepath):
                    os.remove(filepath)
                    logger.info(f"Removed expired cache file: {filename}")
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            files = os.listdir(self.cache_dir)
            total_files = len(files)
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in files if os.path.isfile(os.path.join(self.cache_dir, f))
            )
            
            return {
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": self.cache_dir,
                "max_cache_size": self.max_cache_size,
                "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def clear_all_cache(self) -> None:
        """Clear all cache files."""
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            logger.info("Cleared all cache files")
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
