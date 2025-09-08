"""Response cache for storing and retrieving chatbot responses."""

import os
import pickle
import hashlib
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache for chatbot responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize response cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.response_cache_file = os.path.join(cache_dir, "responses.pkl")
        self.responses = self._load_cache()
        self.cache_ttl = timedelta(hours=6)  # Cache responses for 6 hours
    
    def _load_cache(self) -> dict:
        """Load responses from cache file.
        
        Returns:
            Dictionary of cached responses
        """
        try:
            if os.path.exists(self.response_cache_file):
                with open(self.response_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading response cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save responses to cache file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.response_cache_file, 'wb') as f:
                pickle.dump(self.responses, f)
        except Exception as e:
            logger.error(f"Error saving response cache: {e}")
    
    def _get_query_hash(self, query: str, context: str = "") -> str:
        """Generate hash for query and context.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            MD5 hash of query and context
        """
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid.
        
        Args:
            cache_entry: Cache entry with timestamp
            
        Returns:
            True if cache is valid, False otherwise
        """
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < self.cache_ttl
    
    def get_response(self, query: str, context: str = "") -> Optional[str]:
        """Get cached response for query.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Cached response or None if not found/expired
        """
        query_hash = self._get_query_hash(query, context)
        cache_entry = self.responses.get(query_hash)
        
        if cache_entry and self._is_cache_valid(cache_entry):
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cache_entry['response']
        
        # Remove expired entry
        if cache_entry:
            del self.responses[query_hash]
            self._save_cache()
        
        return None
    
    def set_response(self, query: str, response: str, context: str = "", metadata: Dict[str, Any] = None) -> None:
        """Cache response for query.
        
        Args:
            query: User query
            response: Bot response
            context: Additional context
            metadata: Additional metadata to store
        """
        query_hash = self._get_query_hash(query, context)
        
        cache_entry = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'context': context,
            'metadata': metadata or {}
        }
        
        self.responses[query_hash] = cache_entry
        self._save_cache()
        logger.info(f"Cached response for query: {query[:50]}...")
    
    def clear_expired_responses(self) -> int:
        """Clear expired responses from cache.
        
        Returns:
            Number of expired responses removed
        """
        expired_keys = []
        for key, entry in self.responses.items():
            if not self._is_cache_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.responses[key]
        
        if expired_keys:
            self._save_cache()
            logger.info(f"Removed {len(expired_keys)} expired responses")
        
        return len(expired_keys)
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.responses = {}
        self._save_cache()
        logger.info("Cleared response cache")
    
    def get_cache_stats(self) -> dict:
        """Get response cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_responses = len(self.responses)
        valid_responses = sum(1 for entry in self.responses.values() if self._is_cache_valid(entry))
        expired_responses = total_responses - valid_responses
        
        return {
            "total_responses": total_responses,
            "valid_responses": valid_responses,
            "expired_responses": expired_responses,
            "cache_file_size_mb": round(
                os.path.getsize(self.response_cache_file) / (1024 * 1024), 2
            ) if os.path.exists(self.response_cache_file) else 0,
            "cache_ttl_hours": self.cache_ttl.total_seconds() / 3600
        }
    
    def get_recent_queries(self, limit: int = 10) -> list:
        """Get recent queries from cache.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of recent queries
        """
        sorted_entries = sorted(
            self.responses.values(),
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return [
            {
                'query': entry.get('query', ''),
                'timestamp': entry.get('timestamp', ''),
                'response_preview': entry.get('response', '')[:100] + '...' if len(entry.get('response', '')) > 100 else entry.get('response', '')
            }
            for entry in sorted_entries[:limit]
        ]
