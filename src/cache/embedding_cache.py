"""Embedding cache for storing and retrieving text embeddings."""

import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.embedding_cache_file = os.path.join(cache_dir, "embeddings.pkl")
        self.embeddings = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load embeddings from cache file.
        
        Returns:
            Dictionary of cached embeddings
        """
        try:
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading embedding cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save embeddings to cache file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        text_hash = hash(text)
        return self.embeddings.get(text_hash)
    
    def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding vector to cache
        """
        text_hash = hash(text)
        self.embeddings[text_hash] = embedding
        self._save_cache()
    
    def get_batch_embeddings(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[str]]:
        """Get cached embeddings for batch of texts.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            Tuple of (cached_embeddings, texts_needing_embedding)
        """
        cached_embeddings = []
        texts_needing_embedding = []
        
        for text in texts:
            embedding = self.get_embedding(text)
            cached_embeddings.append(embedding)
            if embedding is None:
                texts_needing_embedding.append(text)
        
        return cached_embeddings, texts_needing_embedding
    
    def set_batch_embeddings(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Cache embeddings for batch of texts.
        
        Args:
            texts: List of texts
            embeddings: List of corresponding embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, embedding)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.embeddings = {}
        self._save_cache()
        logger.info("Cleared embedding cache")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings.
        
        Returns:
            Number of cached embeddings
        """
        return len(self.embeddings)
    
    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_embeddings": len(self.embeddings),
            "cache_file_size_mb": round(
                os.path.getsize(self.embedding_cache_file) / (1024 * 1024), 2
            ) if os.path.exists(self.embedding_cache_file) else 0
        }
