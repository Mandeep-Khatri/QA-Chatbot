"""Gemini embeddings implementation for text vectorization."""

import os
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from loguru import logger


class GeminiEmbeddings:
    """Generate embeddings using Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-004"):
        """
        Initialize Gemini embeddings.
        
        Args:
            api_key: Gemini API key (if None, will use environment variable)
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        
        # Set up API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            genai.configure(api_key=api_key)
        
        # Initialize the embedding model
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini embeddings with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            # Use the embed_content method
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings
        """
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Generating embeddings for {total_texts} texts in batches of {batch_size}")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.embed_text(text)
                    batch_embeddings.append(embedding)
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error embedding text batch {i//batch_size + 1}: {e}")
                    # Add zero vector as placeholder
                    batch_embeddings.append([0.0] * 768)  # Default embedding size
            
            embeddings.extend(batch_embeddings)
            
            # Progress logging
            processed = min(i + batch_size, total_texts)
            logger.info(f"Processed {processed}/{total_texts} texts")
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, any]], batch_size: int = 10) -> List[Dict[str, any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            batch_size: Batch size for processing
            
        Returns:
            List of chunks with embeddings added
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            chunk['embedding_model'] = self.model_name
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from the model.
        
        Returns:
            Embedding dimension
        """
        # Test with a small text to get embedding dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
