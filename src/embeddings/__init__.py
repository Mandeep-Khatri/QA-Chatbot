"""Embeddings module for vector storage and retrieval."""

from .gemini_embeddings import GeminiEmbeddings
from .vector_store import VectorStore

__all__ = ["GeminiEmbeddings", "VectorStore"]
