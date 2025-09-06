"""Q&A Chatbot package for course material processing."""

from .pdf_processor import PDFExtractor
from .utils import TextSplitter
from .embeddings import GeminiEmbeddings, VectorStore
from .chatbot import QAChatbot
from .config import Config

__version__ = "1.0.0"
__all__ = [
    "PDFExtractor",
    "TextSplitter", 
    "GeminiEmbeddings",
    "VectorStore",
    "QAChatbot",
    "Config"
]
