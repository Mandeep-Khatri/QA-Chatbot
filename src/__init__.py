"""Q&A Chatbot package for course material processing."""

# Import only the components that don't require PyMuPDF
try:
    from .pdf_processor import PDFExtractor
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from .utils import TextSplitter
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
from .embeddings import GeminiEmbeddings, VectorStore
from .chatbot import QAChatbot
from .config import Config

__version__ = "1.0.0"
__all__ = [
    "GeminiEmbeddings",
    "VectorStore",
    "QAChatbot",
    "Config"
]

if PDF_AVAILABLE:
    __all__.append("PDFExtractor")

if UTILS_AVAILABLE:
    __all__.append("TextSplitter")
