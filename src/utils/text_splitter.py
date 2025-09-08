"""Text splitting and chunking utilities."""

from typing import List, Dict, Optional
import logging

# Try to import optional dependencies
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextSplitter:
    """Advanced text splitting with multiple strategies."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tokenizer for accurate token counting
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base encoding
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None
            logger.warning("tiktoken not available, using character-based estimation")
        
        # Initialize LangChain text splitter if available
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            self.text_splitter = None
            logger.warning("langchain not available, using simple text splitting")
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of text chunks with metadata
        """
        if not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        # Split text using available method
        if self.text_splitter:
            chunks = self.text_splitter.split_text(text)
        else:
            # Simple fallback splitting
            chunks = self._simple_split_text(text)
        
        # Add metadata to each chunk
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'chunk_id': i,
                'text': chunk,
                'char_count': len(chunk),
                'token_count': self._count_tokens(chunk),
                'metadata': metadata or {}
            }
            result.append(chunk_data)
        
        logger.info(f"Split text into {len(result)} chunks")
        return result
    
    def _simple_split_text(self, text: str) -> List[str]:
        """Simple text splitting fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size // 2, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # Look for paragraph break
                    for i in range(end, max(start + self.chunk_size // 2, start), -1):
                        if text[i] == '\n':
                            end = i + 1
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def split_pdf_pages(self, pdf_data: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split PDF pages into chunks while preserving page information.
        
        Args:
            pdf_data: PDF extraction result from PDFExtractor
            
        Returns:
            List of chunks with page metadata
        """
        all_chunks = []
        
        for page in pdf_data.get('pages', []):
            page_text = page['text']
            page_metadata = {
                'source_file': pdf_data['file_name'],
                'page_number': page['page_number'],
                'file_path': pdf_data['file_path']
            }
            
            # Split page text into chunks
            page_chunks = self.split_text(page_text, page_metadata)
            all_chunks.extend(page_chunks)
        
        logger.info(f"Split {pdf_data['file_name']} into {len(all_chunks)} chunks")
        return all_chunks
    
    def split_multiple_pdfs(self, pdf_results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split multiple PDFs into chunks.
        
        Args:
            pdf_results: List of PDF extraction results
            
        Returns:
            List of all chunks from all PDFs
        """
        all_chunks = []
        
        for pdf_data in pdf_results:
            pdf_chunks = self.split_pdf_pages(pdf_data)
            all_chunks.extend(pdf_chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Error counting tokens: {e}")
                # Fallback to character-based estimation
                return len(text) // 4
        else:
            # Character-based estimation (rough approximation)
            return len(text) // 4
    
    def get_chunk_statistics(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        char_counts = [chunk['char_count'] for chunk in chunks]
        token_counts = [chunk['token_count'] for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(char_counts),
            'total_tokens': sum(token_counts),
            'avg_chars_per_chunk': sum(char_counts) / len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'min_chars': min(char_counts),
            'max_chars': max(char_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts)
        }
        
        return stats
    
    def filter_chunks_by_size(self, 
                            chunks: List[Dict[str, any]], 
                            min_chars: int = 50,
                            max_chars: int = 2000) -> List[Dict[str, any]]:
        """
        Filter chunks by character count.
        
        Args:
            chunks: List of text chunks
            min_chars: Minimum characters per chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            Filtered list of chunks
        """
        filtered = [
            chunk for chunk in chunks 
            if min_chars <= chunk['char_count'] <= max_chars
        ]
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} chunks")
        return filtered
