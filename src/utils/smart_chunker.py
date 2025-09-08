"""Smart text chunking for better document processing."""

import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SmartChunker:
    """Advanced text chunking with semantic awareness."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize smart chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for different content types
        self.section_patterns = [
            r'^#{1,6}\s+.+',  # Markdown headers
            r'^\d+\.\s+.+',   # Numbered lists
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            r'^Chapter\s+\d+',   # Chapter headers
            r'^Section\s+\d+',   # Section headers
        ]
        
        # Sentence boundary patterns
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        
        # Paragraph boundary patterns
        self.paragraph_boundaries = r'\n\s*\n'
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text using smart strategies.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Try different chunking strategies
        chunks = self._semantic_chunking(cleaned_text)
        
        if not chunks:
            chunks = self._paragraph_chunking(cleaned_text)
        
        if not chunks:
            chunks = self._sentence_chunking(cleaned_text)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['metadata'] = metadata or {}
            chunk['chunk_size'] = len(chunk['text'])
            chunk['word_count'] = len(chunk['text'].split())
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def _semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text based on semantic boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section boundary
            if self._is_section_boundary(line):
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.chunk_size * 0.5:  # Minimum chunk size
                        chunks.append({
                            'text': chunk_text,
                            'type': 'semantic',
                            'boundary': 'section'
                        })
                    current_chunk = []
                    current_size = 0
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # Check if chunk is large enough
            if current_size >= self.chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'semantic',
                    'boundary': 'size'
                })
                current_chunk = []
                current_size = 0
        
        # Add remaining content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.chunk_size * 0.3:  # Minimum chunk size
                chunks.append({
                    'text': chunk_text,
                    'type': 'semantic',
                    'boundary': 'end'
                })
        
        return chunks
    
    def _paragraph_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of paragraph chunks
        """
        paragraphs = re.split(self.paragraph_boundaries, text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_size + len(paragraph) > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'paragraph',
                    'boundary': 'size'
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(paragraph)
            current_size += len(paragraph)
        
        # Add remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'type': 'paragraph',
                'boundary': 'end'
            })
        
        return chunks
    
    def _sentence_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by sentences as fallback.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of sentence chunks
        """
        sentences = re.split(self.sentence_endings, text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'text': chunk_text,
                    'type': 'sentence',
                    'boundary': 'size'
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        # Add remaining content
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                'text': chunk_text,
                'type': 'sentence',
                'boundary': 'end'
            })
        
        return chunks
    
    def _is_section_boundary(self, line: str) -> bool:
        """Check if line is a section boundary.
        
        Args:
            line: Line to check
            
        Returns:
            True if line is a section boundary
        """
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        return False
    
    def add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add overlap between chunks for better context.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            
            # Add overlap from previous chunk
            if i > 0:
                prev_text = chunks[i-1]['text']
                overlap_text = self._get_overlap_text(prev_text, self.chunk_overlap)
                if overlap_text:
                    text = overlap_text + '\n' + text
            
            # Add overlap to next chunk
            if i < len(chunks) - 1:
                next_text = chunks[i+1]['text']
                overlap_text = self._get_overlap_text(next_text, self.chunk_overlap)
                if overlap_text:
                    text = text + '\n' + overlap_text
            
            overlapped_chunk = chunk.copy()
            overlapped_chunk['text'] = text
            overlapped_chunk['has_overlap'] = True
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk.
        
        Args:
            text: Text to get overlap from
            overlap_size: Size of overlap
            
        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary
        sentences = re.split(self.sentence_endings, text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= overlap_size:
                overlap_text = sentence + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        sizes = [chunk['chunk_size'] for chunk in chunks]
        word_counts = [chunk['word_count'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'chunk_types': {
                chunk_type: sum(1 for chunk in chunks if chunk.get('type') == chunk_type)
                for chunk_type in set(chunk.get('type', 'unknown') for chunk in chunks)
            }
        }
