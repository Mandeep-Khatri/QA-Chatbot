"""PDF text extraction using PyMuPDF."""

import fitz  # PyMuPDF
import os
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF."""
    
    def __init__(self, preserve_layout: bool = True, include_images: bool = False):
        """
        Initialize PDF extractor.
        
        Args:
            preserve_layout: Whether to preserve text layout and formatting
            include_images: Whether to extract image information
        """
        self.preserve_layout = preserve_layout
        self.include_images = include_images
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            total_pages = len(doc)
            
            logger.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                if self.preserve_layout:
                    # Extract text with layout preservation
                    text = page.get_text("text")
                else:
                    # Extract plain text
                    text = page.get_text()
                
                # Clean up the text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text)
                    })
            
            doc.close()
            
            # Combine all text
            full_text = "\n\n".join([page['text'] for page in pages_text])
            
            result = {
                'file_path': pdf_path,
                'file_name': Path(pdf_path).name,
                'total_pages': total_pages,
                'pages_with_text': len(pages_text),
                'full_text': full_text,
                'pages': pages_text,
                'total_characters': len(full_text),
                'total_words': len(full_text.split())
            }
            
            logger.info(f"Extracted {result['total_words']} words from {result['pages_with_text']} pages")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_from_directory(self, directory_path: str, 
                             file_pattern: str = "*.pdf") -> List[Dict[str, any]]:
        """
        Extract text from all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            file_pattern: File pattern to match (default: *.pdf)
            
        Returns:
            List of dictionaries containing extracted text from each PDF
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pdf_files = list(directory.glob(file_pattern))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.extract_text_from_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(results)} out of {len(pdf_files)} PDF files")
        return results
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) > 3:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            result = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'page_count': len(doc),
                'file_size': os.path.getsize(pdf_path)
            }
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            return {}
