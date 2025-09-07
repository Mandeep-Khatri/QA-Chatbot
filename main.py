"""Mandy's Q&A Chatbot - Main application entry point."""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.pdf_processor import PDFExtractor
from src.utils import TextSplitter
from src.embeddings import GeminiEmbeddings, VectorStore
from src.chatbot import QAChatbot
from loguru import logger


class ChatbotApp:
    """Main application class for the Q&A chatbot."""
    
    def __init__(self):
        """Initialize the chatbot application."""
        # Setup configuration
        Config.setup_logging()
        
        if not Config.validate_config():
            logger.error("Configuration validation failed. Please check your environment variables.")
            sys.exit(1)
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_splitter = TextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embedding_model = GeminiEmbeddings(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        self.vector_store = VectorStore(
            persist_directory=Config.VECTOR_DB_PATH,
            collection_name=Config.COLLECTION_NAME
        )
        self.chatbot = QAChatbot(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )
        
        logger.info("Chatbot application initialized successfully")
    
    def process_pdfs(self, pdf_directory: str) -> None:
        """
        Process PDF files and create embeddings.
        
        Args:
            pdf_directory: Directory containing PDF files
        """
        logger.info(f"Processing PDFs from directory: {pdf_directory}")
        
        # Extract text from PDFs
        pdf_results = self.pdf_extractor.extract_from_directory(pdf_directory)
        
        if not pdf_results:
            logger.warning("No PDF files found or processed")
            return
        
        # Split text into chunks
        all_chunks = self.text_splitter.split_multiple_pdfs(pdf_results)
        
        # Generate embeddings
        chunks_with_embeddings = self.embedding_model.embed_chunks(
            all_chunks, 
            batch_size=Config.BATCH_SIZE
        )
        
        # Store in vector database
        self.vector_store.add_chunks(chunks_with_embeddings)
        
        # Print statistics
        stats = self.text_splitter.get_chunk_statistics(chunks_with_embeddings)
        logger.info(f"Processing complete. Statistics: {stats}")
        
        vector_stats = self.vector_store.get_collection_stats()
        logger.info(f"Vector store statistics: {vector_stats}")
    
    def interactive_chat(self) -> None:
        """Start interactive chat session."""
        logger.info("Starting interactive chat session. Type 'quit' to exit.")
        
        while True:
            try:
                question = input("\n🤖 Ask a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    logger.info("Chat session ended by user")
                    break
                
                if not question:
                    continue
                
                # Get answer
                response = self.chatbot.ask_question(question)
                
                print(f"\n📚 Answer: {response['answer']}")
                
                if 'sources' in response and response['sources']:
                    print(f"\n📖 Sources ({response['num_sources']}):")
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"  {i}. {source['content']}")
                        if 'page_number' in source['metadata']:
                            print(f"     Page: {source['metadata']['page_number']}")
                
            except KeyboardInterrupt:
                logger.info("Chat session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in chat session: {e}")
                print(f"❌ Error: {e}")
    
    def search_documents(self, query: str, n_results: int = 5) -> None:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
        """
        logger.info(f"Searching for: {query}")
        
        results = self.chatbot.search_documents(query, n_results)
        
        if not results:
            print("No relevant documents found.")
            return
        
        print(f"\n🔍 Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Source: {result['source']}")
            print(f"   Content: {result['content'][:200]}...")
    
    def get_system_info(self) -> None:
        """Display system information."""
        model_info = self.chatbot.get_model_info()
        vector_stats = self.vector_store.get_collection_stats()
        
        print("\n📊 System Information:")
        print(f"Model: {model_info['model_name']}")
        print(f"Temperature: {model_info['temperature']}")
        print(f"Max Tokens: {model_info['max_tokens']}")
        print(f"Embedding Model: {model_info['embedding_model']}")
        print(f"Vector Store: {model_info['vector_store_collection']}")
        print(f"Total Documents: {vector_stats.get('total_documents', 0)}")
        print(f"Average Document Length: {vector_stats.get('avg_document_length', 0):.0f} chars")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Q&A Chatbot for Course Materials")
    parser.add_argument("--process-pdfs", type=str, help="Process PDFs from directory")
    parser.add_argument("--search", type=str, help="Search for documents")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--results", type=int, default=5, help="Number of search results")
    
    args = parser.parse_args()
    
    try:
        app = ChatbotApp()
        
        if args.process_pdfs:
            app.process_pdfs(args.process_pdfs)
        elif args.search:
            app.search_documents(args.search, args.results)
        elif args.chat:
            app.interactive_chat()
        elif args.info:
            app.get_system_info()
        else:
            # Default: start interactive chat
            app.interactive_chat()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
