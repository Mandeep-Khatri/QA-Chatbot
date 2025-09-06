"""Configuration management for the Q&A chatbot."""

import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the chatbot application."""
    
    # Google Cloud and Gemini Configuration
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    GOOGLE_CLOUD_PROJECT: str = os.getenv('GOOGLE_CLOUD_PROJECT', '')
    GOOGLE_CLOUD_LOCATION: str = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Model Configuration
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'gemini-1.5-pro')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.1'))
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', '8192'))
    
    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv('VECTOR_DB_PATH', './data/vector_db')
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'course_documents')
    
    # Data Paths
    DATA_DIR: str = os.getenv('DATA_DIR', './data')
    PDF_DIR: str = os.getenv('PDF_DIR', './data/pdfs')
    LOGS_DIR: str = os.getenv('LOGS_DIR', './logs')
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', './logs/chatbot.log')
    
    # Application Configuration
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '10'))
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY: float = float(os.getenv('RETRY_DELAY', '1.0'))
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_vars = [
            'GEMINI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        # Create necessary directories
        cls._create_directories()
        
        return True
    
    @classmethod
    def _create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.PDF_DIR,
            cls.VECTOR_DB_PATH,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get model configuration dictionary.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'model_name': cls.MODEL_NAME,
            'embedding_model': cls.EMBEDDING_MODEL,
            'temperature': cls.TEMPERATURE,
            'max_tokens': cls.MAX_TOKENS,
            'chunk_size': cls.CHUNK_SIZE,
            'chunk_overlap': cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def get_vector_store_config(cls) -> dict:
        """
        Get vector store configuration dictionary.
        
        Returns:
            Dictionary with vector store configuration
        """
        return {
            'persist_directory': cls.VECTOR_DB_PATH,
            'collection_name': cls.COLLECTION_NAME
        }
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration."""
        # Remove default logger
        logger.remove()
        
        # Add console logging
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=cls.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file logging
        logger.add(
            sink=cls.LOG_FILE,
            level=cls.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
        
        logger.info("Logging configured successfully")
