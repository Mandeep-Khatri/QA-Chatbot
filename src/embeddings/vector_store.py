"""Vector store implementation using ChromaDB for document retrieval."""

import os
import json
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from loguru import logger


class VectorStore:
    """Vector store for storing and retrieving document embeddings."""
    
    def __init__(self, 
                 persist_directory: str = "./data/vector_db",
                 collection_name: str = "course_documents"):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to store documents
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Course document embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[Dict[str, any]]) -> None:
        """
        Add text chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}_{chunk.get('chunk_id', i)}"
            ids.append(chunk_id)
            embeddings.append(chunk['embedding'])
            documents.append(chunk['text'])
            
            # Prepare metadata
            metadata = {
                'chunk_id': chunk.get('chunk_id', i),
                'char_count': chunk.get('char_count', 0),
                'token_count': chunk.get('token_count', 0),
                'embedding_model': chunk.get('embedding_model', 'unknown'),
                **chunk.get('metadata', {})
            }
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(self, 
               query_embedding: List[float], 
               n_results: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'id': results['ids'][0][i]
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def search_by_text(self, 
                      query_text: str, 
                      embedding_model,
                      n_results: int = 5,
                      filter_metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Search using text query (converts to embedding first).
        
        Args:
            query_text: Text query
            embedding_model: Model to generate query embedding
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        # Generate embedding for query
        query_embedding = embedding_model.embed_text(query_text)
        
        # Search using embedding
        return self.search(query_embedding, n_results, filter_metadata)
    
    def get_collection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=100)
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
            if sample_results['documents']:
                # Calculate average document length
                doc_lengths = [len(doc) for doc in sample_results['documents']]
                stats['avg_document_length'] = sum(doc_lengths) / len(doc_lengths)
                stats['min_document_length'] = min(doc_lengths)
                stats['max_document_length'] = max(doc_lengths)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Cleared {len(all_docs['ids'])} documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def export_collection(self, export_path: str) -> None:
        """
        Export collection data to JSON file.
        
        Args:
            export_path: Path to save the exported data
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            export_data = {
                'collection_name': self.collection_name,
                'total_documents': len(all_docs['ids']),
                'documents': []
            }
            
            for i in range(len(all_docs['ids'])):
                doc_data = {
                    'id': all_docs['ids'][i],
                    'document': all_docs['documents'][i],
                    'metadata': all_docs['metadatas'][i]
                }
                export_data['documents'].append(doc_data)
            
            # Save to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported collection to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            raise
