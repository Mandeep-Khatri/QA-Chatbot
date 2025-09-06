"""Q&A Chatbot implementation using LangChain and Gemini 1.5 Pro."""

import os
from typing import List, Dict, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from loguru import logger

from ..embeddings import GeminiEmbeddings, VectorStore


class QAChatbot:
    """Q&A Chatbot using Gemini 1.5 Pro and vector retrieval."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-pro",
                 temperature: float = 0.1,
                 max_tokens: int = 8192,
                 vector_store: Optional[VectorStore] = None,
                 embedding_model: Optional[GeminiEmbeddings] = None):
        """
        Initialize Q&A chatbot.
        
        Args:
            api_key: Gemini API key
            model_name: Name of the Gemini model to use
            temperature: Temperature for text generation
            max_tokens: Maximum tokens for response
            vector_store: Vector store for document retrieval
            embedding_model: Embedding model for queries
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Gemini LLM
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        
        # Initialize embedding model
        self.embedding_model = embedding_model or GeminiEmbeddings(api_key=api_key)
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStore()
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info(f"Initialized Q&A chatbot with model: {model_name}")
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the retrieval QA chain."""
        
        # Custom prompt template for better responses
        prompt_template = """
        You are a helpful AI assistant that answers questions based on course materials and documents.
        
        Context from course materials:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer the question based primarily on the provided context
        2. If the context doesn't contain enough information, say so clearly
        3. Be accurate and cite specific information from the context when possible
        4. If you're unsure about something, express that uncertainty
        5. Keep your answer concise but comprehensive
        6. Use clear, academic language appropriate for course materials
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._create_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    
    def _create_retriever(self):
        """Create a retriever from the vector store."""
        from langchain.vectorstores import Chroma
        from langchain.embeddings import GoogleGenerativeAIEmbeddings
        
        # Create LangChain compatible embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        
        # Create Chroma retriever
        vectorstore = Chroma(
            client=self.vector_store.client,
            collection_name=self.vector_store.collection_name,
            embedding_function=embeddings
        )
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def ask_question(self, question: str, include_sources: bool = True) -> Dict[str, any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            
            response = {
                'question': question,
                'answer': result['result'],
                'model_used': self.model_name
            }
            
            if include_sources and 'source_documents' in result:
                sources = []
                for doc in result['source_documents']:
                    source_info = {
                        'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        'metadata': doc.metadata
                    }
                    sources.append(source_info)
                
                response['sources'] = sources
                response['num_sources'] = len(sources)
            
            logger.info(f"Generated answer with {response.get('num_sources', 0)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'error': str(e)
            }
    
    def ask_question_with_context(self, 
                                 question: str, 
                                 context_documents: List[str],
                                 include_sources: bool = True) -> Dict[str, any]:
        """
        Ask a question with additional context documents.
        
        Args:
            question: The question to ask
            context_documents: Additional context documents
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer and context
        """
        try:
            # Combine context documents
            context = "\n\n".join(context_documents)
            
            # Create a prompt with the context
            prompt = f"""
            Based on the following context documents, please answer the question.
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context provided.
            """
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            result = {
                'question': question,
                'answer': response.content,
                'context_used': len(context_documents),
                'model_used': self.model_name
            }
            
            if include_sources:
                result['context_documents'] = context_documents
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question with context: {e}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'error': str(e)
            }
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, any]]:
        """
        Search for relevant documents without generating an answer.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, n_results)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'content': result['document'],
                    'metadata': result['metadata'],
                    'similarity_score': 1 - result['distance'],  # Convert distance to similarity
                    'source': result['metadata'].get('source_file', 'Unknown')
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history (placeholder for future implementation).
        
        Returns:
            List of conversation turns
        """
        # This would be implemented with a conversation memory system
        return []
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        # This would be implemented with a conversation memory system
        logger.info("Conversation history cleared")
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'embedding_model': self.embedding_model.model_name,
            'vector_store_collection': self.vector_store.collection_name
        }
