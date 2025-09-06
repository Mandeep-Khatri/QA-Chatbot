"""Streamlit web interface for the Q&A chatbot."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.chatbot import QAChatbot
from src.embeddings import GeminiEmbeddings, VectorStore
from loguru import logger


@st.cache_resource
def initialize_chatbot():
    """Initialize and cache the chatbot."""
    try:
        # Setup configuration
        Config.setup_logging()
        
        if not Config.validate_config():
            st.error("Configuration validation failed. Please check your environment variables.")
            st.stop()
        
        # Initialize components
        embedding_model = GeminiEmbeddings(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        vector_store = VectorStore(
            persist_directory=Config.VECTOR_DB_PATH,
            collection_name=Config.COLLECTION_NAME
        )
        chatbot = QAChatbot(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        return chatbot, vector_store
        
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Q&A Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Q&A Chatbot for Course Materials")
    st.markdown("Ask questions about your course materials using AI-powered search and retrieval.")
    
    # Initialize chatbot
    chatbot, vector_store = initialize_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        
        # Get system info
        model_info = chatbot.get_model_info()
        vector_stats = vector_store.get_collection_stats()
        
        st.metric("Total Documents", vector_stats.get('total_documents', 0))
        st.metric("Model", model_info['model_name'])
        st.metric("Temperature", model_info['temperature'])
        
        st.header("🔧 Settings")
        
        # Temperature slider
        temperature = st.slider(
            "Response Temperature",
            min_value=0.0,
            max_value=1.0,
            value=model_info['temperature'],
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
        
        # Number of sources
        num_sources = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of source documents to retrieve"
        )
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        st.write(f"**Source {i}:**")
                        st.write(source['content'])
                        if 'page_number' in source['metadata']:
                            st.caption(f"Page: {source['metadata']['page_number']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your course materials..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chatbot.ask_question(prompt, include_sources=True)
                    
                    # Display answer
                    st.markdown(response['answer'])
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": response['answer']
                    }
                    
                    if 'sources' in response:
                        assistant_message["sources"] = response['sources']
                    
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Search functionality
    st.header("🔍 Document Search")
    
    search_query = st.text_input("Search for specific information in documents:")
    
    if st.button("Search") and search_query:
        with st.spinner("Searching documents..."):
            try:
                results = chatbot.search_documents(search_query, num_sources)
                
                if results:
                    st.write(f"Found {len(results)} relevant documents:")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - Similarity: {result['similarity_score']:.3f}"):
                            st.write(f"**Source:** {result['source']}")
                            st.write(f"**Content:** {result['content']}")
                else:
                    st.write("No relevant documents found.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
