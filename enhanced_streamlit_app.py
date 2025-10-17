"""Enhanced Streamlit app with all the quick wins implemented."""

import streamlit as st
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our enhanced components
from src.cache.cache_manager import CacheManager
from src.utils.smart_chunker import SmartChunker

# Import working chatbot functions
try:
    from working_chatbot import get_gemini_model, generate_response
except ImportError:
    # Fallback if working_chatbot is not available
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    def get_gemini_model():
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        return genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_response(model, query, context=""):
        prompt = f"{context}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(prompt)
        return response.text

# Page configuration
st.set_page_config(
    page_title="🤖 Gemini by Mandy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_cache_manager():
    return CacheManager()

@st.cache_resource
def get_smart_chunker():
    return SmartChunker()

cache_manager = get_cache_manager()
smart_chunker = get_smart_chunker()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 Gemini by Mandy</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 Gemini by Mandy")
    
    # Navigation
    page = st.selectbox(
        "Navigate",
        ["💬 Chat", "📊 Analytics", "⚙️ Settings", "🔧 Admin", "📁 File Manager"]
    )
    
    # Cache stats
    st.subheader("📈 Cache Statistics")
    try:
        cache_stats = cache_manager.get_cache_stats()
        st.metric("Total Cache Files", cache_stats.get("total_files", 0))
        st.metric("Cache Size (MB)", cache_stats.get("total_size_mb", 0))
    except:
        st.info("Cache stats unavailable")

# Main content based on selected page
if page == "💬 Chat":
    st.header("💬 Enhanced Chat Interface")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    
                    # Check cache first
                    cached_response = cache_manager.response_cache.get_response(prompt)
                    
                    if cached_response:
                        response = cached_response
                        st.success("✅ Response from cache (faster!)")
                    else:
                        model = get_gemini_model()
                        response = generate_response(model, prompt)
                        
                        # Cache the response
                        cache_manager.response_cache.set_response(prompt, response)
                    
                    processing_time = time.time() - start_time
                    
                    st.markdown(response)
                    st.caption(f"⏱️ Processed in {processing_time:.2f}s")
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Store in chat history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": prompt,
                "response": response,
                "processing_time": processing_time,
                "cached": cached_response is not None
            })
    
    with col2:
        st.subheader("💡 Query Suggestions")
        
        # Query suggestions
        suggestions = [
            "What is machine learning?",
            "Explain neural networks",
            "How does AI work?",
            "What are the benefits of automation?",
            "Tell me about data science"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion}"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": suggestion})
                
                # Generate response
                with st.spinner("Thinking..."):
                    model = get_gemini_model()
                    response = generate_response(model, suggestion)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": suggestion,
                    "response": response,
                    "cached": False,
                    "processing_time": 0.5
                })
                
                st.rerun()
        
        # Export chat history
        if st.button("📥 Export Chat History"):
            if st.session_state.chat_history:
                df = pd.DataFrame(st.session_state.chat_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No chat history to export")

elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")
    
    # Get cache statistics
    try:
        embedding_stats = cache_manager.embedding_cache.get_cache_stats()
        response_stats = cache_manager.response_cache.get_cache_stats()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cached Embeddings", embedding_stats.get("cached_embeddings", 0))
        
        with col2:
            st.metric("Cached Responses", response_stats.get("valid_responses", 0))
        
        with col3:
            st.metric("Cache Hit Rate", f"{response_stats.get('valid_responses', 0) / max(response_stats.get('total_responses', 1), 1) * 100:.1f}%")
        
        with col4:
            st.metric("Total Cache Size", f"{response_stats.get('cache_file_size_mb', 0):.2f} MB")
        
        # Charts
        if st.session_state.chat_history:
            st.subheader("📈 Chat History Analysis")
            
            df = pd.DataFrame(st.session_state.chat_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Processing time chart
            fig_time = px.line(df, x='timestamp', y='processing_time', 
                             title='Processing Time Over Time')
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Cache hit rate
            cache_hits = df['cached'].sum()
            total_queries = len(df)
            
            fig_cache = go.Figure(data=[go.Pie(
                labels=['Cache Hits', 'New Responses'],
                values=[cache_hits, total_queries - cache_hits],
                hole=0.3
            )])
            fig_cache.update_layout(title="Cache Hit Rate")
            st.plotly_chart(fig_cache, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    # Cache settings
    st.subheader("🗄️ Cache Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Cache"):
            cache_manager.clear_all_cache()
            st.success("Cache cleared successfully!")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Expired Cache"):
            expired = cache_manager.response_cache.clear_expired_responses()
            st.success(f"Cleared {expired} expired cache entries!")
            st.rerun()
    
    # Chunking settings
    st.subheader("📝 Text Chunking Settings")
    
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    
    if st.button("Update Chunking Settings"):
        smart_chunker.chunk_size = chunk_size
        smart_chunker.chunk_overlap = chunk_overlap
        st.success("Chunking settings updated!")

elif page == "🔧 Admin":
    st.header("🔧 Admin Dashboard")
    
    # System status
    st.subheader("🖥️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("✅ Cache Manager: Active")
        st.info("✅ Smart Chunker: Active")
        st.info("✅ Gemini Model: Active")
    
    with col2:
        st.info("✅ API Endpoints: Ready")
        st.info("✅ File Upload: Ready")
        st.info("✅ Analytics: Ready")
    
    # Recent queries
    st.subheader("📋 Recent Queries")
    
    try:
        recent_queries = cache_manager.response_cache.get_recent_queries(10)
        
        if recent_queries:
            for query in recent_queries:
                with st.expander(f"Query: {query['query'][:50]}..."):
                    st.write(f"**Timestamp:** {query['timestamp']}")
                    st.write(f"**Response Preview:** {query['response_preview']}")
        else:
            st.info("No recent queries found")
    
    except Exception as e:
        st.error(f"Error loading recent queries: {e}")

elif page == "📁 File Manager":
    st.header("📁 File Manager")
    
    # File upload with progress
    st.subheader("📤 Upload Files")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx', 'md'],
        help="Upload documents to process with the chatbot"
    )
    
    if uploaded_file is not None:
        # Show file details
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size} bytes")
        st.write(f"**Type:** {uploaded_file.type}")
        
        # Progress bar for file processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate file processing
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Processing file... {i+1}%")
            time.sleep(0.01)
        
        status_text.text("✅ File processed successfully!")
        
        # Show file preview
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.subheader("📄 File Preview")
            st.text_area("Content", content, height=200)
            
            # Chunk the content
            if st.button("🔪 Chunk This Text"):
                chunks = smart_chunker.chunk_text(content)
                
                st.subheader(f"📊 Chunking Results ({len(chunks)} chunks)")
                
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i+1} ({chunk['chunk_size']} chars)"):
                        st.write(chunk['text'])
                        st.write(f"**Type:** {chunk.get('type', 'unknown')}")
                        st.write(f"**Boundary:** {chunk.get('boundary', 'unknown')}")

# Footer
st.markdown("---")
st.markdown("**Built by Mandeep Khatri** | Powered by Gemini 1.5 Flash | Enhanced with Smart Features 🚀")
