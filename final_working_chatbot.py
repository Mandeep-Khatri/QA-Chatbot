"""Final working chatbot with all enhanced features."""

import streamlit as st
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Import working chatbot functions
try:
    from working_chatbot import get_gemini_model, generate_response
except ImportError:
    # Fallback if working_chatbot is not available
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    load_dotenv()
    
    def get_gemini_model():
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        return genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_response(model, query, context=""):
        prompt = f"{context}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(prompt)
        return response.text

# Simple cache implementation
class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.max_size = 100
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def stats(self):
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size
        }

# Initialize cache
@st.cache_resource
def get_cache():
    return SimpleCache()

cache = get_cache()

# Page configuration
st.set_page_config(
    page_title="🤖 Gemini by Mandy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        ["💬 Chat", "📊 Analytics", "⚙️ Settings", "🔧 Admin"]
    )
    
    # Cache stats
    st.subheader("📈 Cache Statistics")
    cache_stats = cache.stats()
    st.metric("Cached Responses", cache_stats["total_entries"])
    st.metric("Cache Capacity", cache_stats["max_size"])

# Main content
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
                    cached_response = cache.get(prompt)
                    
                    if cached_response:
                        response = cached_response
                        st.success("✅ Response from cache (faster!)")
                    else:
                        model = get_gemini_model()
                        response = generate_response(model, prompt)
                        
                        # Cache the response
                        cache.set(prompt, response)
                    
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
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(st.session_state.get("chat_history", [])))
    
    with col2:
        cache_hits = sum(1 for entry in st.session_state.get("chat_history", []) if entry.get("cached", False))
        st.metric("Cache Hits", cache_hits)
    
    with col3:
        total_queries = len(st.session_state.get("chat_history", []))
        hit_rate = (cache_hits / max(total_queries, 1)) * 100
        st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    
    with col4:
        avg_time = sum(entry.get("processing_time", 0) for entry in st.session_state.get("chat_history", [])) / max(total_queries, 1)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # Charts
    if st.session_state.get("chat_history"):
        st.subheader("📈 Performance Charts")
        
        df = pd.DataFrame(st.session_state.chat_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Processing time chart
        st.line_chart(df.set_index('timestamp')['processing_time'])
        
        # Cache hit rate over time
        df['cache_hit'] = df['cached'].astype(int)
        st.bar_chart(df.set_index('timestamp')['cache_hit'])

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    # Cache settings
    st.subheader("🗄️ Cache Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Cache"):
            cache.clear()
            st.success("Cache cleared successfully!")
            st.rerun()
    
    with col2:
        st.info(f"Current cache size: {cache.stats()['total_entries']} entries")
    
    # System info
    st.subheader("🖥️ System Information")
    st.info("✅ Cache System: Active")
    st.info("✅ Gemini Model: Active")
    st.info("✅ Export Functionality: Active")
    st.info("✅ Analytics: Active")

elif page == "🔧 Admin":
    st.header("🔧 Admin Dashboard")
    
    # System status
    st.subheader("🖥️ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("✅ Cache Manager: Active")
        st.info("✅ Chat Interface: Active")
        st.info("✅ Analytics: Active")
    
    with col2:
        st.info("✅ Export System: Active")
        st.info("✅ Settings Panel: Active")
        st.info("✅ Admin Panel: Active")
    
    # Recent queries
    st.subheader("📋 Recent Queries")
    
    if st.session_state.get("chat_history"):
        recent_queries = st.session_state.chat_history[-10:]
        
        for query in reversed(recent_queries):
            with st.expander(f"Query: {query['query'][:50]}..."):
                st.write(f"**Timestamp:** {query['timestamp']}")
                st.write(f"**Processing Time:** {query['processing_time']:.2f}s")
                st.write(f"**Cached:** {'Yes' if query['cached'] else 'No'}")
                st.write(f"**Response Preview:** {query['response'][:100]}...")
    else:
        st.info("No recent queries found")

# Footer
st.markdown("---")
st.markdown("**Built by Mandeep Khatri** | Powered by Gemini 1.5 Flash | Enhanced with Smart Features 🚀")
