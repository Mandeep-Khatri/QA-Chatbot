"""
Mandy's Q&A Chatbot - Simple Streamlit web interface (without PyMuPDF).
"""

import streamlit as st
import os
from pathlib import Path

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Google AI Studio by Mandy",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Google AI Studio by Mandy")
    st.markdown("**Mandy's Q&A Chatbot for Course Materials** - Ask questions about your course materials using AI-powered search and retrieval.")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Google AI Studio by Mandy")
        st.markdown("---")
        st.header("📊 System Information")
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API Key: Configured")
        else:
            st.error("❌ Gemini API Key: Not configured")
            st.info("Set GEMINI_API_KEY environment variable")
        
        st.markdown("---")
        st.header("🚀 Quick Start")
        st.markdown("""
        1. **Set up API Key**: Get your Gemini API key from AI Studio
        2. **Add PDFs**: Place your course PDFs in the `data/pdfs/` folder
        3. **Process Documents**: Use the main app to process and chat
        4. **Ask Questions**: Get AI-powered answers about your materials
        """)
    
    # Main content
    st.header("🎯 Welcome to Mandy's Q&A Chatbot!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Features")
        st.markdown("""
        - **PDF Processing**: Extract text from course materials
        - **AI-Powered Search**: Find relevant information quickly
        - **Smart Q&A**: Get accurate answers using Gemini 1.5 Pro
        - **Vector Storage**: Efficient document retrieval
        - **Web Interface**: Easy-to-use chat interface
        """)
    
    with col2:
        st.subheader("🛠️ Setup Status")
        
        # Check if data directory exists
        data_dir = Path("data")
        if data_dir.exists():
            st.success("✅ Data directory exists")
        else:
            st.warning("⚠️ Data directory not found")
        
        # Check if PDFs directory exists
        pdfs_dir = Path("data/pdfs")
        if pdfs_dir.exists():
            pdf_files = list(pdfs_dir.glob("*.pdf"))
            if pdf_files:
                st.success(f"✅ Found {len(pdf_files)} PDF files")
                for pdf in pdf_files[:3]:  # Show first 3
                    st.text(f"  📄 {pdf.name}")
                if len(pdf_files) > 3:
                    st.text(f"  ... and {len(pdf_files) - 3} more")
            else:
                st.info("ℹ️ No PDF files found in data/pdfs/")
        else:
            st.warning("⚠️ PDFs directory not found")
    
    st.markdown("---")
    
    # Demo section
    st.header("💬 Demo Chat")
    st.info("This is a simplified version. For full functionality, install PyMuPDF and set up your API key.")
    
    # Simple chat interface
    user_question = st.text_input("Ask a question about your course materials:", placeholder="e.g., What is machine learning?")
    
    if st.button("Ask Question", type="primary"):
        if user_question:
            st.success(f"Question received: '{user_question}'")
            st.info("In the full version, this would process your question using AI and return an answer based on your course materials.")
        else:
            st.warning("Please enter a question first.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built by Mandy** | Powered by Gemini 1.5 Pro & LangChain")

if __name__ == "__main__":
    main()
