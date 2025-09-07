"""
Mandy's Working Q&A Chatbot - Simple version that actually answers questions.
"""

import streamlit as st
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

def initialize_gemini():
    """Initialize Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None, "API key not configured"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model, "API configured successfully"
    except Exception as e:
        return None, f"Error configuring API: {str(e)}"

def get_ai_response(model, question):
    """Get response from Gemini."""
    try:
        response = model.generate_content(question)
        return response.text, None
    except Exception as e:
        return None, f"Error getting response: {str(e)}"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Google AI Studio by Mandy",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Google AI Studio by Mandy")
    st.markdown("**Mandy's Q&A Chatbot for Course Materials** - Ask questions and get AI-powered answers!")
    
    # Initialize Gemini
    model, status = initialize_gemini()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Google AI Studio by Mandy")
        st.markdown("---")
        st.header("📊 System Information")
        
        if model:
            st.success("✅ Gemini API: Connected")
            st.success("✅ Ready to answer questions!")
        else:
            st.error("❌ Gemini API: Not configured")
            st.info("Set your API key in config.env file")
            st.markdown("**Get your API key from:**")
            st.markdown("[AI Studio](https://makersuite.google.com/app/apikey)")
        
        st.markdown("---")
        st.header("🚀 How to Use")
        st.markdown("""
        1. **Set API Key**: Edit `config.env` file
        2. **Ask Questions**: Type your question below
        3. **Get Answers**: AI will respond instantly
        4. **Learn More**: Ask about any topic!
        """)
    
    # Main content
    st.header("💬 Ask Me Anything!")
    
    if not model:
        st.warning("⚠️ Please configure your Gemini API key to start chatting.")
        st.info("""
        **To get started:**
        1. Get your API key from [AI Studio](https://makersuite.google.com/app/apikey)
        2. Edit the `config.env` file in your project
        3. Replace `your_api_key_here` with your actual API key
        4. Restart this app
        """)
        return
    
    # Chat interface
    st.success("🎉 Ready to chat! Ask me anything!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your course materials or any topic!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, error = get_ai_response(model, prompt)
                
                if response:
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Error: {error}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built by Mandy** | Powered by Gemini 1.5 Flash")

if __name__ == "__main__":
    main()
