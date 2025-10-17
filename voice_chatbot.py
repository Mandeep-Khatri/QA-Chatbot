"""Voice-enabled chatbot with microphone input and text-to-speech output."""

import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
import json
import time
import base64
import io
import wave
import pyaudio
import speech_recognition as sr
import pyttsx3
from typing import Dict, List, Optional
import threading
import tempfile

# Configure page
st.set_page_config(
    page_title="🤖 Gemini by Mandy - Voice Chat",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv("config.env")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

# Audio configuration
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

def get_gemini_model():
    """Get Gemini model with API key validation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        st.error("❌ Gemini API Key: Not configured")
        st.info("Please add your API key to config.env file")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"❌ Error initializing Gemini: {str(e)}")
        return None

def generate_response(model, prompt: str) -> str:
    """Generate response using Gemini model."""
    if not model:
        return "❌ Model not available. Please check your API key."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

def record_audio():
    """Record audio from microphone."""
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Convert to bytes
        audio_data = b''.join(frames)
        return audio_data
    except Exception as e:
        st.error(f"❌ Error recording audio: {str(e)}")
        return None

def audio_to_text(audio_data):
    """Convert audio to text using speech recognition."""
    try:
        r = sr.Recognizer()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_filename = temp_file.name
        
        # Load audio file
        with sr.AudioFile(temp_filename) as source:
            audio = r.record(source)
        
        # Recognize speech
        text = r.recognize_google(audio)
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        return text
    except Exception as e:
        st.error(f"❌ Error converting audio to text: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()
        
        # Read the generated audio file
        with open(temp_filename, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        return audio_bytes
    except Exception as e:
        st.error(f"❌ Error converting text to speech: {str(e)}")
        return None

def create_audio_player(audio_bytes):
    """Create HTML audio player for the audio bytes."""
    try:
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Create audio HTML
        audio_html = f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"❌ Error creating audio player: {str(e)}")
        return None

# Main app
def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🤖 Gemini by Mandy</h1>
        <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Your AI Assistant with Voice Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎤 Voice Settings")
        
        # Voice settings
        st.session_state.voice_enabled = st.checkbox("Enable Voice Input/Output", value=st.session_state.voice_enabled)
        auto_play = st.checkbox("Auto-play Voice Responses", value=True)
        
        # Model status
        model = get_gemini_model()
        if model:
            st.success("✅ Gemini Model: Ready")
        else:
            st.error("❌ Gemini Model: Not Available")
        
        # Chat history
        st.subheader("📚 Chat History")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Chat {len(st.session_state.chat_history)-i}"):
                    st.write(f"**Query:** {chat['query']}")
                    st.write(f"**Response:** {chat['response'][:100]}...")
                    st.write(f"**Time:** {chat['timestamp']}")
        else:
            st.info("No chat history yet")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with Gemini")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Add voice playback for assistant messages if voice is enabled
                if message["role"] == "assistant" and st.session_state.voice_enabled and auto_play:
                    if "audio" in message:
                        st.markdown(create_audio_player(message["audio"]), unsafe_allow_html=True)
        
        # Combined text and voice input
        st.subheader("💬 Ask Your Question")
        
        # Create input row with text box and mic button
        col_text, col_mic = st.columns([4, 1])
        
        with col_text:
            # Text input
            user_input = st.text_input(
                "Ask me anything about your course materials or any topic!",
                value=st.session_state.user_input,
                key="text_input",
                placeholder="Type your question or click mic to record..."
            )
        
        with col_mic:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            
            # Microphone button
            if st.session_state.voice_enabled:
                if st.button("🎤", key="mic_button", help="Click to record voice"):
                    if not st.session_state.is_recording:
                        # Start recording
                        st.session_state.is_recording = True
                        st.info("🎤 Recording... Speak now!")
                        
                        # Record audio
                        audio_data = record_audio()
                        if audio_data:
                            st.session_state.audio_data = audio_data
                            st.success("✅ Audio recorded!")
                            
                            # Convert to text
                            text = audio_to_text(audio_data)
                            if text:
                                st.session_state.user_input = text
                                st.write(f"🎯 Recognized: {text}")
                            else:
                                st.error("❌ Could not recognize speech")
                        else:
                            st.error("❌ Failed to record audio")
                        
                        st.session_state.is_recording = False
                    else:
                        # Stop recording
                        st.session_state.is_recording = False
                        st.info("⏹️ Recording stopped")
            else:
                st.button("🎤", key="mic_button", disabled=True, help="Voice features disabled")
        
        # Process the input (from text or voice)
        if user_input:
            st.session_state.user_input = user_input
        
        # Also handle Enter key press (when user types and presses Enter)
        if st.session_state.user_input and st.session_state.user_input != user_input:
            # This handles the case when user types and presses Enter
            pass
        
        # Query suggestions
        st.write("💡 Quick Suggestions:")
        suggestions = [
            "What is machine learning?",
            "Explain neural networks",
            "How does AI work?",
            "What are the benefits of automation?",
            "Tell me about data science"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{suggestion}"):
                    st.session_state.user_input = suggestion
        
        # Process input (from text input, voice, or suggestions)
        if st.session_state.user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
            
            # Generate response
            with st.spinner("Thinking..."):
                response = generate_response(model, st.session_state.user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Generate voice response if enabled
            if st.session_state.voice_enabled:
                with st.spinner("Generating voice response..."):
                    audio_bytes = text_to_speech(response)
                    if audio_bytes:
                        # Add audio to the last message
                        st.session_state.messages[-1]["audio"] = audio_bytes
            
            # Add to chat history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": st.session_state.user_input,
                "response": response,
                "cached": False,
                "processing_time": 0.5
            })
            
            # Clear input
            st.session_state.user_input = ""
            st.rerun()
    
    with col2:
        st.subheader("📊 Analytics")
        
        # Basic stats
        total_chats = len(st.session_state.chat_history)
        st.metric("Total Chats", total_chats)
        
        if total_chats > 0:
            avg_response_time = sum(chat.get("processing_time", 0) for chat in st.session_state.chat_history) / total_chats
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        # Voice stats
        if st.session_state.voice_enabled:
            st.metric("Voice Mode", "🎤 Active")
        else:
            st.metric("Voice Mode", "🔇 Disabled")
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history[-3:]):
                st.write(f"**{chat['timestamp']}**")
                st.write(f"Q: {chat['query'][:30]}...")
                st.write(f"A: {chat['response'][:30]}...")
                st.write("---")
        else:
            st.info("No recent activity")
        
        # Export functionality
        st.subheader("📤 Export")
        if st.button("📥 Download Chat History"):
            if st.session_state.chat_history:
                chat_data = {
                    "export_date": datetime.now().isoformat(),
                    "total_chats": len(st.session_state.chat_history),
                    "chat_history": st.session_state.chat_history
                }
                
                json_str = json.dumps(chat_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"voice_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No chat history to export")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Built by Mandeep Khatri</strong> | Powered by Gemini 1.5 Flash | Enhanced with Voice Features 🎤</p>
        <p>🎤 Voice Input/Output | 📊 Analytics | 📤 Export | 🎯 Smart Suggestions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
