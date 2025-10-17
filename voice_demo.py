#!/usr/bin/env python3
"""
Voice Chatbot Demo - Simple demonstration of voice features
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

def test_voice_packages():
    """Test if voice packages are installed correctly."""
    print("🎤 Testing Voice Packages...")
    
    try:
        import pyaudio
        print("✅ PyAudio: Installed")
    except ImportError:
        print("❌ PyAudio: Not installed")
        return False
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition: Installed")
    except ImportError:
        print("❌ SpeechRecognition: Not installed")
        return False
    
    try:
        import pyttsx3
        print("✅ pyttsx3: Installed")
    except ImportError:
        print("❌ pyttsx3: Not installed")
        return False
    
    return True

def test_microphone():
    """Test microphone access."""
    print("\n🎤 Testing Microphone Access...")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        # List available devices
        print("Available audio devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  - Device {i}: {info['name']}")
        
        audio.terminate()
        print("✅ Microphone: Accessible")
        return True
        
    except Exception as e:
        print(f"❌ Microphone: Error - {e}")
        return False

def test_text_to_speech():
    """Test text-to-speech functionality."""
    print("\n🔊 Testing Text-to-Speech...")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        
        for i, voice in enumerate(voices):
            print(f"  - Voice {i}: {voice.name}")
        
        # Test speech
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        
        print("Speaking test message...")
        engine.say("Hello! This is a test of the text-to-speech system.")
        engine.runAndWait()
        
        print("✅ Text-to-Speech: Working")
        return True
        
    except Exception as e:
        print(f"❌ Text-to-Speech: Error - {e}")
        return False

def test_speech_recognition():
    """Test speech recognition functionality."""
    print("\n🎯 Testing Speech Recognition...")
    
    try:
        import speech_recognition as sr
        
        r = sr.Recognizer()
        
        # Test with microphone
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("✅ Speech Recognition: Ready")
            print("Note: Actual recognition requires microphone input")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech Recognition: Error - {e}")
        return False

def test_gemini_api():
    """Test Gemini API connection."""
    print("\n🤖 Testing Gemini API...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("❌ Gemini API: No API key configured")
        print("Please add your API key to config.env file")
        return False
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Test with a simple query
        response = model.generate_content("Say hello in one word")
        print(f"✅ Gemini API: Working - Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API: Error - {e}")
        return False

def main():
    """Run all voice feature tests."""
    print("🎤 Voice Chatbot Demo - Testing Voice Features")
    print("=" * 50)
    
    # Test all components
    tests = [
        ("Voice Packages", test_voice_packages),
        ("Microphone", test_microphone),
        ("Text-to-Speech", test_text_to_speech),
        ("Speech Recognition", test_speech_recognition),
        ("Gemini API", test_gemini_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Voice chatbot is ready to use.")
        print("Run: streamlit run voice_chatbot.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure to install all dependencies and configure your API key.")

if __name__ == "__main__":
    main()


