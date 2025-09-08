"""Test script for enhanced features."""

import requests
import json
import time

def test_api():
    """Test the REST API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Enhanced Features...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            print(f"   Status: {response.json()['status']}")
        else:
            print("❌ API Health Check: FAILED")
    except Exception as e:
        print(f"❌ API Health Check: ERROR - {e}")
    
    # Test query endpoint
    try:
        query_data = {
            "query": "What is artificial intelligence?",
            "use_cache": True
        }
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Query Endpoint: PASSED")
            print(f"   Response: {result['response'][:100]}...")
            print(f"   Cached: {result['cached']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
        else:
            print("❌ Query Endpoint: FAILED")
    except Exception as e:
        print(f"❌ Query Endpoint: ERROR - {e}")
    
    # Test chunking endpoint
    try:
        chunk_data = {
            "text": "This is a test document. It has multiple sentences. Each sentence should be processed separately. The chunking should work properly.",
            "chunk_size": 100,
            "chunk_overlap": 20
        }
        response = requests.post(f"{base_url}/chunk", json=chunk_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Chunking Endpoint: PASSED")
            print(f"   Chunks Created: {len(result['chunks'])}")
            print(f"   Statistics: {result['statistics']}")
        else:
            print("❌ Chunking Endpoint: FAILED")
    except Exception as e:
        print(f"❌ Chunking Endpoint: ERROR - {e}")
    
    # Test cache stats
    try:
        response = requests.get(f"{base_url}/cache/stats")
        if response.status_code == 200:
            result = response.json()
            print("✅ Cache Stats: PASSED")
            print(f"   Embedding Cache: {result['embedding_cache']}")
            print(f"   Response Cache: {result['response_cache']}")
        else:
            print("❌ Cache Stats: FAILED")
    except Exception as e:
        print(f"❌ Cache Stats: ERROR - {e}")

def test_streamlit():
    """Test Streamlit app."""
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("✅ Streamlit App: RUNNING")
            print("   URL: http://localhost:8501")
        else:
            print("❌ Streamlit App: NOT RUNNING")
    except Exception as e:
        print(f"❌ Streamlit App: ERROR - {e}")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Q&A Chatbot Features")
    print("=" * 50)
    
    test_streamlit()
    print()
    test_api()
    
    print("\n" + "=" * 50)
    print("🎉 Testing Complete!")
    print("\n📱 Access Points:")
    print("   • Enhanced Streamlit App: http://localhost:8501")
    print("   • REST API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
