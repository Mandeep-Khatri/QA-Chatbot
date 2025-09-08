"""Simple API test without complex imports."""

import requests
import json

def test_simple_api():
    """Test a simple API endpoint."""
    try:
        # Test if the API is running
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Simple API...")
    test_simple_api()
