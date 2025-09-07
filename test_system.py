"""Test script to verify the chatbot system components."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing System Components")
    print("=" * 40)
    
    tests = [
        ("Configuration", "src.config", "Config"),
        ("PDF Processor", "src.pdf_processor", "PDFExtractor"),
        ("Text Splitter", "src.utils", "TextSplitter"),
        ("Embeddings", "src.embeddings", "GeminiEmbeddings"),
        ("Vector Store", "src.embeddings", "VectorStore"),
        ("Chatbot", "src.chatbot", "QAChatbot"),
    ]
    
    results = []
    
    for name, module_path, class_name in tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {name}: OK")
            results.append(True)
        except ImportError as e:
            print(f"❌ {name}: Import Error - {e}")
            results.append(False)
        except AttributeError as e:
            print(f"❌ {name}: Class Error - {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} components working")
    return all(results)


def test_dependencies():
    """Test if required dependencies are installed."""
    print("\n📦 Testing Dependencies")
    print("=" * 30)
    
    dependencies = [
        ("google-generativeai", "Gemini API"),
        ("langchain", "LangChain Framework"),
        ("langchain_google_genai", "LangChain Gemini Integration"),
        ("python-dotenv", "Environment Variables"),
        ("loguru", "Logging"),
    ]
    
    results = []
    
    for package, description in dependencies:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {description}: OK")
            results.append(True)
        except ImportError:
            print(f"❌ {description}: Not installed")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} dependencies available")
    return all(results)


def test_file_structure():
    """Test if all required files exist."""
    print("\n📁 Testing File Structure")
    print("=" * 30)
    
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/pdf_processor/__init__.py",
        "src/pdf_processor/pdf_extractor.py",
        "src/utils/__init__.py",
        "src/utils/text_splitter.py",
        "src/embeddings/__init__.py",
        "src/embeddings/gemini_embeddings.py",
        "src/embeddings/vector_store.py",
        "src/chatbot/__init__.py",
        "src/chatbot/qa_chatbot.py",
        "main.py",
        "streamlit_app.py",
        "demo.py",
        "requirements.txt",
        "README.md",
    ]
    
    results = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            results.append(True)
        else:
            print(f"❌ {file_path}: Missing")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} files present")
    return all(results)


def test_directories():
    """Test if required directories exist."""
    print("\n📂 Testing Directory Structure")
    print("=" * 35)
    
    required_dirs = [
        "src",
        "src/pdf_processor",
        "src/utils",
        "src/embeddings",
        "src/chatbot",
        "data",
        "data/pdfs",
        "data/vector_db",
        "logs",
        "notebooks",
        "tests",
    ]
    
    results = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
            results.append(True)
        else:
            print(f"❌ {dir_path}/: Missing")
            results.append(False)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} directories present")
    return all(results)


def show_system_info():
    """Show system information and next steps."""
    print("\n📋 System Information")
    print("=" * 25)
    
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Environment: {'Virtual Environment' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'System Python'}")
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"Gemini API Key: {'✅ Set' if api_key else '❌ Not set'}")
    else:
        print("Gemini API Key: ❌ Not set")
    
    print("\n🚀 Next Steps:")
    print("1. Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("2. Set environment variable: export GEMINI_API_KEY='your-key'")
    print("3. Run demo: python demo.py")
    print("4. Process PDFs: python main.py --process-pdfs ./data/pdfs")
    print("5. Start chat: python main.py --chat")
    print("6. Web interface: streamlit run streamlit_app.py")


def main():
    """Main test function."""
    print("🔍 Mandy's Q&A Chatbot System Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_file_structure,
        test_directories,
        test_dependencies,
        test_imports,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    # Show system info
    show_system_info()
    
    # Final summary
    print("\n🎯 Final Summary")
    print("=" * 20)
    
    if all(results):
        print("✅ All tests passed! System is ready to use.")
        print("🚀 You can now run the demo or start using the chatbot.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("💡 Most issues can be resolved by installing missing dependencies.")
    
    print(f"\n📊 Overall Score: {sum(results)}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
