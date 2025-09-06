"""Simple demo without PDF processing dependencies."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def setup_logging():
    """Setup basic logging."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )


def demo_basic_qa():
    """Demo basic Q&A functionality."""
    print("🤖 Q&A Chatbot Demo - Gemini 1.5 Pro")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("   Then run: export GEMINI_API_KEY='your-api-key'")
        return False
    
    try:
        # Initialize Gemini
        genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            max_output_tokens=1000
        )
        
        print("✅ Gemini 1.5 Pro initialized successfully")
        
        # Sample course content
        course_content = """
        Machine Learning Course Material:
        
        Chapter 1: Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that enables computers to learn 
        and make decisions from data without being explicitly programmed. It focuses on building 
        algorithms that can identify patterns in data and make predictions or decisions.
        
        Types of Machine Learning:
        
        1. Supervised Learning:
           - Uses labeled training data
           - Examples: Linear regression, decision trees, neural networks
           - Goal: Learn mapping from inputs to outputs
        
        2. Unsupervised Learning:
           - Works with unlabeled data
           - Examples: K-means clustering, PCA, association rules
           - Goal: Find hidden patterns in data
        
        3. Reinforcement Learning:
           - Learns through interaction with environment
           - Examples: Q-learning, policy gradient methods
           - Goal: Maximize cumulative reward
        
        Key Concepts:
        - Training Data: Dataset used to train the model
        - Features: Input variables used for prediction
        - Labels: Target variables we want to predict
        - Overfitting: Model performs well on training data but poorly on new data
        - Cross-validation: Technique to evaluate model performance
        - Bias-Variance Tradeoff: Balance between model complexity and generalization
        """
        
        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        chunks = text_splitter.split_text(course_content)
        print(f"📚 Split content into {len(chunks)} chunks")
        
        # Demo questions
        questions = [
            "What is machine learning?",
            "What are the three types of machine learning?",
            "What is overfitting?",
            "Explain supervised learning with examples",
            "What is the bias-variance tradeoff?"
        ]
        
        print("\n💬 Demo Q&A Session:")
        print("-" * 30)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            
            # Create context from relevant chunks
            context = "\n\n".join(chunks)
            
            # Create prompt with context
            prompt = f"""
            Based on the following machine learning course material, please answer the question.
            
            Course Material:
            {context}
            
            Question: {question}
            
            Please provide a clear, accurate, and educational answer based on the course material.
            If the answer is not in the material, please say so.
            """
            
            # Get response
            response = model.invoke(prompt)
            print(f"📚 Answer: {response.content}")
            print("-" * 50)
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo error: {e}")
        return False


def demo_embeddings():
    """Demo Gemini embeddings functionality."""
    print("\n🔍 Embeddings Demo")
    print("=" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        return False
    
    try:
        genai.configure(api_key=api_key)
        
        # Sample texts
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Supervised learning requires labeled training data",
            "Unsupervised learning finds patterns in unlabeled data"
        ]
        
        print("📝 Generating embeddings for sample texts...")
        
        for i, text in enumerate(texts, 1):
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            print(f"   {i}. Text: {text[:50]}...")
            print(f"      Embedding dimension: {len(embedding)}")
            print(f"      First 3 values: {[round(x, 4) for x in embedding[:3]]}")
        
        print("✅ Embeddings generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def interactive_demo():
    """Interactive demo where user can ask questions."""
    print("\n🎯 Interactive Demo")
    print("=" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1
        )
        
        # Sample course content
        course_content = """
        Computer Science Fundamentals:
        
        Data Structures:
        - Arrays: Fixed-size collections of elements with index-based access
        - Linked Lists: Dynamic collections with pointers connecting nodes
        - Stacks: LIFO (Last In, First Out) data structure
        - Queues: FIFO (First In, First Out) data structure
        - Trees: Hierarchical data structures with parent-child relationships
        - Graphs: Networks of nodes connected by edges
        
        Algorithms:
        - Sorting: Bubble sort O(n²), Quick sort O(n log n), Merge sort O(n log n)
        - Searching: Linear search O(n), Binary search O(log n)
        - Graph algorithms: BFS, DFS, Dijkstra's shortest path
        
        Complexity Analysis:
        - Big O notation: Describes how algorithm performance scales
        - Time complexity: How runtime grows with input size
        - Space complexity: How memory usage grows with input size
        - Common complexities: O(1), O(log n), O(n), O(n log n), O(n²)
        """
        
        print("📚 Course content loaded: Computer Science Fundamentals")
        print("💡 You can ask questions about data structures, algorithms, or complexity analysis")
        print("   Type 'quit' to exit")
        
        while True:
            question = input("\n🤖 Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                prompt = f"""
                Based on the following computer science course material, answer the question.
                
                Course Material:
                {course_content}
                
                Question: {question}
                
                Provide a clear and educational answer based on the course material.
                If the answer is not in the material, please say so.
                """
                
                response = model.invoke(prompt)
                print(f"📚 Answer: {response.content}")
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_system_status():
    """Show current system status."""
    print("\n📋 System Status")
    print("=" * 20)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"Gemini API Key: {'✅ Set' if api_key else '❌ Not set'}")
    
    # Check Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check working directory
    print(f"Working Directory: {os.getcwd()}")
    
    # Check if we're in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"Virtual Environment: {'✅ Active' if in_venv else '❌ Not active'}")
    
    print("\n🚀 What this demo shows:")
    print("   • Gemini 1.5 Pro integration")
    print("   • Text processing and chunking")
    print("   • Context-aware Q&A")
    print("   • Embeddings generation")
    print("   • Interactive chat functionality")
    
    print("\n📚 Full system features:")
    print("   • PDF processing with PyMuPDF")
    print("   • Vector storage with ChromaDB")
    print("   • Web interface with Streamlit")
    print("   • Advanced text splitting")
    print("   • Source citation and retrieval")


def main():
    """Main demo function."""
    setup_logging()
    
    print("🚀 Q&A Chatbot Demo - Built with Gemini 1.5 Pro & LangChain")
    print("=" * 60)
    
    # Show system status
    show_system_status()
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  GEMINI_API_KEY not found in environment variables")
        print("   Please set it with: export GEMINI_API_KEY='your-api-key'")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Run demos
    print("\n" + "="*60)
    success1 = demo_basic_qa()
    
    if success1:
        print("\n" + "="*60)
        success2 = demo_embeddings()
        
        if success2:
            # Ask if user wants interactive demo
            response = input("\n🎮 Would you like to try the interactive demo? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n" + "="*60)
                interactive_demo()
    
    print("\n🎉 Demo completed!")
    print("💡 To use the full system with PDF processing:")
    print("   1. Install PyMuPDF: pip install PyMuPDF")
    print("   2. Add your PDF files to ./data/pdfs/")
    print("   3. Run: python main.py --process-pdfs ./data/pdfs")
    print("   4. Start chat: python main.py --chat")


if __name__ == "__main__":
    main()
