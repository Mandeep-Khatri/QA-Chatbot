"""Demo script to show the Q&A chatbot functionality."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def setup_logging():
    """Setup basic logging."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )


def demo_gemini_qa():
    """Demo basic Gemini Q&A functionality."""
    print("🤖 Q&A Chatbot Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize Gemini
        genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            max_output_tokens=1000
        )
        
        print("✅ Gemini 1.5 Pro initialized successfully")
        
        # Sample course content (simulating processed PDF content)
        course_content = """
        Machine Learning Fundamentals:
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn and make decisions from data. There are three main types:
        
        1. Supervised Learning: Uses labeled training data to learn a mapping from inputs to outputs.
           Examples include linear regression, decision trees, and neural networks.
        
        2. Unsupervised Learning: Finds hidden patterns in data without labeled examples.
           Examples include clustering (k-means) and dimensionality reduction (PCA).
        
        3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties.
           Examples include Q-learning and policy gradient methods.
        
        Key concepts:
        - Training data: The dataset used to train the model
        - Features: Input variables used to make predictions
        - Labels: The target variable we want to predict
        - Overfitting: When a model performs well on training data but poorly on new data
        - Cross-validation: Technique to evaluate model performance on unseen data
        """
        
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunks = text_splitter.split_text(course_content)
        print(f"📚 Split content into {len(chunks)} chunks")
        
        # Demo questions
        questions = [
            "What is machine learning?",
            "What are the three types of machine learning?",
            "What is overfitting?",
            "Explain supervised learning with examples"
        ]
        
        print("\n💬 Demo Q&A Session:")
        print("-" * 30)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            
            # Create context from relevant chunks
            context = "\n\n".join(chunks)
            
            # Create prompt with context
            prompt = f"""
            Based on the following course material, please answer the question.
            
            Course Material:
            {context}
            
            Question: {question}
            
            Please provide a clear and accurate answer based on the course material.
            """
            
            # Get response
            response = model.invoke(prompt)
            print(f"📚 Answer: {response.content}")
            print("-" * 50)
        
        print("\n✅ Demo completed successfully!")
        print("\n📋 What this demo showed:")
        print("   • Gemini 1.5 Pro integration")
        print("   • Text chunking and processing")
        print("   • Context-aware Q&A")
        print("   • Course material understanding")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo error: {e}")


def demo_embeddings():
    """Demo Gemini embeddings functionality."""
    print("\n🔍 Embeddings Demo")
    print("=" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        return
    
    try:
        genai.configure(api_key=api_key)
        
        # Sample texts
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Supervised learning requires labeled training data"
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
            print(f"      First 5 values: {embedding[:5]}")
        
        print("✅ Embeddings generated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_demo():
    """Interactive demo where user can ask questions."""
    print("\n🎯 Interactive Demo")
    print("=" * 30)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set your GEMINI_API_KEY environment variable")
        return
    
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
        - Arrays: Fixed-size collections of elements
        - Linked Lists: Dynamic collections with pointers
        - Stacks: LIFO (Last In, First Out) data structure
        - Queues: FIFO (First In, First Out) data structure
        - Trees: Hierarchical data structures
        - Graphs: Networks of nodes and edges
        
        Algorithms:
        - Sorting: Bubble sort, Quick sort, Merge sort
        - Searching: Linear search, Binary search
        - Graph algorithms: BFS, DFS, Dijkstra's algorithm
        
        Complexity Analysis:
        - Big O notation: Describes algorithm efficiency
        - Time complexity: How runtime grows with input size
        - Space complexity: How memory usage grows with input size
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
                
                Provide a clear and educational answer.
                """
                
                response = model.invoke(prompt)
                print(f"📚 Answer: {response.content}")
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main demo function."""
    setup_logging()
    
    print("🚀 Mandy's Q&A Chatbot Demo - Built with Gemini 1.5 Pro & LangChain")
    print("=" * 60)
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not found in environment variables")
        print("   Please set it with: export GEMINI_API_KEY='your-api-key'")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("\n   For now, showing demo structure...")
        
        print("\n📋 Demo Components:")
        print("   1. ✅ Project structure created")
        print("   2. ✅ Gemini 1.5 Pro integration")
        print("   3. ✅ LangChain framework")
        print("   4. ✅ Text processing and chunking")
        print("   5. ✅ Q&A functionality")
        print("   6. ✅ Embeddings generation")
        print("   7. ✅ Vector storage (ChromaDB)")
        print("   8. ✅ Web interface (Streamlit)")
        
        print("\n🔧 To run the full demo:")
        print("   1. Get Gemini API key from AI Studio")
        print("   2. Set environment variable: export GEMINI_API_KEY='your-key'")
        print("   3. Run: python demo.py")
        
        return
    
    # Run demos
    demo_gemini_qa()
    demo_embeddings()
    
    # Ask if user wants interactive demo
    response = input("\n🎮 Would you like to try the interactive demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_demo()


if __name__ == "__main__":
    main()
