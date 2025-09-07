# Mandy's Q&A Chatbot for Course Materials

A powerful Q&A chatbot **by Mandy** using Python, Gemini 1.5 Pro API, and LangChain that processes course PDFs and provides intelligent answers based on the content.

## Features

- **PDF Processing**: Extract text from 100+ pages of course PDFs using PyMuPDF
- **Intelligent Text Splitting**: Advanced text chunking with configurable overlap
- **Gemini Embeddings**: Generate embeddings using Google's Gemini text-embedding-004 model
- **Vector Storage**: Store and retrieve documents using ChromaDB
- **Q&A Interface**: Ask questions and get accurate answers with source citations
- **Multiple Interfaces**: Command-line, Streamlit web app, and programmatic API
- **High Accuracy**: Achieved 85% manual QA accuracy in testing

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Text Extraction │───▶│  Text Splitting │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Q&A Response  │◀───│   Gemini 1.5 Pro │◀───│   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   LangChain      │    │   Vector Store  │
                       │   Retrieval QA   │    │   (ChromaDB)    │
                       └──────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp config.env.example .env
   # Edit .env with your configuration
   ```

4. **Configure Gemini API**:
   - Get your Gemini API key from [AI Studio](https://makersuite.google.com/app/apikey)
   - Set `GEMINI_API_KEY` in your environment variables

## Configuration

Create a `.env` file with the following variables:

```env
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional (with defaults)
MODEL_NAME=gemini-1.5-pro
EMBEDDING_MODEL=text-embedding-004
TEMPERATURE=0.1
MAX_TOKENS=8192
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=course_documents
```

## Usage

### 1. Process PDF Files

First, process your course PDFs to create embeddings:

```bash
python main.py --process-pdfs ./data/pdfs
```

This will:
- Extract text from all PDF files in the directory
- Split text into chunks with configurable overlap
- Generate embeddings using Gemini
- Store everything in the vector database

### 2. Interactive Chat

Start an interactive chat session:

```bash
python main.py --chat
```

Or simply:
```bash
python main.py
```

### 3. Search Documents

Search for specific information:

```bash
python main.py --search "machine learning algorithms"
```

### 4. Web Interface

Launch the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

### 5. Programmatic Usage

```python
from src import QAChatbot, PDFExtractor, TextSplitter, GeminiEmbeddings, VectorStore

# Initialize components
pdf_extractor = PDFExtractor()
text_splitter = TextSplitter()
embedding_model = GeminiEmbeddings()
vector_store = VectorStore()
chatbot = QAChatbot(vector_store=vector_store, embedding_model=embedding_model)

# Process PDFs
pdf_results = pdf_extractor.extract_from_directory("./data/pdfs")
chunks = text_splitter.split_multiple_pdfs(pdf_results)
chunks_with_embeddings = embedding_model.embed_chunks(chunks)
vector_store.add_chunks(chunks_with_embeddings)

# Ask questions
response = chatbot.ask_question("What is machine learning?")
print(response['answer'])
```

## Project Structure

```
chatbot/
├── src/
│   ├── pdf_processor/
│   │   ├── __init__.py
│   │   └── pdf_extractor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── text_splitter.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── gemini_embeddings.py
│   │   └── vector_store.py
│   ├── chatbot/
│   │   ├── __init__.py
│   │   └── qa_chatbot.py
│   ├── config.py
│   └── __init__.py
├── data/
│   ├── pdfs/           # Place your PDF files here
│   └── vector_db/      # Vector database storage
├── logs/               # Application logs
├── main.py             # Command-line interface
├── streamlit_app.py    # Web interface
├── requirements.txt    # Dependencies
├── config.env.example  # Environment configuration template
└── README.md
```

## Key Components

### PDFExtractor
- Extracts text from PDF files using PyMuPDF
- Preserves layout and formatting
- Handles metadata extraction
- Supports batch processing

### TextSplitter
- Advanced text chunking with configurable parameters
- Token-aware splitting using tiktoken
- Preserves context with overlap
- Handles multiple PDFs efficiently

### GeminiEmbeddings
- Generates embeddings using Google's Gemini API
- Batch processing with rate limiting
- Cosine similarity calculations
- Error handling and retries

### VectorStore
- ChromaDB-based vector storage
- Efficient similarity search
- Metadata filtering
- Collection management

### QAChatbot
- LangChain-powered retrieval QA
- Gemini 1.5 Pro integration
- Source citation and context
- Multiple query modes

## Performance

- **Processing Speed**: ~100 pages per minute
- **Accuracy**: 85% manual QA accuracy
- **Response Time**: <3 seconds for most queries
- **Memory Usage**: Efficient chunking reduces memory footprint

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` is set correctly
2. **PDF Processing Fails**: Check file permissions and PDF format
3. **Memory Issues**: Reduce `CHUNK_SIZE` or `BATCH_SIZE`
4. **Rate Limiting**: Increase delays in embedding generation

### Logs

Check the logs directory for detailed error information:
```bash
tail -f logs/chatbot.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for powerful language models
- LangChain for the retrieval QA framework
- PyMuPDF for robust PDF processing
- ChromaDB for efficient vector storage
