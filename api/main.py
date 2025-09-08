"""FastAPI REST API for Mandy's Q&A Chatbot."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging
from datetime import datetime

# Import our chatbot components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cache.cache_manager import CacheManager
from src.utils.smart_chunker import SmartChunker

# Import working chatbot functions
try:
    from working_chatbot import get_gemini_model, generate_response
except ImportError:
    # Fallback if working_chatbot is not available
    def get_gemini_model():
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        return genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_response(model, query, context=""):
        prompt = f"{context}\n\nUser: {query}\nAssistant:"
        response = model.generate_content(prompt)
        return response.text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mandy's Q&A Chatbot API",
    description="REST API for Mandy's intelligent Q&A chatbot powered by Gemini 1.5 Flash",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
cache_manager = CacheManager()
smart_chunker = SmartChunker()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = ""
    use_cache: bool = True

class QueryResponse(BaseModel):
    response: str
    query: str
    timestamp: str
    cached: bool
    processing_time: float

class ChunkRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class CacheStatsResponse(BaseModel):
    embedding_cache: Dict[str, Any]
    response_cache: Dict[str, Any]
    overall_stats: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        components={
            "cache_manager": "active",
            "smart_chunker": "active",
            "gemini_model": "active"
        }
    )

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """Query the chatbot with caching support."""
    start_time = datetime.now()
    
    try:
        # Check cache first if enabled
        cached_response = None
        if request.use_cache:
            cached_response = cache_manager.response_cache.get_response(
                request.query, request.context
            )
        
        if cached_response:
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                response=cached_response,
                query=request.query,
                timestamp=start_time.isoformat(),
                cached=True,
                processing_time=processing_time
            )
        
        # Generate new response
        model = get_gemini_model()
        response = generate_response(model, request.query, request.context)
        
        # Cache the response
        if request.use_cache:
            cache_manager.response_cache.set_response(
                request.query, response, request.context
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            response=response,
            query=request.query,
            timestamp=start_time.isoformat(),
            cached=False,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text chunking endpoint
@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(request: ChunkRequest):
    """Chunk text using smart chunking strategies."""
    try:
        # Update chunker settings
        smart_chunker.chunk_size = request.chunk_size
        smart_chunker.chunk_overlap = request.chunk_overlap
        
        # Chunk the text
        chunks = smart_chunker.chunk_text(request.text)
        
        # Add overlap if requested
        if request.chunk_overlap > 0:
            chunks = smart_chunker.add_overlap(chunks)
        
        # Get statistics
        statistics = smart_chunker.get_chunk_statistics(chunks)
        
        return ChunkResponse(
            chunks=chunks,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics."""
    try:
        embedding_stats = cache_manager.embedding_cache.get_cache_stats()
        response_stats = cache_manager.response_cache.get_cache_stats()
        overall_stats = cache_manager.get_cache_stats()
        
        return CacheStatsResponse(
            embedding_cache=embedding_stats,
            response_cache=response_stats,
            overall_stats=overall_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache."""
    try:
        cache_manager.clear_all_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/clear-expired")
async def clear_expired_cache():
    """Clear expired cache entries."""
    try:
        expired_responses = cache_manager.response_cache.clear_expired_responses()
        cache_manager.clear_expired_cache()
        
        return {
            "message": "Expired cache cleared successfully",
            "expired_responses_removed": expired_responses
        }
        
    except Exception as e:
        logger.error(f"Error clearing expired cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a file."""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file in background
        background_tasks.add_task(process_uploaded_file, file_path, file.filename)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_uploaded_file(file_path: str, filename: str):
    """Process uploaded file in background."""
    try:
        # Here you would add file processing logic
        # For now, just log the file
        logger.info(f"Processing file: {filename} at {file_path}")
        
        # You could add PDF processing, text extraction, etc.
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")

# Recent queries endpoint
@app.get("/recent-queries")
async def get_recent_queries(limit: int = 10):
    """Get recent queries from cache."""
    try:
        recent_queries = cache_manager.response_cache.get_recent_queries(limit)
        return {"recent_queries": recent_queries}
        
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Mandy's Q&A Chatbot API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "/query",
            "chunk": "/chunk",
            "cache_stats": "/cache/stats",
            "upload": "/upload",
            "recent_queries": "/recent-queries"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
