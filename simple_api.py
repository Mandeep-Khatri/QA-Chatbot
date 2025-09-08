"""Simple API without complex dependencies."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from datetime import datetime

# Simple FastAPI app
app = FastAPI(
    title="Gemini by Mandy",
    description="Simple API for Mandy's chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    timestamp: str
    status: str

# Simple endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Gemini by Mandy",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def simple_query(request: QueryRequest):
    """Simple query endpoint."""
    try:
        # Simple response without AI for now
        response_text = f"Hello! You asked: '{request.query}'. This is a simple response from Mandy's API."
        
        return QueryResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
