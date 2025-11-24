"""
FastAPI application for car recommendations RAG chatbot.
"""

import logging
from typing import Optional, Dict, Any, List
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag.chain import create_chain, query_chain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Car Recommendations RAG Chatbot",
    description="RAG-based chatbot for car recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chains per session (in production, use Redis or similar)
session_chains: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    recommended: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]


def get_or_create_chain(session_id: str, filters: Optional[Dict[str, Any]] = None):
    """
    Get or create chain for a session.
    
    Args:
        session_id: Session identifier
        filters: Metadata filters
        
    Returns:
        ConversationalRetrievalChain instance
    """
    # Create a key based on session_id and filters
    chain_key = f"{session_id}_{hash(str(filters))}"
    
    if chain_key not in session_chains:
        logger.info(f"Creating new chain for session: {session_id}")
        chain = create_chain(filters=filters)
        session_chains[chain_key] = chain
    else:
        logger.info(f"Using existing chain for session: {session_id}")
        chain = session_chains[chain_key]
    
    return chain


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Car Recommendations RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Chat endpoint for car recommendations",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for car recommendations.
    
    Args:
        request: Chat request with query, filters, and session_id
        
    Returns:
        Chat response with answer, recommended cars, and sources
    """
    try:
        logger.info(f"Received chat request: query='{request.query}', filters={request.filters}, session_id={request.session_id}")
        
        # Generate session ID if not provided
        session_id = request.session_id or "default"
        
        # Get or create chain
        chain = get_or_create_chain(session_id, request.filters)
        
        # Query chain
        result = query_chain(chain, request.query)
        
        logger.info(f"Generated response with {len(result['recommended'])} recommendations")
        
        return ChatResponse(
            answer=result["answer"],
            recommended=result["recommended"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

