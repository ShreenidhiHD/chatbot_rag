#!/usr/bin/env python3
"""
FastAPI application for RAG Chat API
"""

import sys
import os
from pathlib import Path
import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import RAG system
from core.rag import advanced_rag_query

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Coffee RAG API",
    description="API for RAG-based coffee shop assistant",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history store (would use a database in production)
chat_sessions = {}

# Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    message: str = Field(..., description="User message")

class Product(BaseModel):
    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    buy_link: str = Field(..., description="Link to buy the product")
    image_url: str = Field(..., description="URL of the product image")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for the conversation")
    response: str = Field(..., description="Assistant response")
    intent: str = Field(..., description="Detected intent")
    agent: str = Field(..., description="Agent who handled the request")
    products: List[Product] = Field(default_factory=list, description="Products mentioned in the response")
    sources_count: int = Field(..., description="Number of sources used")
    chat_history: List[ChatMessage] = Field(..., description="Chat history")

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Coffee RAG API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that uses the RAG system to generate responses
    
    - Maintains chat history across requests using session_id
    - Generates responses using the advanced RAG system
    - Returns structured data including product recommendations
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize chat history if this is a new session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Retrieve chat history
        chat_history = chat_sessions[session_id]
        
        # Convert to format expected by RAG system
        rag_chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_history
        ]
        
        # Add user message to history
        user_message = ChatMessage(role="user", content=request.message)
        chat_history.append(user_message)
        
        # Call RAG system with chat history
        logger.info(f"Processing query: '{request.message[:50]}...' for session {session_id}")
        result = advanced_rag_query(
            query=request.message,
            chat_history=rag_chat_history,
            llm_provider="gemini"  # Using Gemini as default
        )
        
        # Add assistant response to history
        assistant_message = ChatMessage(role="assistant", content=result["response"])
        chat_history.append(assistant_message)
        
        # Update session
        chat_sessions[session_id] = chat_history
        
        # Return structured response
        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            intent=result.get("intent", "unknown"),
            agent=result.get("agent", "Assistant"),
            products=result.get("products", []),
            sources_count=len(result.get("sources", [])),
            chat_history=chat_history
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the FastAPI server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
