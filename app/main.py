"""
FastAPI Application for Chatbot API.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chatbot import get_chatbot, reset_chatbot


# ============ Pydantic Models ============

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=500, description="User message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how are you?"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    intent: str
    confidence: float
    understood: bool
    top_intents: dict
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    timestamp: str


class IntentsResponse(BaseModel):
    """Response model for available intents."""
    intents: List[str]
    count: int


class HistoryItem(BaseModel):
    """Single history item."""
    user: str
    bot: str
    intent: str
    confidence: float


class HistoryResponse(BaseModel):
    """Response model for conversation history."""
    history: List[HistoryItem]
    count: int


# ============ FastAPI App ============

app = FastAPI(
    title="AI Chatbot API",
    description="""
    A production-ready AI Chatbot API built with PyTorch and FastAPI.
    
    ## Features
    - Intent classification using Neural Networks
    - Natural language understanding
    - Conversation history tracking
    - RESTful API endpoints
    
    ## Usage
    Send a POST request to `/chat` with your message to get a response.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


# ============ Endpoints ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    index_path = os.path.join(static_dir, 'index.html')
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return HTMLResponse(content="""
    <html>
        <head><title>Chatbot API</title></head>
        <body>
            <h1>🤖 AI Chatbot API</h1>
            <p>Welcome to the AI Chatbot API!</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and model are running."""
    try:
        chatbot = get_chatbot()
        model_loaded = chatbot is not None
    except Exception:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response.
    
    - **message**: The user's message (1-500 characters)
    
    Returns the bot's response along with intent classification details.
    """
    try:
        chatbot = get_chatbot()
        result = chatbot.chat(request.message)
        
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            confidence=result["confidence"],
            understood=result["understood"],
            top_intents=result["top_intents"],
            timestamp=datetime.now().isoformat()
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.get("/intents", response_model=IntentsResponse, tags=["Chat"])
async def get_intents():
    """Get list of available intents the chatbot can recognize."""
    try:
        chatbot = get_chatbot()
        intents = chatbot.get_available_intents()
        
        return IntentsResponse(
            intents=intents,
            count=len(intents)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting intents: {str(e)}"
        )


@app.get("/history", response_model=HistoryResponse, tags=["Chat"])
async def get_history():
    """Get conversation history for the current session."""
    try:
        chatbot = get_chatbot()
        history = chatbot.get_history()
        
        return HistoryResponse(
            history=[HistoryItem(**item) for item in history],
            count=len(history)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting history: {str(e)}"
        )


@app.delete("/history", tags=["Chat"])
async def clear_history():
    """Clear the conversation history."""
    try:
        chatbot = get_chatbot()
        chatbot.clear_history()
        
        return {"message": "History cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing history: {str(e)}"
        )


@app.post("/reset", tags=["System"])
async def reset_bot():
    """Reset the chatbot instance (reloads model)."""
    try:
        reset_chatbot()
        get_chatbot()  # Reinitialize
        
        return {"message": "Chatbot reset successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting chatbot: {str(e)}"
        )


# ============ Startup/Shutdown Events ============

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    print("🚀 Starting Chatbot API...")
    try:
        get_chatbot()
        print("✅ Chatbot loaded successfully!")
    except FileNotFoundError:
        print("⚠️ Model not found. Please train the model first.")
    except Exception as e:
        print(f"⚠️ Error loading chatbot: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down Chatbot API...")
