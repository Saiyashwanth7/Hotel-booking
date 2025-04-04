# src/api.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import time
import os
from datetime import datetime

from .data_processing import DataProcessor
from .analytics import BookingAnalytics
from .rag_system import RAGSystem

app = FastAPI(
    title="Hotel Booking Analytics & QA API",
    description="API for hotel booking analytics and question answering",
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

# Models
class QuestionRequest(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None

# Global variables
DATA_PATH = "data/hotel_bookings.csv"
data_processor = None
booking_analytics = None
rag_system = None

# Startup event
@app.on_event("startup")
async def startup_event():
    global data_processor, booking_analytics, rag_system
    
    # Initialize data processor
    data_processor = DataProcessor(DATA_PATH)
    data = data_processor.get_processed_data()
    
    # Initialize analytics
    booking_analytics = BookingAnalytics(data)
    
    # Initialize RAG system
    rag_system = RAGSystem(data)
    # Start embedding creation in background
    background_tasks = BackgroundTasks()
    background_tasks.add_task(rag_system.create_document_embeddings)

# Dependency to get RAG system
def get_rag_system():
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")
    return rag_system

# Dependency to get booking analytics
def get_booking_analytics():
    if booking_analytics is None:
        raise HTTPException(status_code=503, detail="Analytics system not initialized yet")
    return booking_analytics

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Hotel Booking Analytics & QA API"}

@app.get("/health")
async def health_check():
    """Check system health status"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check data processor
    try:
        if data_processor and data_processor.data is not None:
            health_status["components"]["data_processor"] = {
                "status": "healthy",
                "records": len(data_processor.data)
            }
        else:
            health_status["components"]["data_processor"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["data_processor"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"
    
    # Check analytics
    try:
        if booking_analytics:
            health_status["components"]["analytics"] = {"status": "healthy"}
        else:
            health_status["components"]["analytics"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["analytics"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"
    
    # Check RAG system
    try:
        if rag_system:
            index_status = "initialized" if rag_system.index is not None else "not_initialized"
            health_status["components"]["rag_system"] = {
                "status": "healthy" if index_status == "initialized" else "initializing",
                "index_status": index_status,
                "documents_count": len(rag_system.documents) if rag_system.documents else 0
            }
            if index_status != "initialized":
                health_status["status"] = "degraded"
        else:
            health_status["components"]["rag_system"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["rag_system"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest, analytics: BookingAnalytics = Depends(get_booking_analytics)):
    """Get analytics reports"""
    try:
        start_time = time.time()
        
        # Apply filters if provided
        filtered_data = analytics.data
        if request.filters:
            for key, value in request.filters.items():
                if key in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[key] == value]
        
        # Generate analytics with filtered data
        filtered_analytics = BookingAnalytics(filtered_data)
        result = filtered_analytics.generate_all_analytics()
        
        # Add metadata
        result["metadata"] = {
            "total_records": len(filtered_data),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "filters_applied": request.filters
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest, rag: RAGSystem = Depends(get_rag_system)):
    """Answer booking-related questions"""
    try:
        start_time = time.time()
        
        # Check if RAG system is ready
        if rag.index is None:
            return {
                "status": "initializing",
                "message": "The question answering system is still initializing. Please try again in a few moments."
            }
        
        # Get answer from RAG system
        result = rag.answer_question(request.question)
        
        # Add metadata
        result["metadata"] = {
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.get("/query-history")
async def get_query_history(rag: RAGSystem = Depends(get_rag_system)):
    """Get query history"""
    try:
        return {"history": rag.get_query_history()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving query history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)