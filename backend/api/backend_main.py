# Save as: backend/api/main.py
"""
FastAPI main application
Integrates with existing Solar Panel Optimization agents
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime

# Add project root to path to import existing agents
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your existing agents
try:
    from agents.decision_agent import DecisionOrchestrationAgent
    from utils.config_loader import ConfigLoader
    from utils.logger_utils import setup_logger
    AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import agents: {e}")
    AGENTS_AVAILABLE = False

def create_app():
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Solar Panel Optimization API",
        description="Web interface for ML-powered solar panel optimization",
        version="1.0.0"
    )
    
    # CORS middleware for web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize agents if available
    decision_agent = None
    if AGENTS_AVAILABLE:
        try:
            decision_agent = DecisionOrchestrationAgent()
            logging.info("Decision agent initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize decision agent: {e}")
    
    # Mount static files
    static_path = project_root / "frontend" / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main dashboard"""
        html_file = project_root / "frontend" / "static" / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        return """
        <html>
            <head><title>Solar Panel Optimization</title></head>
            <body>
                <h1>🌞 Solar Panel Optimization Dashboard</h1>
                <p>Your solar optimization system is running!</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for Railway"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents_available": AGENTS_AVAILABLE,
            "version": "1.0.0"
        }
    
    @app.get("/status")
    async def get_status():
        """Get system status"""
        return {
            "status": "online",
            "agents_available": AGENTS_AVAILABLE,
            "features": [
                "Solar panel optimization",
                "Performance prediction",
                "Energy efficiency analysis",
                "Cost optimization"
            ],
            "models_loaded": decision_agent is not None
        }
    
    @app.post("/api/optimize")
    async def optimize_solar_panels(data: Dict[str, Any]):
        """
        Main optimization endpoint
        Accepts solar panel configuration and returns optimization results
        """
        if not AGENTS_AVAILABLE or decision_agent is None:
            raise HTTPException(
                status_code=503, 
                detail="Optimization agents not available"
            )
        
        try:
            # Validate input data
            required_fields = ["location", "panel_specs", "energy_requirements"]
            for field in required_fields:
                if field not in data:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required field: {field}"
                    )
            
            # Run optimization using your existing agent
            result = await asyncio.to_thread(
                decision_agent.orchestrate_comprehensive_decision,
                data
            )
            
            return {
                "success": True,
                "optimization_result": result,
                "timestamp": datetime.now().isoformat(),
                "processing_time": "calculated_in_agent"
            }
            
        except Exception as e:
            logging.error(f"Optimization error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {str(e)}"
            )
    
    @app.post("/api/analyze")
    async def analyze_performance(data: Dict[str, Any]):
        """
        Performance analysis endpoint
        """
        if not AGENTS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Analysis agents not available"
            )
        
        try:
            # Mock analysis - replace with your actual analysis logic
            analysis_result = {
                "efficiency_score": 85.2,
                "energy_output": data.get("panel_capacity", 0) * 0.85,
                "cost_savings": 1250.50,
                "recommendations": [
                    "Adjust panel angle by 5 degrees",
                    "Clean panels monthly",
                    "Consider battery storage"
                ]
            }
            
            return {
                "success": True,
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
    
    @app.get("/api/models")
    async def get_available_models():
        """Get information about available ML models"""
        models_info = {
            "decision_agent": AGENTS_AVAILABLE,
            "models_path": str(project_root / "models"),
            "available_models": []
        }
        
        # Check for model files
        models_path = project_root / "models"
        if models_path.exists():
            models_info["available_models"] = [
                f.name for f in models_path.glob("*.pkl") 
                if f.is_file()
            ]
        
        return models_info
    
    return app

# For development
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)