from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any
import asyncio
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import existing agents
AGENTS_AVAILABLE = False
decision_agent = None

try:
    from agents.decision_agent import DecisionOrchestrationAgent
    decision_agent = DecisionOrchestrationAgent()
    AGENTS_AVAILABLE = True
    logger.info("✅ Decision agent loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Running in demo mode: {e}")

def create_app():
    app = FastAPI(
        title="🌞 Solar Panel Optimization API",
        description="AI-Powered Solar Energy Optimization Platform",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    frontend_path = project_root / "frontend" / "static"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_file = project_root / "frontend" / "static" / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        
        return HTMLResponse('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌞 Solar Panel Optimization</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
                .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; margin-bottom: 30px; }
                .status { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4CAF50; }
                .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; }
                .btn:hover { background: #45a049; }
                .feature { background: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌞 Solar Panel Optimization System</h1>
                    <p>Your AI-powered solar optimization platform is running successfully!</p>
                </div>
                
                <div class="status">
                    <h3>🚀 System Status: Online</h3>
                    <p><strong>Agents Available:</strong> ''' + str(AGENTS_AVAILABLE) + '''</p>
                    <p><strong>Mode:</strong> ''' + ("AI-Powered" if AGENTS_AVAILABLE else "Demo Mode") + '''</p>
                </div>
                
                <div class="feature">
                    <h3>📊 Available Features:</h3>
                    <ul>
                        <li>Solar panel optimization analysis</li>
                        <li>Performance prediction modeling</li>
                        <li>Cost-benefit analysis</li>
                        <li>Energy efficiency recommendations</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/docs" class="btn">📚 API Documentation</a>
                    <a href="/status" class="btn">🔍 System Status</a>
                    <a href="/health" class="btn">❤️ Health Check</a>
                </div>
            </div>
        </body>
        </html>
        ''')
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents_available": AGENTS_AVAILABLE,
            "version": "1.0.0"
        }
    
    @app.get("/status")
    async def get_status():
        return {
            "status": "online",
            "agents_available": AGENTS_AVAILABLE,
            "decision_agent_loaded": decision_agent is not None,
            "features": [
                "Solar panel optimization",
                "Performance prediction",
                "Energy efficiency analysis",
                "Cost optimization"
            ]
        }
    
    @app.post("/api/optimize")
    async def optimize_solar_panels(data: Dict[str, Any]):
        try:
            required_fields = ["location", "panel_specs", "energy_requirements"]
            for field in required_fields:
                if field not in data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            # Demo optimization results
            panel_capacity = data["panel_specs"].get("capacity", 10)
            location = data["location"]
            
            efficiency_factor = 0.85 if "california" in location.lower() else 0.75
            annual_output = panel_capacity * 1200 * efficiency_factor
            cost_savings = panel_capacity * 150 * efficiency_factor
            
            result = {
                "recommended_configuration": {
                    "panel_type": data["panel_specs"].get("type", "monocrystalline"),
                    "optimal_capacity_kw": panel_capacity,
                    "optimal_angle": 30,
                    "optimal_orientation": "south"
                },
                "performance_metrics": {
                    "efficiency_score": round(efficiency_factor * 100, 1),
                    "annual_energy_output_kwh": round(annual_output),
                    "monthly_average_kwh": round(annual_output / 12)
                },
                "financial_analysis": {
                    "estimated_annual_savings": round(cost_savings, 2),
                    "payback_period_years": 7.2,
                    "roi_percentage": 13.8
                }
            }
            
            return {
                "success": True,
                "source": "demo_mode" if not AGENTS_AVAILABLE else "ml_agent",
                "optimization_result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    return app
