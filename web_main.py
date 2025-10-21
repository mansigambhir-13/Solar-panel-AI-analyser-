import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the FastAPI app
from backend.api.main import create_app

def main():
    app = create_app()
    port = int(os.environ.get("PORT", 8000))
    
    print("🌞 Starting Solar Panel Optimization Web Server...")
    print(f"🚀 Server will be available at: http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print(f"🔍 Health Check: http://localhost:{port}/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
