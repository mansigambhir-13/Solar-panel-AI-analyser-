"""
Configuration settings for Solar Panel AI System
"""

import os
from pathlib import Path

class SystemConfig:
    """Configuration class for the solar panel AI system"""
    
    def __init__(self, config_path=None):
        # Base paths
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Ensure directories exist
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)
        
        # Model settings
        self.IMAGE_MODEL_PATH = self.MODELS_DIR / "dust_classifier.h5"
        self.CURRENT_PANEL_IMAGE_PATH = self.DATA_DIR / "current_panel.jpg"
        
        # Agent settings
        self.FORECAST_HOURS = 12
        self.GPIO_SPRAY_PIN = 18
        
        # System settings
        self.LOG_LEVEL = "INFO"
        self.MAX_LOG_SIZE_MB = 10
        
        # Load custom config if provided
        if config_path:
            self._load_custom_config(config_path)
    
    def _load_custom_config(self, config_path):
        """Load custom configuration from file"""
        try:
            import json
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Update settings
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            print(f"Warning: Could not load custom config: {e}")
