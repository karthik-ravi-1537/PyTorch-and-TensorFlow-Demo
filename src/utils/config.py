"""
Configuration Management Utilities

Shared configuration management for ML experiments.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]) -> None:
        """Update configuration with another dictionary."""
        self._deep_update(self._config, other_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Load configuration from file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Config file {file_path} not found, using empty config")
            return cls()
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(config_dict)
    
    def save(self, file_path: str) -> None:
        """Save configuration to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(self._config, f, default_flow_style=False)
            elif path.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")


# Default configurations
DEFAULT_PYTORCH_CONFIG = {
    'model': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'device': 'cpu'
    },
    'data': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1
    }
}

DEFAULT_TENSORFLOW_CONFIG = {
    'model': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    },
    'data': {
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1
    }
}