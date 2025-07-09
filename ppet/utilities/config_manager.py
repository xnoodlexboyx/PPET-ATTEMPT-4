import json
import yaml
from typing import Dict, Any, Optional
import os
from pathlib import Path

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
    
    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_path.suffix}"
        )
    
    return validate_config(config)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Validated configuration dictionary
    """
    # Define required fields and their types
    required_fields = {
        'puf_params': dict,
        'auth_threshold': float,
        'logging': dict
    }
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in config:
            # Use defaults if field missing
            config[field] = get_default_config()[field]
        elif not isinstance(config[field], field_type):
            raise ValueError(
                f"Invalid type for {field}: expected {field_type}, "
                f"got {type(config[field])}"
            )
    
    # Validate PUF parameters
    puf_params = config['puf_params']
    required_puf_params = {
        'challenge_length': int,
        'noise_sigma': float,
        'variation_sigma': float
    }
    
    for param, param_type in required_puf_params.items():
        if param not in puf_params:
            puf_params[param] = get_default_config()['puf_params'][param]
        elif not isinstance(puf_params[param], param_type):
            raise ValueError(
                f"Invalid type for puf_params.{param}: "
                f"expected {param_type}, got {type(puf_params[param])}"
            )
    
    # Validate logging configuration
    logging_config = config['logging']
    required_logging_params = {
        'level': str,
        'file': str,
        'format': str
    }
    
    for param, param_type in required_logging_params.items():
        if param not in logging_config:
            logging_config[param] = get_default_config()['logging'][param]
        elif not isinstance(logging_config[param], param_type):
            raise ValueError(
                f"Invalid type for logging.{param}: "
                f"expected {param_type}, got {type(logging_config[param])}"
            )
    
    return config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'puf_params': {
            'challenge_length': 64,
            'noise_sigma': 0.1,
            'variation_sigma': 0.2
        },
        'auth_threshold': 0.9,
        'logging': {
            'level': 'INFO',
            'file': 'ppet.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_path.suffix}"
        ) 