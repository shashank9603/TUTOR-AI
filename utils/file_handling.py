"""Utilities for file handling."""

import os
import tempfile
import shutil
import logging
from typing import Optional, List, Tuple, Dict, Any
import json

logger = logging.getLogger("file_handling")

def create_temporary_directory() -> str:
    """Create a temporary directory for file processing."""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def clean_temporary_directory(temp_dir: str) -> bool:
    """Clean up a temporary directory."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")
        return False

def save_uploaded_file(uploaded_file, target_dir: str) -> Optional[str]:
    """Save an uploaded file to a target directory."""
    if not uploaded_file:
        return None
        
    try:
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def save_data_to_json(data: Any, filename: str) -> bool:
    """Save data to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        return False

def load_data_from_json(filename: str) -> Optional[Any]:
    """Load data from a JSON file."""
    try:
        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return None
            
        with open(filename, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        return None