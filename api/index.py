import os
import sys

# Add the existing backend directories to Python's import path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
BACKEND_API_DIR = os.path.join(BACKEND_DIR, "api")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BACKEND_API_DIR)

# Use the existing FastAPI application
from server import app