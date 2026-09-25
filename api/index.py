import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
BACKEND_API_DIR = os.path.join(BACKEND_DIR, "api")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BACKEND_API_DIR)

from server import app