import os
import sys

# Make the existing backend package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from api.server import app