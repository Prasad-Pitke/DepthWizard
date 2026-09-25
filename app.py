import os
import sys

from fastapi import FastAPI

# Make the existing backend importable
ROOT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

sys.path.insert(0, BACKEND_DIR)

# Import the existing DepthWizard FastAPI application
from api.server import app as depthwizard_app

# Vercel-recognized FastAPI application
app: FastAPI = depthwizard_app