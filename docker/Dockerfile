# DepthWizard — standalone deployment image
# Serves both the FastAPI backend and the Three.js frontend from a
# single container/process.
FROM python:3.11-slim

# System deps for rasterio/GDAL and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Optional: uncomment to bake in a learned depth backbone for production
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
#     && pip install --no-cache-dir transformers timm

COPY backend /app/backend
COPY frontend /app/frontend

ENV DEPTHWIZARD_DATA_DIR=/data/jobs
RUN mkdir -p /data/jobs
VOLUME ["/data/jobs"]

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
