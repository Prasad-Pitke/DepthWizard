"""
server.py
----------
FastAPI application powering the Interactive Visualization Platform.

Endpoints:
  POST /api/process        upload an image (+optional reference DEM / GCP csv)
                            -> runs the DepthWizardPipeline -> returns a job id
  GET  /api/jobs/{job_id}   job status + result metadata
  GET  /api/jobs/{job_id}/viewer/{filename}
                            serves texture.jpg / heightmap.png / meta.json
                            for the Three.js viewer
  GET  /api/jobs/{job_id}/dsm.tif       download the DSM GeoTIFF
  GET  /api/jobs/{job_id}/terrain.glb   download the textured mesh
  POST /api/jobs/{job_id}/validate      validate the job's DSM against an
                                         uploaded reference elevation raster

Run with:
    uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from depthwizard.pipeline import DepthWizardPipeline  # noqa: E402
from depthwizard.mesh import MeshBuildConfig  # noqa: E402

APP_DATA_DIR = os.environ.get("DEPTHWIZARD_DATA_DIR", os.path.join(tempfile.gettempdir(), "depthwizard_jobs"))
os.makedirs(APP_DATA_DIR, exist_ok=True)

app = FastAPI(title="DepthWizard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

JOBS = {}          # job_id -> dict(status, result, error, dir)
JOBS_LOCK = threading.Lock()


def _job_dir(job_id: str) -> str:
    return os.path.join(APP_DATA_DIR, job_id)


def _run_job(job_id: str, image_path: str, backbone: str, reference_dem_path: Optional[str],
             gcps_path: Optional[str], pixel_size_m: Optional[float], vertical_exaggeration: float,
             mesh_resolution: int, reconstruction_mode: str = "relief",
             building_min_height_m: float = 4.0, building_max_height_m: float = 25.0,
             building_min_area_px: int = 60, building_max_area_frac: float = 0.15,
             building_min_extent: float = 0.35, building_max_aspect_ratio: float = 6.0,
             building_min_width_px: float = 6.0, building_min_separation_px: int = 12):
    try:
        gcps = None
        if gcps_path:
            gcps = []
            with open(gcps_path) as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    try:
                        gcps.append((int(row[0]), int(row[1]), float(row[2])))
                    except ValueError:
                        continue  # header row

        pipeline = DepthWizardPipeline(
            backbone=backbone, mesh_config=MeshBuildConfig(max_mesh_resolution=mesh_resolution)
        )
        result = pipeline.run(
            image_path=image_path, output_dir=_job_dir(job_id),
            reference_dem_path=reference_dem_path, gcps=gcps,
            pixel_size_m=pixel_size_m, vertical_exaggeration=vertical_exaggeration,
            reconstruction_mode=reconstruction_mode,
            building_min_height_m=building_min_height_m, building_max_height_m=building_max_height_m,
            building_min_area_px=building_min_area_px, building_max_area_frac=building_max_area_frac,
            building_min_extent=building_min_extent, building_max_aspect_ratio=building_max_aspect_ratio,
            building_min_width_px=building_min_width_px, building_min_separation_px=building_min_separation_px,
        )
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result.to_dict()
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)


@app.post("/api/process")
async def process(
    image: UploadFile = File(...),
    backbone: str = Form("auto"),
    reference_dem: Optional[UploadFile] = File(None),
    gcps_csv: Optional[UploadFile] = File(None),
    pixel_size_m: Optional[float] = Form(None),
    vertical_exaggeration: float = Form(1.0),
    mesh_resolution: int = Form(512),
    reconstruction_mode: str = Form("relief"),
    building_min_height_m: float = Form(4.0),
    building_max_height_m: float = Form(25.0),
    building_min_area_px: int = Form(60),
    building_max_area_frac: float = Form(0.15),
    building_min_extent: float = Form(0.35),
    building_max_aspect_ratio: float = Form(6.0),
    building_min_width_px: float = Form(6.0),
    building_min_separation_px: int = Form(12),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    image_path = os.path.join(job_dir, "input_" + image.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    ref_path = None
    if reference_dem is not None and reference_dem.filename:
        ref_path = os.path.join(job_dir, "reference_" + reference_dem.filename)
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference_dem.file, f)

    gcps_path = None
    if gcps_csv is not None and gcps_csv.filename:
        gcps_path = os.path.join(job_dir, "gcps.csv")
        with open(gcps_path, "wb") as f:
            shutil.copyfileobj(gcps_csv.file, f)

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "result": None, "error": None, "dir": job_dir}

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, image_path, backbone, ref_path, gcps_path, pixel_size_m, vertical_exaggeration, mesh_resolution,
              reconstruction_mode, building_min_height_m, building_max_height_m,
              building_min_area_px, building_max_area_frac, building_min_extent,
              building_max_aspect_ratio, building_min_width_px, building_min_separation_px),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return {"job_id": job_id, "status": job["status"], "result": job["result"], "error": job["error"]}


@app.get("/api/jobs/{job_id}/viewer/{filename}")
async def viewer_asset(job_id: str, filename: str):
    path = os.path.join(_job_dir(job_id), "viewer_bundle", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/dsm.tif")
async def download_dsm(job_id: str):
    path = os.path.join(_job_dir(job_id), "dsm.tif")
    if not os.path.isfile(path):
        raise HTTPException(404, "DSM not found")
    return FileResponse(path, filename="dsm.tif")


@app.get("/api/jobs/{job_id}/terrain.glb")
async def download_glb(job_id: str):
    path = os.path.join(_job_dir(job_id), "terrain.glb")
    if not os.path.isfile(path):
        raise HTTPException(404, "Mesh not found")
    return FileResponse(path, filename="terrain.glb")


@app.get("/api/jobs/{job_id}/buildings.json")
async def download_buildings(job_id: str):
    path = os.path.join(_job_dir(job_id), "buildings.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No building footprints for this job (was it run in 'buildings' mode?)")
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/jobs/{job_id}/calibration_report.json")
async def calibration_report(job_id: str):
    path = os.path.join(_job_dir(job_id), "calibration_report.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "Calibration report not found")
    with open(path) as f:
        return JSONResponse(json.load(f))


@app.post("/api/jobs/{job_id}/validate")
async def validate_job(job_id: str, reference: UploadFile = File(...)):
    from depthwizard.pipeline import DepthWizardPipeline
    from depthwizard import geo_io

    job_dir = _job_dir(job_id)
    dsm_path = os.path.join(job_dir, "dsm.tif")
    if not os.path.isfile(dsm_path):
        raise HTTPException(404, "DSM for this job not found (job may still be running)")

    ref_path = os.path.join(job_dir, "validate_ref_" + reference.filename)
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    input_candidates = [f for f in os.listdir(job_dir) if f.startswith("input_")]
    rgb = None
    if input_candidates:
        loaded = geo_io.load_image(os.path.join(job_dir, input_candidates[0]))
        rgb = loaded.rgb

    pipeline = DepthWizardPipeline(backbone="classical")  # backbone unused for validation
    report = pipeline.validate(dsm_path, ref_path, rgb_for_stratification=rgb)
    return report.to_dict()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the static frontend (Three.js viewer) if present, so the whole
# app can be run as a single standalone process.
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
