// api.js — thin client for the DepthWizard FastAPI backend.
// If the app is served from the same origin as the API (recommended,
// see backend/api/server.py static mount), API_BASE can stay empty.
export const API_BASE = window.DEPTHWIZARD_API_BASE || "";

export async function submitJob({ image, backbone, calibMode, demFile, gcpFile, verticalExaggeration, meshResolution, reconstructionMode, buildingMinHeight, buildingMaxHeight }) {
  const form = new FormData();
  form.append("image", image);
  form.append("backbone", backbone);
  form.append("vertical_exaggeration", String(verticalExaggeration));
  form.append("mesh_resolution", String(meshResolution));
  form.append("reconstruction_mode", reconstructionMode || "relief");
  if (buildingMinHeight != null) form.append("building_min_height_m", String(buildingMinHeight));
  if (buildingMaxHeight != null) form.append("building_max_height_m", String(buildingMaxHeight));
  if (calibMode === "dem" && demFile) form.append("reference_dem", demFile);
  if (calibMode === "gcp" && gcpFile) form.append("gcps_csv", gcpFile);

  const res = await fetch(`${API_BASE}/api/process`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
  return res.json(); // { job_id, status }
}

export async function pollJob(jobId, { intervalMs = 1200, onTick } = {}) {
  while (true) {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`Job status check failed: ${res.status}`);
    const data = await res.json();
    if (onTick) onTick(data);
    if (data.status === "done" || data.status === "error") return data;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export function viewerAssetUrl(jobId, filename) {
  return `${API_BASE}/api/jobs/${jobId}/viewer/${filename}`;
}

export function dsmDownloadUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/dsm.tif`;
}

export function glbDownloadUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/terrain.glb`;
}

export async function validateJob(jobId, referenceFile) {
  const form = new FormData();
  form.append("reference", referenceFile);
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/validate`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Validation failed: ${res.status} ${await res.text()}`);
  return res.json();
}
