// main.js — DepthWizard Interactive Visualization Platform
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { loadTerrain, loadBuildingMesh } from "./terrain.js";
import { submitJob, pollJob, viewerAssetUrl, dsmDownloadUrl, glbDownloadUrl, validateJob } from "./api.js";

// ------------------------------------------------------------------
// DOM references
// ------------------------------------------------------------------
const el = (id) => document.getElementById(id);
const dropzone = el("dropzone");
const fileImageInput = el("file-image");
const fileImageName = el("file-image-name");
const selectBackbone = el("select-backbone");
const calibRadios = document.querySelectorAll('input[name="calib-mode"]');
const calibDemRow = el("calib-dem-row");
const calibGcpRow = el("calib-gcp-row");
const fileDem = el("file-dem");
const fileGcps = el("file-gcps");
const rangeExaggeration = el("range-exaggeration");
const valExaggeration = el("val-exaggeration");
const rowExaggeration = el("row-exaggeration");
const reconModeRadios = document.querySelectorAll('input[name="recon-mode"]');
const buildingHeightControls = el("building-height-controls");
const rangeBuildingMin = el("range-building-min");
const valBuildingMin = el("val-building-min");
const rangeBuildingMax = el("range-building-max");
const valBuildingMax = el("val-building-max");
const rangeMeshRes = el("range-mesh-res");
const valMeshRes = el("val-mesh-res");
const btnProcess = el("btn-process");
const progressLog = el("progress-log");
const jobStatusPill = el("job-status-pill");
const sectionResults = el("section-results");
const resultsSummary = el("results-summary");
const dlDsm = el("dl-dsm");
const dlGlb = el("dl-glb");
const sectionPreviews = el("section-previews");
const previewDepth = el("preview-depth");
const previewHeight = el("preview-height");
const previewColormap = el("preview-colormap");
const sectionValidate = el("section-validate");
const fileValidateRef = el("file-validate-ref");
const btnValidate = el("btn-validate");
const validationResults = el("validation-results");
const btnTogglePanel = el("btn-toggle-panel");
const panel = el("panel");
const btnPanelMinimize = el("btn-panel-minimize");
const btnResultsRailToggle = el("btn-results-rail-toggle");
const resultsRail = el("results-rail");
btnResultsRailToggle?.classList.add("results-toggle-hidden");
const viewportHint = el("viewport-hint");
const hud = el("hud");
const hudMode = el("hud-mode");
const hudElev = el("hud-elev");
const hudCamH = el("hud-cam-h");
const controlsHelp = el("controls-help");
const viewportToolbar = el("viewport-toolbar");
const btnModeOrbit = el("btn-mode-orbit");
const btnModeFly = el("btn-mode-fly");
const btnWireframe = el("btn-wireframe");
const btnColormap = el("btn-colormap");
const btnResetView = el("btn-reset-view");

let selectedImageFile = null;
let currentJobId = null;

// ------------------------------------------------------------------
// UI wiring: uploads, calibration mode, sliders
// ------------------------------------------------------------------
dropzone.addEventListener("click", () => fileImageInput.click());
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) setSelectedImage(f);
});
fileImageInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) setSelectedImage(f);
});

function setSelectedImage(file) {
  selectedImageFile = file;
  fileImageName.textContent = file.name;
  btnProcess.disabled = false;
}

calibRadios.forEach((r) =>
  r.addEventListener("change", () => {
    calibDemRow.classList.toggle("hidden", r.value !== "dem" || !r.checked);
    calibGcpRow.classList.toggle("hidden", r.value !== "gcp" || !r.checked);
    document.querySelectorAll('input[name="calib-mode"]').forEach((rr) => {
      if (rr.value === "dem") calibDemRow.classList.toggle("hidden", !rr.checked);
      if (rr.value === "gcp") calibGcpRow.classList.toggle("hidden", !rr.checked);
    });
  })
);

rangeExaggeration.addEventListener("input", () => {
  valExaggeration.textContent = `${parseFloat(rangeExaggeration.value).toFixed(1)}×`;
});
rangeMeshRes.addEventListener("input", () => {
  valMeshRes.textContent = rangeMeshRes.value;
});
rangeBuildingMin.addEventListener("input", () => {
  valBuildingMin.textContent = `${rangeBuildingMin.value} m`;
});
rangeBuildingMax.addEventListener("input", () => {
  valBuildingMax.textContent = `${rangeBuildingMax.value} m`;
});

function currentReconMode() {
  return document.querySelector('input[name="recon-mode"]:checked').value;
}

reconModeRadios.forEach((r) =>
  r.addEventListener("change", () => {
    const isBuildings = currentReconMode() === "buildings";
    buildingHeightControls.classList.toggle("hidden", !isBuildings);
    rowExaggeration.classList.toggle("hidden", isBuildings);
  })
);

btnTogglePanel.addEventListener("click", () => panel.classList.toggle("collapsed"));
btnPanelMinimize?.addEventListener("click", () => panel.classList.toggle("collapsed"));
btnResultsRailToggle?.addEventListener("click", () => {
  const collapsed = resultsRail.classList.toggle("collapsed");
  btnResultsRailToggle.setAttribute("aria-expanded", String(!collapsed));
  btnResultsRailToggle.textContent = collapsed ? "›" : "‹";
  btnResultsRailToggle.title = collapsed ? "Expand results panel" : "Collapse results panel";
});

// ------------------------------------------------------------------
// Job submission
// ------------------------------------------------------------------
btnProcess.addEventListener("click", async () => {
  if (!selectedImageFile) return;
  btnProcess.disabled = true;
  progressLog.classList.remove("hidden");
  logProgress("Uploading imagery…");
  setStatusPill("running", "Running");

  const calibMode = document.querySelector('input[name="calib-mode"]:checked').value;
  const reconMode = currentReconMode();

  try {
    const { job_id } = await submitJob({
      image: selectedImageFile,
      backbone: selectBackbone.value,
      calibMode,
      demFile: fileDem.files?.[0],
      gcpFile: fileGcps.files?.[0],
      verticalExaggeration: parseFloat(rangeExaggeration.value),
      meshResolution: parseInt(rangeMeshRes.value, 10),
      reconstructionMode: reconMode,
      buildingMinHeight: parseFloat(rangeBuildingMin.value),
      buildingMaxHeight: parseFloat(rangeBuildingMax.value),
    });
    currentJobId = job_id;
    logProgress(`Job ${job_id} queued — running depth backbone & calibration…`);

    const final = await pollJob(job_id, {
      onTick: (d) => { if (d.status === "running") logProgress("…still processing"); },
    });

    if (final.status === "error") {
      setStatusPill("error", "Error");
      logProgress(`Error: ${final.error}`);
      btnProcess.disabled = false;
      return;
    }

    setStatusPill("done", "Done");
    logProgress("Complete. Loading 3D terrain…");
    await displayResults(job_id, final.result);
  } catch (err) {
    setStatusPill("error", "Error");
    logProgress(`Error: ${err.message}`);
  } finally {
    btnProcess.disabled = false;
  }
});

function logProgress(msg) {
  const line = document.createElement("div");
  line.textContent = `› ${msg}`;
  progressLog.appendChild(line);
  progressLog.scrollTop = progressLog.scrollHeight;
}

function setStatusPill(kind, text) {
  jobStatusPill.className = `pill ${kind}`;
  jobStatusPill.textContent = text;
  jobStatusPill.classList.remove("hidden");
}

async function displayResults(jobId, result) {
  sectionResults.hidden = false;
  sectionValidate.hidden = false;
  btnResultsRailToggle?.classList.remove("results-toggle-hidden");
  const badgeClass = result.is_absolute ? "" : "relative";
  const badgeText = result.is_absolute ? "Absolute DSM" : "Relative rDSM";
  let calibLine = "";
  if (result.calibration_rmse != null) {
    calibLine = `<div>Calibration RMSE: <b>${result.calibration_rmse.toFixed(2)} m</b> · MAE: <b>${result.calibration_mae.toFixed(2)} m</b>` +
      (result.calibration_r2 != null ? ` · R²: <b>${result.calibration_r2.toFixed(3)}</b>` : "") + `</div>`;
  }
  resultsSummary.innerHTML = `
    <div>Method: <b>${result.calibration_method}</b> <span class="badge ${badgeClass}">${badgeText}</span></div>
    ${calibLine}
  `;
  dlDsm.href = dsmDownloadUrl(jobId);
  dlGlb.href = glbDownloadUrl(jobId);

  sectionPreviews.hidden = false;
  previewDepth.src = viewerAssetUrl(jobId, "depth_map_preview.png");
  previewHeight.src = viewerAssetUrl(jobId, "height_map_preview.png");
  previewColormap.src = viewerAssetUrl(jobId, "height_map_colormap.png");

  viewportHint.classList.add("hidden");
  hud.classList.remove("hidden");
  controlsHelp.classList.remove("hidden");
  viewportToolbar.classList.remove("hidden");

  const isBuildings = result.reconstruction_mode === "buildings";
  btnWireframe.classList.toggle("hidden", isBuildings);
  btnColormap.classList.toggle("hidden", isBuildings);
  if (isBuildings) {
    logProgress(`Loading ${result.n_buildings_detected ?? 0} building blocks…`);
    await loadAndShowTerrain({ mode: "buildings", glbUrl: glbDownloadUrl(jobId) });
  } else {
    await loadAndShowTerrain({ mode: "relief", bundleBaseUrl: bundleBaseUrlFor(jobId) });
  }
}

function bundleBaseUrlFor(jobId) {
  // viewerAssetUrl(jobId, 'meta.json') -> strip filename to get folder base
  const url = viewerAssetUrl(jobId, "meta.json");
  return url.slice(0, url.lastIndexOf("/"));
}

// ------------------------------------------------------------------
// Validation panel
// ------------------------------------------------------------------
btnValidate.addEventListener("click", async () => {
  const ref = fileValidateRef.files?.[0];
  if (!ref || !currentJobId) return;
  validationResults.innerHTML = `<div class="hint">Running validation…</div>`;
  try {
    const report = await validateJob(currentJobId, ref);
    renderValidationReport(report);
  } catch (err) {
    validationResults.innerHTML = `<div class="hint" style="color:#ff6b6b">${err.message}</div>`;
  }
});

function renderValidationReport(report) {
  const o = report.overall;
  let html = `
    <div><b>RMSE:</b> ${o.rmse.toFixed(2)} m &nbsp; <b>MAE:</b> ${o.mae.toFixed(2)} m &nbsp; <b>Corr:</b> ${o.correlation.toFixed(3)}</div>
    <div class="hint">${o.n_pixels.toLocaleString()} overlapping valid pixels</div>
  `;
  const classes = Object.keys(report.per_class || {});
  if (classes.length) {
    html += `<table><thead><tr><th>Class</th><th>RMSE</th><th>MAE</th><th>Corr</th><th>N</th></tr></thead><tbody>`;
    for (const c of classes) {
      const m = report.per_class[c];
      html += `<tr><td>${c}</td><td>${m.rmse.toFixed(2)}</td><td>${m.mae.toFixed(2)}</td><td>${m.correlation.toFixed(3)}</td><td>${m.n_pixels.toLocaleString()}</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  validationResults.innerHTML = html;
}

// ------------------------------------------------------------------
// Three.js scene
// ------------------------------------------------------------------
const canvas = el("viewport");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060810);
scene.fog = new THREE.Fog(0x060810, 400, 3000);

const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 20000);
camera.position.set(0, 300, 400);

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x1a1408, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(300, 500, 200);
scene.add(sun);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.08;

const flyControls = new PointerLockControls(camera, renderer.domElement);
let mode = "orbit"; // 'orbit' | 'fly'
const moveState = { forward: false, back: false, left: false, right: false, up: false, down: false };
const flyVelocity = new THREE.Vector3();
const FLY_SPEED = 180; // m/s at 1x, scaled by terrain size below

let terrainInfo = null; // { mesh, sampleElevation, meta, worldSize }
let currentMaterialMode = "texture"; // 'texture' | 'wireframe' | 'colormap'

async function loadAndShowTerrain(source) {
  if (terrainInfo) {
    scene.remove(terrainInfo.mesh);
    disposeObject3D(terrainInfo.mesh);
  }

  const info = source.mode === "buildings"
    ? await loadBuildingMesh(source.glbUrl, { onProgress: logProgress })
    : await loadTerrain(source.bundleBaseUrl, { onProgress: logProgress });

  terrainInfo = info;
  scene.add(info.mesh);

  // frame the camera around the new terrain
  const { w, h } = info.worldSize;
  const dist = Math.max(w, h, 10) * 0.9;
  camera.position.set(w * 0.15, dist * 0.55, dist * 0.75);
  orbitControls.target.set(0, info.elevRange * 0.2, 0);
  orbitControls.update();

  scene.fog.near = dist * 0.5;
  scene.fog.far = dist * 4;
  logProgress(info.isBuildingMesh
    ? "Building blocks ready. Explore in Orbit or Flythrough mode."
    : "Terrain ready. Explore in Orbit or Flythrough mode.");
}

function disposeObject3D(obj) {
  obj.traverse?.((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      mats.forEach((m) => m.dispose());
    }
  });
}

// ---- mode switching ----
btnModeOrbit.addEventListener("click", () => setMode("orbit"));
btnModeFly.addEventListener("click", () => setMode("fly"));
window.addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); setMode(mode === "orbit" ? "fly" : "orbit"); } });

function setMode(next) {
  mode = next;
  hudMode.textContent = mode === "orbit" ? "Orbit" : "Flythrough";
  btnModeOrbit.classList.toggle("active", mode === "orbit");
  btnModeFly.classList.toggle("active", mode === "fly");
  if (mode === "fly") {
    orbitControls.enabled = false;
    renderer.domElement.requestPointerLock?.();
  } else {
    orbitControls.enabled = true;
    document.exitPointerLock?.();
  }
}

flyControls.addEventListener("unlock", () => {
  if (mode === "fly") setMode("orbit");
});

window.addEventListener("keydown", (e) => {
  switch (e.code) {
    case "KeyW": case "ArrowUp": moveState.forward = true; break;
    case "KeyS": case "ArrowDown": moveState.back = true; break;
    case "KeyA": case "ArrowLeft": moveState.left = true; break;
    case "KeyD": case "ArrowRight": moveState.right = true; break;
    case "Space": moveState.up = true; break;
    case "ShiftLeft": case "ShiftRight": moveState.down = true; break;
  }
});
window.addEventListener("keyup", (e) => {
  switch (e.code) {
    case "KeyW": case "ArrowUp": moveState.forward = false; break;
    case "KeyS": case "ArrowDown": moveState.back = false; break;
    case "KeyA": case "ArrowLeft": moveState.left = false; break;
    case "KeyD": case "ArrowRight": moveState.right = false; break;
    case "Space": moveState.up = false; break;
    case "ShiftLeft": case "ShiftRight": moveState.down = false; break;
  }
});

// ---- material toggles ----
btnWireframe.addEventListener("click", () => {
  if (!terrainInfo) return;
  currentMaterialMode = currentMaterialMode === "wireframe" ? "texture" : "wireframe";
  applyMaterialMode();
  btnWireframe.classList.toggle("active", currentMaterialMode === "wireframe");
});
btnColormap.addEventListener("click", () => {
  if (!terrainInfo) return;
  currentMaterialMode = currentMaterialMode === "colormap" ? "texture" : "colormap";
  applyMaterialMode();
  btnColormap.classList.toggle("active", currentMaterialMode === "colormap");
});
btnResetView.addEventListener("click", () => {
  if (!terrainInfo) return;
  const { w, h } = terrainInfo.worldSize;
  const dist = Math.max(w, h) * 0.9;
  camera.position.set(w * 0.15, dist * 0.55, dist * 0.75);
  orbitControls.target.set(0, terrainInfo.elevRange * 0.2, 0);
  orbitControls.update();
});

function applyMaterialMode() {
  const mesh = terrainInfo.mesh;
  if (!mesh.material) return; // building-block mode: mesh is a Group, these toggles don't apply
  mesh.material.wireframe = currentMaterialMode === "wireframe";
  if (currentMaterialMode === "colormap") {
    if (!mesh.userData.colormapMaterial) {
      mesh.userData.colormapMaterial = new THREE.ShaderMaterial({
        uniforms: { minE: { value: 0 }, maxE: { value: terrainInfo.elevRange } },
        vertexShader: `
          varying float vElevation;
          void main() { vElevation = position.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
        `,
        fragmentShader: `
          varying float vElevation;
          uniform float minE; uniform float maxE;
          vec3 ramp(float t) {
            vec3 c0 = vec3(0.10,0.25,0.55), c1 = vec3(0.15,0.55,0.30), c2 = vec3(0.85,0.80,0.25), c3 = vec3(0.55,0.35,0.20), c4 = vec3(0.95,0.95,0.95);
            if (t < 0.25) return mix(c0,c1,t/0.25);
            if (t < 0.5) return mix(c1,c2,(t-0.25)/0.25);
            if (t < 0.75) return mix(c2,c3,(t-0.5)/0.25);
            return mix(c3,c4,(t-0.75)/0.25);
          }
          void main() { float t = clamp((vElevation - minE)/max(maxE-minE,0.0001),0.0,1.0); gl_FragColor = vec4(ramp(t), 1.0); }
        `,
      });
    }
    mesh.userData.originalMaterial = mesh.userData.originalMaterial || mesh.material;
    mesh.material = mesh.userData.colormapMaterial;
  } else if (mesh.userData.originalMaterial) {
    mesh.material = mesh.userData.originalMaterial;
    mesh.material.wireframe = currentMaterialMode === "wireframe";
  }
}

// ---- HUD raycast for cursor elevation ----
const raycaster = new THREE.Raycaster();
const mouseNdc = new THREE.Vector2();
renderer.domElement.addEventListener("mousemove", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouseNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
});

function updateHud() {
  if (!terrainInfo) return;
  hudCamH.textContent = `${camera.position.y.toFixed(1)} m`;
  if (mode === "orbit") {
    raycaster.setFromCamera(mouseNdc, camera);
    const hits = raycaster.intersectObject(terrainInfo.mesh, true);
    hudElev.textContent = hits.length ? `${hits[0].point.y.toFixed(1)} m` : "—";
  } else {
    // in flythrough, report elevation directly beneath the camera
    const down = new THREE.Vector3(0, -1, 0);
    raycaster.set(camera.position, down);
    const hits = raycaster.intersectObject(terrainInfo.mesh, true);
    hudElev.textContent = hits.length ? `${hits[0].point.y.toFixed(1)} m (ground below)` : "—";
  }
}

// ------------------------------------------------------------------
// Render loop
// ------------------------------------------------------------------
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);

  if (mode === "orbit") {
    orbitControls.update();
  } else if (mode === "fly" && flyControls.isLocked) {
    const speed = FLY_SPEED * (terrainInfo ? Math.max(terrainInfo.worldSize.w, terrainInfo.worldSize.h) / 512 : 1);
    flyVelocity.set(0, 0, 0);
    if (moveState.forward) flyVelocity.z -= 1;
    if (moveState.back) flyVelocity.z += 1;
    if (moveState.left) flyVelocity.x -= 1;
    if (moveState.right) flyVelocity.x += 1;
    if (flyVelocity.lengthSq() > 0) flyVelocity.normalize();
    flyControls.moveRight(flyVelocity.x * speed * dt);
    flyControls.moveForward(-flyVelocity.z * speed * dt);
    if (moveState.up) camera.position.y += speed * dt;
    if (moveState.down) camera.position.y -= speed * dt;
  }

  updateHud();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
});
