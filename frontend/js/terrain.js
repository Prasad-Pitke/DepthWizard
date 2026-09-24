// terrain.js — builds a displaced, textured terrain mesh from the
// DepthWizard viewer bundle (texture.jpg + heightmap.png + meta.json)
// and exposes elevation lookups for the HUD / validation UI.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/**
 * Loads a building-block GLB (produced by reconstruction_mode="buildings")
 * directly, since that geometry is discrete extruded boxes, not a
 * displaced heightfield — there's no heightmap to sample. Returns the
 * same shape as loadTerrain() so main.js can treat both modes uniformly;
 * sampleElevation() here does a downward raycast against the loaded
 * mesh instead of a heightmap lookup.
 */
export async function loadBuildingMesh(glbUrl, { onProgress } = {}) {
  const loader = new GLTFLoader();
  onProgress?.("Fetching building mesh");
  const gltf = await loader.loadAsync(glbUrl);
  onProgress?.("Decoded building mesh");

  const group = gltf.scene;
  group.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = false;
      child.receiveShadow = true;
    }
  });

  const box = new THREE.Box3().setFromObject(group);
  const size = new THREE.Vector3();
  box.getSize(size);
  const elevRange = Math.max(size.y, 1);

  const raycaster = new THREE.Raycaster();
  function sampleElevation(worldX, worldZ) {
    raycaster.set(new THREE.Vector3(worldX, elevRange + 50, worldZ), new THREE.Vector3(0, -1, 0));
    const hits = raycaster.intersectObject(group, true);
    return hits.length ? hits[0].point.y : 0;
  }

  return {
    mesh: group,
    sampleElevation,
    meta: { elevation_min_m: 0, elevation_max_m: elevRange, vertical_exaggeration: 1 },
    worldSize: { w: size.x, h: size.z },
    elevRange,
    isBuildingMesh: true,
  };
}

/**
 * Loads heightmap + texture and returns { mesh, sampleElevation, meta }
 * sampleElevation(u, v) -> elevation in meters, u/v in [0,1] (image space).
 */
export async function loadTerrain(baseUrl, { onProgress } = {}) {
  const meta = await fetch(`${baseUrl}/meta.json`).then((r) => r.json());
  onProgress?.("Fetched terrain metadata");

  const heightImg = await loadImage(`${baseUrl}/heightmap.png`);
  onProgress?.("Decoded heightmap");
  const heightData = imageToGrayscaleFloat(heightImg); // Float32Array, [0,1], row-major

  const segX = Math.min(meta.mesh_resolution, 512);
  const segY = Math.min(meta.mesh_resolution, 512);

  const geometry = new THREE.PlaneGeometry(
    meta.width_px * meta.horizontal_scale_m,
    meta.height_px * meta.horizontal_scale_m,
    segX - 1, segY - 1
  );
  geometry.rotateX(-Math.PI / 2); // lay flat, Y = up

  const pos = geometry.attributes.position;
  const elevRange = meta.elevation_max_m - meta.elevation_min_m || 1;
  const sampler = makeBilinearSampler(heightData, heightImg.width, heightImg.height);

  for (let i = 0; i < pos.count; i++) {
    const gx = i % segX, gy = Math.floor(i / segX);
    const u = gx / (segX - 1), v = gy / (segY - 1);
    const h01 = sampler(u, v);
    const elevation = meta.elevation_min_m + h01 * elevRange;
    pos.setY(i, elevation * meta.vertical_exaggeration);
  }
  geometry.computeVertexNormals();

  const textureLoader = new THREE.TextureLoader();
  const texture = await textureLoader.loadAsync(`${baseUrl}/texture.jpg`);
  texture.colorSpace = THREE.SRGBColorSpace;
  onProgress?.("Draped optical texture");

  const material = new THREE.MeshStandardMaterial({
    map: texture, roughness: 0.95, metalness: 0.0, flatShading: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = false;
  mesh.receiveShadow = true;

  const worldW = meta.width_px * meta.horizontal_scale_m;
  const worldH = meta.height_px * meta.horizontal_scale_m;

  function sampleElevation(u, v) {
    const h01 = sampler(u, v);
    return meta.elevation_min_m + h01 * elevRange;
  }

  return { mesh, sampleElevation, meta, worldSize: { w: worldW, h: worldH }, elevRange };
}

export function buildElevationColormapMaterial(baseMaterial, meta) {
  // Simple height-ramp shader material swap (hypsometric tint), used
  // when the user toggles "Elevation Colormap" in the toolbar.
  const vertexShader = `
    varying float vElevation;
    varying vec2 vUv;
    void main() {
      vElevation = position.y;
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  const fragmentShader = `
    varying float vElevation;
    varying vec2 vUv;
    uniform float minE;
    uniform float maxE;
    vec3 ramp(float t) {
      // blue -> green -> yellow -> brown -> white
      vec3 c0 = vec3(0.10, 0.25, 0.55);
      vec3 c1 = vec3(0.15, 0.55, 0.30);
      vec3 c2 = vec3(0.85, 0.80, 0.25);
      vec3 c3 = vec3(0.55, 0.35, 0.20);
      vec3 c4 = vec3(0.95, 0.95, 0.95);
      if (t < 0.25) return mix(c0, c1, t / 0.25);
      if (t < 0.5)  return mix(c1, c2, (t - 0.25) / 0.25);
      if (t < 0.75) return mix(c2, c3, (t - 0.5) / 0.25);
      return mix(c3, c4, (t - 0.75) / 0.25);
    }
    void main() {
      float t = clamp((vElevation - minE) / max(maxE - minE, 0.0001), 0.0, 1.0);
      gl_FragColor = vec4(ramp(t), 1.0);
    }
  `;
  const THREE_MOD = baseMaterial.constructor === undefined ? null : null;
  return { vertexShader, fragmentShader };
}

// ---------------- helpers ----------------
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function imageToGrayscaleFloat(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, img.width, img.height);
  const out = new Float32Array(img.width * img.height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = data[i] / 255; // R channel; heightmap is grayscale so R=G=B
  }
  return out;
}

function makeBilinearSampler(data, w, h) {
  return (u, v) => {
    const x = Math.min(Math.max(u, 0), 1) * (w - 1);
    const y = Math.min(Math.max(v, 0), 1) * (h - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
    const fx = x - x0, fy = y - y0;
    const v00 = data[y0 * w + x0], v10 = data[y0 * w + x1];
    const v01 = data[y1 * w + x0], v11 = data[y1 * w + x1];
    const top = v00 * (1 - fx) + v10 * fx;
    const bot = v01 * (1 - fx) + v11 * fx;
    return top * (1 - fy) + bot * fy;
  };
}
