/**
 * F1 Predict — Landing Page app.js
 * Drives the 3D scroll-controlled F1 showcase using Three.js.
 * Smooth camera splines, viewport tracking, and mouse cursor parallax.
 */

'use strict';

// ════════════════════════════════════════════════════════════════
// THREE.JS BOILERPLATE
// ════════════════════════════════════════════════════════════════
const canvas = document.getElementById('webgl');
const scene = new THREE.Scene();

// Camera Setup
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
scene.add(camera);

// Renderer Setup
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
  antialias: true,
  alpha: true,
  powerPreference: "high-performance"
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// ════════════════════════════════════════════════════════════════
// LIGHTING
// ════════════════════════════════════════════════════════════════
const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1.8);
directionalLight.position.set(5, 12, 8);
directionalLight.castShadow = true;
directionalLight.shadow.mapSize.width = 2048;
directionalLight.shadow.mapSize.height = 2048;
directionalLight.shadow.camera.near = 0.5;
directionalLight.shadow.camera.far = 25;
directionalLight.shadow.camera.left = -6;
directionalLight.shadow.camera.right = 6;
directionalLight.shadow.camera.top = 6;
directionalLight.shadow.camera.bottom = -6;
directionalLight.shadow.bias = -0.0005;
scene.add(directionalLight);

const spotlight = new THREE.SpotLight(0xffffff, 2.5);
spotlight.position.set(-5, 8, -5);
spotlight.angle = Math.PI / 4;
spotlight.penumbra = 0.8;
spotlight.castShadow = true;
scene.add(spotlight);

// Helper to generate soft radial gradient textures for volumetric clouds
function createSoftCircleTexture(colorStr) {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, colorStr);
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

// Coloured clouds grouping
const clouds = new THREE.Group();
scene.add(clouds);

const frontTexture = createSoftCircleTexture('rgba(255, 70, 0, 0.25)'); // Warm red/orange for front wing
const rearTexture = createSoftCircleTexture('rgba(0, 150, 255, 0.25)');  // Cool neon blue for diffuser

const frontSprites = [];
const rearSprites = [];

// Front support cloud (near front wing: Z around 2.2, Y around -0.6)
for (let i = 0; i < 16; i++) {
  const mat = new THREE.SpriteMaterial({
    map: frontTexture,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const sprite = new THREE.Sprite(mat);
  sprite.position.set(
    (Math.random() - 0.5) * 1.5,
    -0.75 + (Math.random() - 0.5) * 0.4,
    2.2 + (Math.random() - 0.5) * 1.2
  );
  const scale = 1.0 + Math.random() * 1.2;
  sprite.scale.set(scale, scale, 1);
  clouds.add(sprite);
  frontSprites.push({
    sprite,
    basePos: sprite.position.clone(),
    baseScale: scale,
    seed: Math.random() * 100
  });
}

// Rear support cloud (near rear diffuser: Z around -2.2, Y around -0.6)
for (let i = 0; i < 16; i++) {
  const mat = new THREE.SpriteMaterial({
    map: rearTexture,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const sprite = new THREE.Sprite(mat);
  sprite.position.set(
    (Math.random() - 0.5) * 1.5,
    -0.75 + (Math.random() - 0.5) * 0.4,
    -2.2 + (Math.random() - 0.5) * 1.2
  );
  const scale = 1.0 + Math.random() * 1.2;
  sprite.scale.set(scale, scale, 1);
  clouds.add(sprite);
  rearSprites.push({
    sprite,
    basePos: sprite.position.clone(),
    baseScale: scale,
    seed: Math.random() * 100
  });
}

// ════════════════════════════════════════════════════════════════
// MODEL LOADING
// ════════════════════════════════════════════════════════════════
let carModel = null;
const loader = new THREE.GLTFLoader();

loader.load(
  'f1_car.glb',
  (gltf) => {
    carModel = gltf.scene;
    
    // Auto-center and auto-scale model to fit a standard bounding box
    const box = new THREE.Box3().setFromObject(carModel);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    
    // Scale car to a standard longitudinal length of ~6.0 units
    const scaleFactor = 6.0 / maxDim;
    carModel.scale.set(scaleFactor, scaleFactor, scaleFactor);
    
    // Center it relative to the scene ground height
    const center = box.getCenter(new THREE.Vector3());
    carModel.position.x = -center.x * scaleFactor;
    carModel.position.y = -center.y * scaleFactor - 0.4; // suspended in air
    carModel.position.z = -center.z * scaleFactor;
    
    // Configure shadows and material quality
    carModel.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        
        // Enhance metalness/roughness for carbon composite reflection
        if (child.material) {
          child.material.envMapIntensity = 1.5;
          child.material.needsUpdate = true;
        }
      }
    });
    
    scene.add(carModel);
    
    // Fade out loading screen
    const loaderEl = document.getElementById('loadingOverlay');
    loaderEl.style.opacity = '0';
    setTimeout(() => loaderEl.classList.add('hidden'), 500);
  },
  (xhr) => {
    // Update loading percent progress
    if (xhr.total > 0) {
      const pct = Math.round((xhr.loaded / xhr.total) * 100);
      document.getElementById('loadingPercent').textContent = `Loading F1 Model (${pct}%)`;
    }
  },
  (error) => {
    console.error('An error happened while loading GLB model:', error);
    document.getElementById('loadingPercent').textContent = 'Error loading 3D Model';
  }
);

// ════════════════════════════════════════════════════════════════
// SCROLL-DRIVEN PATH TIMELINE
// ════════════════════════════════════════════════════════════════
// Keyframe coordinates: x, y, z positions and target focus spots
const keyframes = [
  { scroll: 0.00, pos: { x: 5.2, y: 1.25, z: 5.2 }, target: { x: 0, y: 0.25, z: 0 } },         // Slide 1: Side front overview
  { scroll: 0.25, pos: { x: -3.8, y: 0.35, z: 3.8 }, target: { x: -2.0, y: -0.05, z: 0.4 } },  // Slide 2: Close-up front wing & tires
  { scroll: 0.50, pos: { x: 3.8, y: 0.85, z: -3.2 }, target: { x: 0.8, y: 0.05, z: -1.0 } },    // Slide 3: Rear sidepod aerodynamic sweep
  { scroll: 0.75, pos: { x: 0.0, y: 5.25, z: 0.4 }, target: { x: 0, y: -0.15, z: -0.4 } },       // Slide 4: Cockpit & halo overview
  { scroll: 1.00, pos: { x: 0.0, y: 0.55, z: -6.2 }, target: { x: 0, y: 0.15, z: 2.0 } }        // Slide 5: Diffuser/exhaust tail view
];

let scrollRatio = 0;
let currentLookAt = new THREE.Vector3(0, 0, 0);

// Interpolates between keyframes to find position & target look-at vector
function getInterpolatedState(ratio) {
  // Clamp boundaries
  if (ratio <= 0) return { pos: keyframes[0].pos, target: keyframes[0].target };
  if (ratio >= 1) return { pos: keyframes[keyframes.length - 1].pos, target: keyframes[keyframes.length - 1].target };
  
  // Find keyframe interval
  let i = 0;
  for (i = 0; i < keyframes.length - 1; i++) {
    if (ratio >= keyframes[i].scroll && ratio <= keyframes[i + 1].scroll) {
      break;
    }
  }
  
  const kf1 = keyframes[i];
  const kf2 = keyframes[i + 1];
  
  // Calculate relative interpolation factor t (0 to 1)
  const t = (ratio - kf1.scroll) / (kf2.scroll - kf1.scroll);
  
  return {
    pos: {
      x: kf1.pos.x + (kf2.pos.x - kf1.pos.x) * t,
      y: kf1.pos.y + (kf2.pos.y - kf1.pos.y) * t,
      z: kf1.pos.z + (kf2.pos.z - kf1.pos.z) * t
    },
    target: {
      x: kf1.target.x + (kf2.target.x - kf1.target.x) * t,
      y: kf1.target.y + (kf2.target.y - kf1.target.y) * t,
      z: kf1.target.z + (kf2.target.z - kf1.target.z) * t
    }
  };
}

// Track viewport scroll ratios
window.addEventListener('scroll', () => {
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  scrollRatio = maxScroll > 0 ? window.scrollY / maxScroll : 0;
});

// ════════════════════════════════════════════════════════════════
// MOUSE CURSOR PARALLAX DRIFT
// ════════════════════════════════════════════════════════════════
let mouseX = 0;
let mouseY = 0;

window.addEventListener('mousemove', (e) => {
  // Normalize client coordinates to -1 to +1 range
  mouseX = (e.clientX - window.innerWidth / 2) / (window.innerWidth / 2);
  mouseY = (e.clientY - window.innerHeight / 2) / (window.innerHeight / 2);
});

// ════════════════════════════════════════════════════════════════
// RENDER LOOP
// ════════════════════════════════════════════════════════════════
const clock = new THREE.Clock();

function tick() {
  const elapsedTime = clock.getElapsedTime();
  
  // 1. Get base camera positions from scroll interpolation
  const state = getInterpolatedState(scrollRatio);
  
  // 2. Add smooth mouse parallax drift (scale offset based on distance)
  const parallaxX = mouseX * 0.9;
  const parallaxY = -mouseY * 0.6; // Inverted Y coordinates
  
  const targetX = state.pos.x + parallaxX;
  const targetY = state.pos.y + parallaxY;
  const targetZ = state.pos.z;
  
  // 3. Linearly interpolate (lerp) camera position for smooth damping
  camera.position.x += (targetX - camera.position.x) * 0.06;
  camera.position.y += (targetY - camera.position.y) * 0.06;
  camera.position.z += (targetZ - camera.position.z) * 0.06;
  
  // 4. Lerp camera focus vector (lookAt target)
  currentLookAt.x += (state.target.x - currentLookAt.x) * 0.06;
  currentLookAt.y += (state.target.y - currentLookAt.y) * 0.06;
  currentLookAt.z += (state.target.z - currentLookAt.z) * 0.06;
  
  camera.lookAt(currentLookAt);
  
  // 5. Apply static floating wave and cloud pulsations when model is loaded
  if (carModel) {
    // Floating suspension animation
    carModel.position.y = (-0.4) + Math.sin(elapsedTime * 1.0) * 0.08;
    
    // Animate support clouds
    const t = elapsedTime;
    frontSprites.forEach(item => {
      // Gentle floating drift
      item.sprite.position.y = item.basePos.y + Math.sin(t * 1.2 + item.seed) * 0.06 + Math.sin(t * 1.0) * 0.08;
      item.sprite.position.x = item.basePos.x + Math.cos(t * 0.6 + item.seed) * 0.04;
      // Pulse scale
      const scaleVal = item.baseScale * (1.0 + Math.sin(t * 1.5 + item.seed) * 0.12);
      item.sprite.scale.set(scaleVal, scaleVal, 1);
    });
    
    rearSprites.forEach(item => {
      // Gentle floating drift
      item.sprite.position.y = item.basePos.y + Math.sin(t * 1.2 + item.seed) * 0.06 + Math.sin(t * 1.0) * 0.08;
      item.sprite.position.x = item.basePos.x + Math.cos(t * 0.6 + item.seed) * 0.04;
      // Pulse scale
      const scaleVal = item.baseScale * (1.0 + Math.sin(t * 1.5 + item.seed) * 0.12);
      item.sprite.scale.set(scaleVal, scaleVal, 1);
    });
  }
  
  // Render Scene
  renderer.render(scene, camera);
  
  window.requestAnimationFrame(tick);
}

// Start Render loop
tick();

// ════════════════════════════════════════════════════════════════
// RESPONSIVE RESIZE HANDLER
// ════════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {
  // Update camera properties
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  
  // Update renderer
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// ════════════════════════════════════════════════════════════════
// SCROLL REVEAL OVERLAYS (INTERSECTION OBSERVER)
// ════════════════════════════════════════════════════════════════
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('revealed');
    } else {
      entry.target.classList.remove('revealed');
    }
  });
}, {
  threshold: 0.15,
  rootMargin: "-8% 0px -8% 0px" // triggers slightly inside the screen
});

// Observe all overlay elements
document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// Load dynamic stats from race_results.json
async function loadDynamicStats() {
  try {
    const resp = await fetch('race_results.json?t=' + Date.now());
    if (!resp.ok) return;
    const data = await resp.json();
    const agg = data.aggregate;
    if (agg) {
      const winnersCalledEl = document.getElementById('landing-winners-called');
      const averageMaeEl = document.getElementById('landing-average-mae');
      if (winnersCalledEl && typeof agg.winners_correct === 'number' && typeof agg.races_with_actuals === 'number') {
        winnersCalledEl.textContent = `${agg.winners_correct} / ${agg.races_with_actuals}`;
      }
      if (averageMaeEl && typeof agg.avg_mae === 'number') {
        averageMaeEl.textContent = agg.avg_mae.toFixed(2);
      }
    }
  } catch (e) {
    console.warn('Could not load dynamic stats:', e);
  }
}

// Call on page load
document.addEventListener('DOMContentLoaded', loadDynamicStats);
loadDynamicStats();
