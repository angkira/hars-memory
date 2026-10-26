// Three.js (WebGL2) starfield renderer for Memory Travel — the PRIMARY
// cosmos engine. WebGPU/GraphGPU is gone (Chrome/Dawn refuses this machine's
// RADV Vulkan driver, and WebGPU is flaky across Linux boxes generally);
// WebGL2 works everywhere we tested, so THREE.Points + a custom glow shader
// is the right vehicle for 47k nodes.
//
// Rendering model (one draw call for all stars, one for all edges):
// - BufferGeometry with per-node position / color / size / alpha / twinkle
//   attributes; ShaderMaterial draws a round soft-glow sprite (radial
//   gradient in the fragment shader — never a square point), size-
//   attenuated by camera distance, AdditiveBlending so dense clusters glow
//   where they overlap. Depth cue comes from distance attenuation + a
//   subtle exponential fog term in the shader.
// - Edges are a single LineSegments cloud with per-vertex colour and a
//   per-edge `aRing` attribute: journey stages reveal BFS rings purely by
//   tweening one uniform (uRing) — no per-frame buffer uploads.
// - OrbitControls (damping) gives full 3D rotate/zoom/pan — the "вращать"
//   requirement. Journey mode temporarily disables them and drives the
//   camera itself (cameraFly / orbitAround).
// - Picking is precision screen-space: all node positions are projected
//   once per camera move (cached) and picks resolve to the nearest star
//   within a pixel radius. Equivalent to raycasting Points with a tight
//   threshold but zoom-independent and exact.
//
// The class implements the shared CosmosLike surface (lib/cosmos.ts) plus
// the Journey3D extension used by the Travel view (feature-detected).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  INK,
  WORLD_SCALE,
  type CosmosLike,
  type RGBA,
  type SelectedNode,
  type XYZ,
} from "./cosmos";
import { applyEase, solveArrival, type CamEase } from "./camPose";
import { colorForType, familyOf, hexToRGBA } from "./palette";
import type { CosmosGraph } from "./types";

export type { SelectedNode } from "./cosmos";

const BG = 0x03040c;
/** Spotlight dim factor: non-matching stars drop to ~20% brightness. */
const SPOT_DIM = 0.18;

// ---- zoom-adaptive edge opacity -------------------------------------------
// The single edge cloud used to run at a fixed 0.028 — a pleasant haze from
// the overview, but effectively invisible once the camera dives to a node
// (the cosmos view never called setEdgeOpacity, so it was stuck there).
// The base opacity now follows the camera distance `d` to the graph:
//
//   d >= EDGE_FAR_FRAC * fitDist  -> EDGE_OPACITY_FAR  (0.028, unchanged look)
//   d <= EDGE_NEAR_DIST           -> EDGE_OPACITY_NEAR (0.42, readable lines)
//   in between                    -> smoothstep in LOG distance (even ramp
//                                    across the ~2.5 orders of magnitude, no
//                                    threshold pops; monotone by construction)
//
// Values are for NON-focused edges. A selection multiplies them by the
// per-edge focus weight below, so incident edges land at ~0.42*1.15 ≈ 0.48
// (the acceptance value) and unrelated edges keep 0.30 of the base
// (>= 0.12 close-up — still readable, just quiet).
const EDGE_OPACITY_FAR = 0.028;
const EDGE_OPACITY_NEAR = 0.42;
/** Camera→graph distance (world units) at which the close-up value is reached. */
const EDGE_NEAR_DIST = 12;
/** Far reference = this fraction of the fit distance (~190 at fit ≈ 350). */
const EDGE_FAR_FRAC = 0.55;
/** Unrelated edges keep this fraction of the base when a selection is active. */
const EDGE_SEL_DIM = 0.3;
/** Hover is milder than selection, but still clearly a highlight. */
const EDGE_HOV_DIM = 0.35;
/** Incident-edge boost (× base): the "lit up" look. */
const EDGE_SEL_BOOST = 1.15;
const EDGE_HOV_BOOST = 1.3;
/** 1-hop neighbour edge structural weight (incident edges are 1.0). */
const EDGE_HOP_W = 0.5;
/** Uniform ease time constant (s): selection/hover/toggle fade, no popping. */
const EDGE_EASE_TAU = 0.1;

/**
 * Base edge opacity for a camera distance `d` from the graph (see the block
 * comment above). Pure + monotone — safe to call every frame.
 */
function edgeOpacityForDistance(d: number, fitDist: number): number {
  const far = Math.max(EDGE_NEAR_DIST * 3, fitDist * EDGE_FAR_FRAC);
  if (d >= far) return EDGE_OPACITY_FAR;
  if (d <= EDGE_NEAR_DIST) return EDGE_OPACITY_NEAR;
  const u = (Math.log(far) - Math.log(d)) / (Math.log(far) - Math.log(EDGE_NEAR_DIST));
  const s = u * u * (3 - 2 * u); // smoothstep
  return EDGE_OPACITY_FAR + (EDGE_OPACITY_NEAR - EDGE_OPACITY_FAR) * s;
}

export interface Journey3D {
  addMarker3(label: string, p: XYZ, size: number, color: RGBA): number;
  moveMarker3(id: number, p: XYZ, size?: number): void;
  cameraFly(pos: XYZ, target: XYZ, ms: number, ease?: "out" | "inout" | "linear"): void;
  cameraSet(pos: XYZ, target: XYZ): void;
  cameraNow(): { pos: XYZ; target: XYZ };
  /** Current perspective FOV (deg) and canvas aspect — for framing keyframes. */
  viewInfo(): { fov: number; aspect: number };
  /** Eased flight TO A NODE: target tweens onto the node, camera ends at a
   *  comfortable viewing distance along the current line of sight. */
  flyToNode(p: XYZ, dist?: number, ms?: number): void;
  /** Instant camera placement on the surface of a sphere around `center`. */
  orbitAround(center: XYZ, dist: number, azimuth: number, elevation: number): void;
  setControlsEnabled(on: boolean): void;
  /** Per-edge BFS depth; null clears (all edges at base opacity). */
  setEdgeRings(depths: Float32Array | null): void;
  /** Continuous reveal frontier — edges with aRing <= v are lit. */
  setEdgeRingProgress(v: number): void;
  setEdgeOpacity(v: number): void;
  setContextLines(from: XYZ, tos: XYZ[], color: RGBA): void;
  clearContextLines(): void;
  /** Glowing journey trail through the given world points. */
  setTrail(points: XYZ[], color: RGBA): void;
  clearTrail(): void;
  /** Distinct glowing query beacon / pulsing target sphere in 3D space (Chapter 3). */
  setQueryBeacon(pos: XYZ | null, color?: RGBA): void;
  clearQueryBeacon(): void;
  /** Animated graph walk relations with traveling photon beams (Chapter 5). */
  setGraphWalkEdges(edges: { from: XYZ; to: XYZ; color?: RGBA }[], options?: { intensity?: number }): void;
  clearGraphWalkEdges(): void;
  /** Bounding cluster hull / cage tightly enclosing context entities (Chapter 7). */
  setClusterHull(points: XYZ[], color?: RGBA): void;
  clearClusterHull(): void;
  /** Connected citation beams from answer hub to evidence nodes (Chapter 8). */
  setCitationBeams(items: { from: XYZ; to: XYZ; color?: RGBA }[]): void;
  clearCitationBeams(): void;
}

interface MarkerObj {
  id: number;
  label: string;
  obj: THREE.Points;
  mat: THREE.ShaderMaterial;
  size: number;
}

const POINT_VERT = /* glsl */ `
  attribute float aSize;
  attribute vec3 aColor;
  attribute float aAlpha;
  attribute float aTwinkle;
  attribute float aSel;   // 1 on the selected star: size + brightness boost
  uniform float uScale;   // px-per-world at unit distance (set on resize)
  uniform float uTime;    // seconds, for the gentle twinkle
  uniform float uFog;     // fog density (depth cue)
  uniform float uFogRef;  // reference distance (fit distance)
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = max(0.1, -mv.z);
    // size attenuation: farther stars are smaller…
    float px = aSize * (1.0 + aSel * 1.1) * uScale / dist;
    // …and slightly dimmer (exponential fog) — subtle depth cue
    float fog = exp(-uFog * (dist - uFogRef) / max(1.0, uFogRef));
    float tw = 0.92 + 0.08 * sin(uTime * (0.6 + aTwinkle * 1.7) + aTwinkle * 40.0);
    vColor = aColor + vec3(1.0) * aSel * 0.35; // emissive push toward white
    vAlpha = aAlpha * tw * clamp(fog, 0.55, 1.0) * (1.0 + aSel * 0.9);
    gl_PointSize = clamp(px, 1.0, 90.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const POINT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv) * 2.0;          // 0 at centre, 1 at edge
    if (d > 1.0) discard;                 // round sprite, never a square
    float core = smoothstep(0.42, 0.0, d);   // hot centre
    float halo = smoothstep(1.0, 0.3, d);    // soft radial falloff
    // Small per-star contribution: with 47k stars the GLOW comes from
    // additive overlap in dense clusters — vivid cores wash to white if
    // each sprite contributes too much.
    float a = (core * 0.28 + halo * 0.10) * vAlpha;
    gl_FragColor = vec4(vColor * (0.72 + core * 0.55), a);
  }
`;

const EDGE_VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aRing;
  attribute float aFocus;  // selection weight: 0 unrelated, .5 1-hop, 1 incident
  attribute float aHover;  // hover weight (same encoding)
  uniform float uRing; // reveal frontier (1e9 = everything visible)
  varying vec3 vColor;
  varying float vRingA;
  varying float vFocus;
  varying float vHover;
  void main() {
    vColor = aColor;
    vRingA = 1.0 - smoothstep(uRing - 0.6, uRing + 0.6, aRing);
    vFocus = aFocus;
    vHover = aHover;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const EDGE_FRAG = /* glsl */ `
  uniform float uOpacity; // zoom-adaptive base opacity
  uniform float uFocus;   // eased selection focus strength 0..1
  uniform float uHover;   // eased hover strength 0..1
  uniform float uSelDim;  // unrelated-edge factor while selected
  uniform float uHovDim;  // unrelated-edge factor while hovered
  uniform float uSelBoost; // incident-edge boost while selected
  uniform float uHovBoost; // incident-edge boost while hovered
  varying vec3 vColor;
  varying float vRingA;
  varying float vFocus;
  varying float vHover;
  void main() {
    // Selection weight; hover folded in multiplicatively so an inactive focus
    // is exactly 1.0 (no residue). Both are per-vertex attributes — the dim /
    // highlight is pure attribute math, still one draw call.
    float selW = mix(1.0, uSelDim + (uSelBoost - uSelDim) * vFocus, uFocus);
    float hovW = mix(1.0, uHovDim + (uHovBoost - uHovDim) * vHover, uHover);
    float w = min(selW * hovW, 1.35);
    // focused/hovered edges glow: push colour toward white a touch
    vec3 col = vColor * (1.0 + (vFocus * uFocus + vHover * uHover) * 0.6);
    gl_FragColor = vec4(col, uOpacity * vRingA * w);
  }
`;

const LINE_VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aAlpha;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = aColor;
    vAlpha = aAlpha;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const LINE_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

const MARKER_VERT = POINT_VERT;

/** Minimum on-screen radius (CSS px) of the selection ring — see animateRing. */
const RING_MIN_PX = 12;
const MARKER_FRAG = /* glsl */ `
  uniform float uPulse; // seconds — beacons breathe
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv) * 2.0;
    if (d > 1.0) discard;
    float core = smoothstep(0.34, 0.0, d);
    float halo = smoothstep(1.0, 0.05, d);
    // pulsing ring: markers breathe so they read as beacons, not stars
    float pulse = 0.82 + 0.18 * sin(uPulse * 3.4);
    float a = (core * 1.1 + halo * 0.5) * vAlpha * pulse;
    gl_FragColor = vec4(vColor * (0.8 + core * 0.7), a);
  }
`;

const PULSE_EDGE_VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aProgress; // 0 at seed / source, 1 at neighbor / target
  varying vec3 vColor;
  varying float vProgress;
  void main() {
    vColor = aColor;
    vProgress = aProgress;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const PULSE_EDGE_FRAG = /* glsl */ `
  uniform float uTime;
  uniform float uBaseAlpha;
  varying vec3 vColor;
  varying float vProgress;
  void main() {
    // Continuous animated traveling photon pulses along relations
    float phase = fract(vProgress * 2.2 - uTime * 2.0);
    float packet = smoothstep(0.0, 0.25, phase) * smoothstep(0.7, 0.25, phase);
    float a = clamp(uBaseAlpha + packet * 0.85, 0.0, 1.0);
    vec3 col = vColor * (0.88 + packet * 0.95);
    gl_FragColor = vec4(col, a);
  }
`;

export class Cosmos3D implements CosmosLike, Journey3D {
  private canvas: HTMLCanvasElement | null = null;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private ro: ResizeObserver | null = null;
  private dpr = 1;

  // graph data
  private n = 0;
  private geo: THREE.BufferGeometry | null = null;
  private points: THREE.Points | null = null;
  private ptsMat: THREE.ShaderMaterial | null = null;
  private pos: Float32Array = new Float32Array(0);
  private colAttr: THREE.BufferAttribute | null = null;
  private alphaAttr: THREE.BufferAttribute | null = null;
  private sizeAttr: THREE.BufferAttribute | null = null;
  private selAttr: THREE.BufferAttribute | null = null;
  private selIdx = -1;
  private selRing: THREE.Mesh | null = null;
  private baseSize: Float32Array = new Float32Array(0);
  private baseCol: Float32Array = new Float32Array(0);
  private types: string[] = [];
  private names: string[] = [];
  private hidden = new Set<string>();

  // edges
  private edgeGeo: THREE.BufferGeometry | null = null;
  private edges: THREE.LineSegments | null = null;
  private edgeMat: THREE.ShaderMaterial | null = null;
  private edgeCount = 0;
  // edge focus/hover + zoom-adaptive opacity (all attribute/uniform driven)
  private edgeS = new Int32Array(0);
  private edgeT = new Int32Array(0);
  private adj: number[][] = [];
  private focusAttr: THREE.BufferAttribute | null = null;
  private hoverAttr: THREE.BufferAttribute | null = null;
  private focusIdx = -1;
  private hoverIdx = -1;
  private edgeAdaptive = false;
  private edgeEnabled = true;
  private edgeFixed = 0.028;
  /** Eased base opacity written into uOpacity every frame (never jumps). */
  private edgeOpacity = 0.028;
  /** Eased focus/hover strengths (0..1) — see updateEdgeUniforms. */
  private focusStrength = 0;
  private hoverStrength = 0;
  private lastFrameT = 0;
  private lastHoverT = 0;

  // highlights
  private baseAlpha: Float32Array = new Float32Array(0);
  private spot: Set<number> | null = null;
  private paints = new Map<number, { color: RGBA; size?: number }>();
  private halos: { idx: number; color: RGBA; mesh: THREE.Mesh; size: number }[] = [];

  // markers + journey lines
  private markers = new Map<number, MarkerObj>();
  private nextMarker = 1;
  private ctxLines: THREE.LineSegments | null = null;
  private trail: THREE.Line | null = null;
  private pulseRaf = 0;
  private pulseT0 = 0;

  // Chapter 3: 3D Query Beacon
  private queryBeacon: {
    group: THREE.Group;
    core: THREE.Mesh;
    wireSphere: THREE.Mesh;
    ring1: THREE.Mesh;
    ring2: THREE.Mesh;
    beam: THREE.LineSegments;
  } | null = null;

  // Chapter 5: Animated Graph Walk Edges
  private graphWalkMesh: THREE.LineSegments | null = null;
  private graphWalkMat: THREE.ShaderMaterial | null = null;

  // Chapter 7: Context Assembly Cluster Hull
  private clusterHull: THREE.Group | null = null;

  // Chapter 8: Citation Beams
  private citationBeams: THREE.LineSegments | null = null;

  // camera flight
  private camRaf = 0;
  /** true while a tween owns the camera: the render loop must not also run
   *  OrbitControls.update() then (one writer per frame, always). */
  private camBusy = false;
  /** OrbitControls state to restore when the (possibly chained) flight ends. */
  private camPrev: { enabled: boolean; auto: boolean } | null = null;
  private fitDist = 150;

  // picking (screen-space projection cache)
  private screen = new Float32Array(0);
  private screenDirty = true;
  private hoverRaf = 0;
  private downAt: { x: number; y: number } | null = null;

  onSelect: ((s: SelectedNode | null) => void) | null = null;
  onHover: ((s: SelectedNode | null) => void) | null = null;

  get ready(): boolean {
    return this.renderer !== null;
  }

  get linksLoaded(): number {
    return this.edgeCount;
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) throw new Error("cosmos3d init failed at stage=canvas: no canvas element");
    // Probe WebGL2 FIRST so the caller can fall back to Canvas2D cleanly
    // (THREE.WebGLRenderer would otherwise throw deep inside its own code).
    const probe = document.createElement("canvas");
    const gl = probe.getContext("webgl2", { antialias: false });
    if (!gl) {
      throw new Error("cosmos3d init failed at stage=webgl2: WebGL2 unavailable in this browser");
    }
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
    } catch (e) {
      throw new Error(`cosmos3d init failed at stage=construct: ${e instanceof Error ? e.message : String(e)}`);
    }
    this.canvas = canvas;
    this.renderer = renderer;
    // SwiftShader-class software rasterizers choke on dpr≥2 at 47k additive
    // points; 1.5 keeps real-GPU sharpness while staying fast in software.
    this.dpr = Math.min(1.5, window.devicePixelRatio || 1);
    renderer.setPixelRatio(this.dpr);
    renderer.setClearColor(BG, 1);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 4000);
    this.camera.position.set(0, 90, this.fitDist);
    const controls = new OrbitControls(this.camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.autoRotate = true; // gentle cinematic drift in the cosmos view
    controls.autoRotateSpeed = 0.22;
    controls.minDistance = 2;
    controls.maxDistance = 900;
    controls.addEventListener("change", () => {
      this.screenDirty = true;
    });
    this.controls = controls;
    this.resize();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.startPulseClock();
    this.renderer.setAnimationLoop(() => {
      // hidden canvases (tab switches, presentation's inactive canvas) skip
      // both the controls update and the render — keeps software rasterizers
      // from burning frames on an invisible view
      if (this.canvas && this.canvas.offsetParent === null) return;
      const now = performance.now();
      const dt = this.lastFrameT ? Math.min(0.25, (now - this.lastFrameT) / 1000) : 0.016;
      this.lastFrameT = now;
      this.updateEdgeUniforms(dt);
      const t = (now - this.pulseT0) / 1000;
      this.animateRing(t);
      this.animateHalos(t);
      this.animateQueryBeacon(t);
      if (this.graphWalkMat) this.graphWalkMat.uniforms.uTime.value = t;
      this.animateClusterHull(t);
      // a camera tween calls controls.update() itself; two writers per frame
      // is how the arrival pose used to get nudged off the node
      if (!this.camBusy) this.controls?.update();
      this.renderer?.render(this.scene, this.camera!);
    });
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerleave", this.onLeave);
    canvas.addEventListener("click", this.onClick);
    canvas.addEventListener("pointerdown", this.onDown);
    canvas.style.touchAction = "none";
    canvas.style.cursor = "grab";
  }

  /** Load the graph at 3D positions. Same role as the old Cosmos.load. */
  async load(
    graph: CosmosGraph,
    posByIdx: (XYZ | null)[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<void> {
    if (!this.renderer) throw new Error("cosmos3d not initialised");
    const n = graph.nodes.length;
    this.n = n;
    this.names = graph.nodes.map((nd) => nd.id);
    this.types = graph.nodes.map((nd) => nd.t);
    this.pos = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const alphas = new Float32Array(n);
    const twinkle = new Float32Array(n);
    this.baseSize = new Float32Array(n);
    this.baseCol = new Float32Array(n * 3);
    this.baseAlpha = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const nd = graph.nodes[i];
      const p = posByIdx[i];
      if (p) {
        this.pos[i * 3] = p[0];
        this.pos[i * 3 + 1] = p[1];
        this.pos[i * 3 + 2] = p[2];
      } else {
        // no coords: scatter on a sparse shell so unpositioned nodes never
        // pile into a fake clump at the origin
        const r = WORLD_SCALE * (1.15 + Math.random() * 0.3);
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        this.pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
        this.pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
        this.pos[i * 3 + 2] = r * Math.cos(ph) * 0.6;
      }
      // colour IS the entity family (legend parity)
      const c = hexToRGBA(colorForType(nd.t));
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
      // hubs slightly larger: degree -> world size (tuned so regular stars
      // are ~2px at fit distance, hubs ~10px)
      const s = 0.16 + Math.min(0.7, Math.sqrt(nd.deg) * 0.075);
      sizes[i] = s;
      alphas[i] = 1;
      twinkle[i] = Math.random();
      this.baseSize[i] = s;
      this.baseCol[i * 3] = c[0];
      this.baseCol[i * 3 + 1] = c[1];
      this.baseCol[i * 3 + 2] = c[2];
      this.baseAlpha[i] = 1;
      if (onProgress && i % 8000 === 0) {
        onProgress(i, n);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    onProgress?.(n, n);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    this.colAttr = new THREE.BufferAttribute(colors, 3);
    this.sizeAttr = new THREE.BufferAttribute(sizes, 1);
    this.alphaAttr = new THREE.BufferAttribute(alphas, 1);
    geo.setAttribute("aColor", this.colAttr);
    geo.setAttribute("aSize", this.sizeAttr);
    geo.setAttribute("aAlpha", this.alphaAttr);
    geo.setAttribute("aTwinkle", new THREE.BufferAttribute(twinkle, 1));
    geo.setAttribute("aSel", this.selAttr = new THREE.BufferAttribute(new Float32Array(n), 1));
    this.geo = geo;
    this.ptsMat = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      uniforms: {
        uScale: { value: this.pointScale() },
        uTime: { value: 0 },
        uFog: { value: 0.4 },
        uFogRef: { value: this.fitDist },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geo, this.ptsMat);
    this.points.frustumCulled = false; // all 47k visible, skip the bounds check
    this.scene.add(this.points);

    // edges: one LineSegments cloud, per-vertex dim colour, ring attribute
    // for the journey BFS reveal.
    const edges = graph.edges.filter(([s, t]) => s >= 0 && t >= 0 && s < n && t < n);
    this.edgeCount = edges.length;
    if (edges.length) {
      const ep = new Float32Array(edges.length * 6);
      const ec = new Float32Array(edges.length * 6);
      const er = new Float32Array(edges.length * 2);
      // adjacency (node -> incident edge indices) for O(deg) focus rebuilds
      this.edgeS = new Int32Array(edges.length);
      this.edgeT = new Int32Array(edges.length);
      this.adj = Array.from({ length: n }, () => [] as number[]);
      for (let k = 0; k < edges.length; k++) {
        const [s, t] = edges[k];
        this.edgeS[k] = s;
        this.edgeT[k] = t;
        this.adj[s].push(k);
        if (t !== s) this.adj[t].push(k);
        for (let c = 0; c < 3; c++) {
          ep[k * 6 + c] = this.pos[s * 3 + c];
          ep[k * 6 + 3 + c] = this.pos[t * 3 + c];
          // dim desaturated blue, halfway to the node colours
          ec[k * 6 + c] = 0.24 + 0.35 * this.baseCol[s * 3 + c];
          ec[k * 6 + 3 + c] = 0.24 + 0.35 * this.baseCol[t * 3 + c];
        }
        er[k * 2] = 1e9;
        er[k * 2 + 1] = 1e9;
      }
      const eg = new THREE.BufferGeometry();
      eg.setAttribute("position", new THREE.BufferAttribute(ep, 3));
      eg.setAttribute("aColor", new THREE.BufferAttribute(ec, 3));
      eg.setAttribute("aRing", new THREE.BufferAttribute(er, 1));
      // per-edge focus strength (duplicated onto both vertices): 0 unrelated,
      // 0.5 = 1-hop neighbour, 1 = incident. Written on (de)selection only.
      this.focusAttr = new THREE.BufferAttribute(new Float32Array(edges.length * 2), 1);
      this.hoverAttr = new THREE.BufferAttribute(new Float32Array(edges.length * 2), 1);
      eg.setAttribute("aFocus", this.focusAttr);
      eg.setAttribute("aHover", this.hoverAttr);
      this.edgeGeo = eg;
      this.edgeMat = new THREE.ShaderMaterial({
        vertexShader: EDGE_VERT,
        fragmentShader: EDGE_FRAG,
        uniforms: {
          uOpacity: { value: this.edgeFixed },
          uRing: { value: 1e9 },
          uFocus: { value: 0 },
          uHover: { value: 0 },
          uSelDim: { value: EDGE_SEL_DIM },
          uHovDim: { value: EDGE_HOV_DIM },
          uSelBoost: { value: EDGE_SEL_BOOST },
          uHovBoost: { value: EDGE_HOV_BOOST },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      this.edgeOpacity = this.edgeEnabled ? this.edgeFixed : 0;
      this.focusIdx = -1;
      this.hoverIdx = -1;
      this.focusStrength = 0;
      this.hoverStrength = 0;
      this.edges = new THREE.LineSegments(eg, this.edgeMat);
      this.edges.frustumCulled = false;
      this.edges.visible = this.edgeEnabled;
      this.scene.add(this.edges);
    } else {
      this.adj = [];
      this.edgeS = new Int32Array(0);
      this.edgeT = new Int32Array(0);
    }

    this.paints.clear();
    this.spot = null;
    this.screenDirty = true;
    // debug/verification handle: expose both active and specific instances
    const win = window as unknown as { __cosmos3d?: Cosmos3D; __cosmosTrace?: Cosmos3D; __cosmosFull?: Cosmos3D };
    win.__cosmos3d = this;
    if (this.n < 500) win.__cosmosTrace = this;
    else win.__cosmosFull = this;
    this.fit(0.12);
  }

  clear(): void {
    if (this.points) {
      this.scene.remove(this.points);
      this.geo?.dispose();
      this.ptsMat?.dispose();
    }
    if (this.edges) {
      this.scene.remove(this.edges);
      this.edgeGeo?.dispose();
      this.edgeMat?.dispose();
    }
    this.geo = this.points = null;
    this.edges = this.edgeGeo = this.edgeMat = null;
    this.focusAttr = this.hoverAttr = null;
    this.adj = [];
    this.edgeS = new Int32Array(0);
    this.edgeT = new Int32Array(0);
    this.focusIdx = -1;
    this.hoverIdx = -1;
    this.ptsMat = null;
    this.selAttr = null;
    if (this.selRing) {
      this.scene.remove(this.selRing);
      this.selRing.geometry.dispose();
      (this.selRing.material as THREE.Material).dispose();
      this.selRing = null;
    }
    this.selIdx = -1;
    this.n = 0;
    this.pos = new Float32Array(0);
    this.screen = new Float32Array(0);
    this.types = [];
    this.names = [];
    this.edgeCount = 0;
    this.paints.clear();
    this.spot = null;
    this.clearMarkers();
    this.clearHalos();
    this.clearContextLines();
    this.clearTrail();
  }

  setHiddenTypes(types: Set<string>): void {
    this.hidden = types;
    this.recomputeAlphas();
  }

  addMarker(label: string, x: number, y: number, size: number, color: RGBA): number {
    return this.addMarker3(label, [x, y, 0], size, color);
  }

  addMarker3(label: string, p: XYZ, size: number, color: RGBA): number {
    if (!this.renderer) throw new Error("cosmos3d not initialised");
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(p), 3));
    g.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array([color[0], color[1], color[2]]), 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array([size]), 1));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(new Float32Array([1]), 1));
    g.setAttribute("aTwinkle", new THREE.BufferAttribute(new Float32Array([0]), 1));
    g.setAttribute("aSel", new THREE.BufferAttribute(new Float32Array([0]), 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: MARKER_VERT,
      fragmentShader: MARKER_FRAG,
      uniforms: {
        uScale: { value: this.pointScale() },
        uTime: { value: 0 },
        uFog: { value: 0 },
        uFogRef: { value: this.fitDist },
        uPulse: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const obj = new THREE.Points(g, mat);
    obj.frustumCulled = false;
    this.scene.add(obj);
    const id = this.nextMarker++;
    this.markers.set(id, { id, label, obj, mat, size });
    return id;
  }

  moveMarker(id: number, x: number, y: number, size?: number): void {
    this.moveMarker3(id, [x, y, 0], size);
  }

  moveMarker3(id: number, p: XYZ, size?: number): void {
    const m = this.markers.get(id);
    if (!m) return;
    const attr = m.obj.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.setXYZ(0, p[0], p[1], p[2]);
    attr.needsUpdate = true;
    if (size !== undefined) {
      const sa = m.obj.geometry.getAttribute("aSize") as THREE.BufferAttribute;
      sa.setX(0, size);
      sa.needsUpdate = true;
      m.size = size;
    }
  }

  removeMarker(id: number): void {
    const m = this.markers.get(id);
    if (!m) return;
    this.scene.remove(m.obj);
    m.obj.geometry.dispose();
    m.mat.dispose();
    this.markers.delete(id);
  }

  private clearMarkers(): void {
    for (const id of [...this.markers.keys()]) this.removeMarker(id);
  }

  paint(paints: Map<number, { color: RGBA; size?: number }> | null): void {
    if (!this.colAttr || !this.sizeAttr || !this.n) return;
    if (paints === null) this.paints.clear();
    else for (const [idx, p] of paints) this.paints.set(idx, p);
    this.recomputeAlphas();
    this.recomputeColorsSizes();
  }

  spotlight(keep: Set<number> | null): void {
    this.spot = keep;
    this.recomputeAlphas();
  }

  setEdgeOpacity(v: number): void {
    // fixed base (journey/presentation) also cancels zoom-adaptive mode
    this.edgeFixed = v;
    this.edgeAdaptive = false;
  }

  setEdgeZoomAdaptive(on: boolean): void {
    this.edgeAdaptive = on;
  }

  setEdgesEnabled(on: boolean): void {
    this.edgeEnabled = on;
    if (this.edges) this.edges.visible = on;
  }

  /**
   * Selection focus. The per-edge strength lives in the `aFocus` attribute,
   * so switching focus is one CPU pass over the edges (adjacency, ~0.2ms) and
   * one buffer upload — the draw stays a single LineSegments call. The uniform
   * uFocus is eased in updateEdgeUniforms, so the transition has no pop.
   */
  setEdgeFocus(idx: number | null): void {
    const v = idx ?? -1;
    if (v === this.focusIdx) return;
    this.focusIdx = v;
    this.rebuildFocusAttr();
  }

  setEdgeHover(idx: number | null): void {
    const v = idx ?? -1;
    if (v === this.hoverIdx) return;
    this.hoverIdx = v;
    this.rebuildHoverAttr();
  }

  edgeIntensity(): number {
    return this.edgeOpacity;
  }

  /** Fill one strength buffer: incident edges = 1, 1-hop neighbours = 0.5. */
  private fillFocusAttr(attr: THREE.BufferAttribute, f: number): void {
    const a = attr.array as Float32Array;
    a.fill(0);
    if (f >= 0 && f < this.n) {
      const inc = this.adj[f] ?? [];
      for (const k of inc) {
        a[k * 2] = 1;
        a[k * 2 + 1] = 1;
      }
      // 1-hop: every edge touching a neighbour of f (not f itself)
      for (const k of inc) {
        const other = this.edgeS[k] === f ? this.edgeT[k] : this.edgeS[k];
        if (other === f) continue;
        const nb = this.adj[other] ?? [];
        for (const k2 of nb) {
          if (a[k2 * 2] === 0) {
            a[k2 * 2] = EDGE_HOP_W;
            a[k2 * 2 + 1] = EDGE_HOP_W;
          }
        }
      }
    }
    attr.needsUpdate = true;
  }

  private rebuildFocusAttr(): void {
    if (this.focusAttr) this.fillFocusAttr(this.focusAttr, this.focusIdx);
  }

  private rebuildHoverAttr(): void {
    if (this.hoverAttr) this.fillFocusAttr(this.hoverAttr, this.hoverIdx);
  }

  /**
   * Per-frame edge uniforms (cheap: three uniform writes):
   *  - uOpacity: eased toward the zoom-adaptive curve (or the fixed value);
   *  - uFocus / uHover: eased toward 0/1 so selection & hover fade, no pop.
   */
  private updateEdgeUniforms(dt: number): void {
    if (!this.edgeMat) return;
    let target = this.edgeFixed;
    if (!this.edgeEnabled) {
      target = 0;
    } else if (this.edgeAdaptive && this.camera) {
      const tgt = this.controls?.target ?? new THREE.Vector3();
      target = edgeOpacityForDistance(this.camera.position.distanceTo(tgt), this.fitDist);
    }
    const k = dt > 0 ? 1 - Math.exp(-dt / EDGE_EASE_TAU) : 1;
    this.edgeOpacity += (target - this.edgeOpacity) * k;
    this.focusStrength += ((this.focusIdx >= 0 ? 1 : 0) - this.focusStrength) * k;
    this.hoverStrength += ((this.hoverIdx >= 0 ? 1 : 0) - this.hoverStrength) * k;
    const u = this.edgeMat.uniforms;
    u.uOpacity.value = this.edgeOpacity;
    u.uFocus.value = this.focusStrength;
    u.uHover.value = this.hoverStrength;
  }

  // ---- Journey3D ----

  setEdgeRings(depths: Float32Array | null): void {
    if (!this.edgeGeo) return;
    const attr = this.edgeGeo.getAttribute("aRing") as THREE.BufferAttribute;
    if (!depths) {
      (attr.array as Float32Array).fill(1e9);
    } else {
      (attr.array as Float32Array).set(depths);
    }
    attr.needsUpdate = true;
  }

  setEdgeRingProgress(v: number): void {
    if (this.edgeMat) this.edgeMat.uniforms.uRing.value = v;
  }

  setContextLines(from: XYZ, tos: XYZ[], color: RGBA): void {
    this.clearContextLines();
    if (!tos.length) return;
    const pos = new Float32Array(tos.length * 6);
    const col = new Float32Array(tos.length * 6);
    const al = new Float32Array(tos.length * 2);
    tos.forEach((t, i) => {
      pos.set([from[0], from[1], from[2], t[0], t[1], t[2]], i * 6);
      col.set([color[0], color[1], color[2], color[0], color[1], color[2]], i * 6);
      al[i * 2] = 0.85;
      al[i * 2 + 1] = 0.35;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(al, 1));
    const mat = new THREE.ShaderMaterial({
      vertexShader: LINE_VERT,
      fragmentShader: LINE_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.ctxLines = new THREE.LineSegments(g, mat);
    this.ctxLines.frustumCulled = false;
    this.scene.add(this.ctxLines);
  }

  clearContextLines(): void {
    if (!this.ctxLines) return;
    this.scene.remove(this.ctxLines);
    this.ctxLines.geometry.dispose();
    (this.ctxLines.material as THREE.Material).dispose();
    this.ctxLines = null;
  }

  setTrail(points: XYZ[], color: RGBA): void {
    this.clearTrail();
    if (points.length < 2) return;
    const pos = new Float32Array(points.length * 3);
    points.forEach((p, i) => pos.set(p, i * 3));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.LineBasicMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.trail = new THREE.Line(g, mat);
    this.trail.frustumCulled = false;
    this.scene.add(this.trail);
  }

  clearTrail(): void {
    if (!this.trail) return;
    this.scene.remove(this.trail);
    this.trail.geometry.dispose();
    (this.trail.material as THREE.Material).dispose();
    this.trail = null;
  }

  // ---- Chapter 3: 3D Query Beacon ----

  setQueryBeacon(pos: XYZ | null, color: RGBA = INK.query): void {
    this.clearQueryBeacon();
    if (!pos || !this.renderer) return;
    const col = new THREE.Color(color[0], color[1], color[2]);
    const group = new THREE.Group();
    group.position.set(pos[0], pos[1], pos[2]);

    // 1. Glowing central core sphere
    const coreGeo = new THREE.SphereGeometry(1.6, 24, 24);
    const coreMat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    group.add(core);

    // 2. Wireframe target sphere
    const wireGeo = new THREE.SphereGeometry(3.2, 16, 16);
    const wireMat = new THREE.MeshBasicMaterial({
      color: col,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });
    const wireSphere = new THREE.Mesh(wireGeo, wireMat);
    group.add(wireSphere);

    // 3. Two concentric sonar ripple rings in XY
    const ringGeo = new THREE.RingGeometry(0.8, 1.4, 48);
    const rMat1 = new THREE.MeshBasicMaterial({
      color: col,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring1 = new THREE.Mesh(ringGeo, rMat1);
    group.add(ring1);

    const rMat2 = new THREE.MeshBasicMaterial({
      color: col,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring2 = new THREE.Mesh(ringGeo.clone(), rMat2);
    group.add(ring2);

    // 4. Vertical beacon laser beam
    const beamPos = new Float32Array([0, 0, -36, 0, 0, 36]);
    const beamGeo = new THREE.BufferGeometry();
    beamGeo.setAttribute("position", new THREE.BufferAttribute(beamPos, 3));
    const beamMat = new THREE.LineBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });
    const beam = new THREE.LineSegments(beamGeo, beamMat);
    group.add(beam);

    this.scene.add(group);
    this.queryBeacon = { group, core, wireSphere, ring1, ring2, beam };
  }

  clearQueryBeacon(): void {
    if (!this.queryBeacon) return;
    const { group, core, wireSphere, ring1, ring2, beam } = this.queryBeacon;
    this.scene.remove(group);
    core.geometry.dispose();
    (core.material as THREE.Material).dispose();
    wireSphere.geometry.dispose();
    (wireSphere.material as THREE.Material).dispose();
    ring1.geometry.dispose();
    (ring1.material as THREE.Material).dispose();
    ring2.geometry.dispose();
    (ring2.material as THREE.Material).dispose();
    beam.geometry.dispose();
    (beam.material as THREE.Material).dispose();
    this.queryBeacon = null;
  }

  private animateQueryBeacon(t: number): void {
    if (!this.queryBeacon) return;
    const { core, wireSphere, ring1, ring2 } = this.queryBeacon;
    wireSphere.rotation.y = t * 0.9;
    wireSphere.rotation.x = t * 0.45;
    const pulse = 1.0 + 0.22 * Math.sin(t * 4.2);
    core.scale.setScalar(pulse);

    // Expand sonar ripple rings continuously
    const p1 = (t * 1.3) % 1.0;
    const s1 = 0.5 + p1 * 8.5;
    ring1.scale.setScalar(s1);
    (ring1.material as THREE.MeshBasicMaterial).opacity = Math.max(0, (1.0 - p1) * 0.9);

    const p2 = (t * 1.3 + 0.5) % 1.0;
    const s2 = 0.5 + p2 * 8.5;
    ring2.scale.setScalar(s2);
    (ring2.material as THREE.MeshBasicMaterial).opacity = Math.max(0, (1.0 - p2) * 0.9);
  }

  // ---- Chapter 5: Animated Graph Walk Edges ----

  setGraphWalkEdges(
    edges: { from: XYZ; to: XYZ; color?: RGBA }[],
    options?: { intensity?: number },
  ): void {
    this.clearGraphWalkEdges();
    if (!edges.length || !this.renderer) return;

    const count = edges.length;
    const pos = new Float32Array(count * 6);
    const col = new Float32Array(count * 6);
    const prog = new Float32Array(count * 2);

    for (let i = 0; i < count; i++) {
      const e = edges[i];
      const c = e.color ?? INK.gold;
      pos[i * 6] = e.from[0];
      pos[i * 6 + 1] = e.from[1];
      pos[i * 6 + 2] = e.from[2];
      pos[i * 6 + 3] = e.to[0];
      pos[i * 6 + 4] = e.to[1];
      pos[i * 6 + 5] = e.to[2];

      for (let j = 0; j < 6; j += 3) {
        col[i * 6 + j] = c[0];
        col[i * 6 + j + 1] = c[1];
        col[i * 6 + j + 2] = c[2];
      }

      prog[i * 2] = 0.0;
      prog[i * 2 + 1] = 1.0;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    geo.setAttribute("aProgress", new THREE.BufferAttribute(prog, 1));

    const mat = new THREE.ShaderMaterial({
      vertexShader: PULSE_EDGE_VERT,
      fragmentShader: PULSE_EDGE_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uBaseAlpha: { value: (options?.intensity ?? 1.0) * 0.55 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const mesh = new THREE.LineSegments(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 3;
    this.scene.add(mesh);

    this.graphWalkMesh = mesh;
    this.graphWalkMat = mat;
  }

  clearGraphWalkEdges(): void {
    if (!this.graphWalkMesh) return;
    this.scene.remove(this.graphWalkMesh);
    this.graphWalkMesh.geometry.dispose();
    this.graphWalkMat?.dispose();
    this.graphWalkMesh = null;
    this.graphWalkMat = null;
  }

  // ---- Chapter 7: Context Assembly Cluster Hull ----

  setClusterHull(points: XYZ[], color: RGBA = INK.gold): void {
    this.clearClusterHull();
    if (!points.length || !this.renderer) return;

    const group = new THREE.Group();
    const c = new THREE.Color(color[0], color[1], color[2]);

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const p of points) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }

    const pad = 3.5;
    const sx = Math.max(5, maxX - minX + pad * 2);
    const sy = Math.max(5, maxY - minY + pad * 2);
    const sz = Math.max(4, maxZ - minZ + pad * 2);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    const boxGeo = new THREE.BoxGeometry(sx, sy, sz);
    const edgeGeo = new THREE.EdgesGeometry(boxGeo);
    const edgeMat = new THREE.LineBasicMaterial({
      color: c,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });
    const edgeMesh = new THREE.LineSegments(edgeGeo, edgeMat);
    edgeMesh.position.set(cx, cy, cz);
    group.add(edgeMesh);

    const fillMat = new THREE.MeshBasicMaterial({
      color: c,
      transparent: true,
      opacity: 0.07,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const fillMesh = new THREE.Mesh(boxGeo, fillMat);
    fillMesh.position.set(cx, cy, cz);
    group.add(fillMesh);

    // Internal constellation links connecting each point to the centroid
    const linksPos = new Float32Array(points.length * 6);
    points.forEach((p, i) => {
      linksPos[i * 6] = cx;
      linksPos[i * 6 + 1] = cy;
      linksPos[i * 6 + 2] = cz;
      linksPos[i * 6 + 3] = p[0];
      linksPos[i * 6 + 4] = p[1];
      linksPos[i * 6 + 5] = p[2];
    });
    const linksGeo = new THREE.BufferGeometry();
    linksGeo.setAttribute("position", new THREE.BufferAttribute(linksPos, 3));
    const linksMat = new THREE.LineBasicMaterial({
      color: c,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    const linksMesh = new THREE.LineSegments(linksGeo, linksMat);
    group.add(linksMesh);

    this.scene.add(group);
    this.clusterHull = group;
  }

  clearClusterHull(): void {
    if (!this.clusterHull) return;
    this.scene.remove(this.clusterHull);
    this.clusterHull.children.forEach((ch: THREE.Object3D) => {
      if ("geometry" in ch) (ch as THREE.Mesh).geometry.dispose();
      if ("material" in ch) {
        const mat = (ch as THREE.Mesh).material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    });
    this.clusterHull = null;
  }

  private animateClusterHull(t: number): void {
    if (!this.clusterHull) return;
    const lineMesh = this.clusterHull.children[0] as THREE.LineSegments | undefined;
    if (lineMesh && lineMesh.material) {
      (lineMesh.material as THREE.LineBasicMaterial).opacity = 0.55 + 0.25 * Math.sin(t * 2.4);
    }
  }

  // ---- Chapter 8: Citation Beams ----

  setCitationBeams(items: { from: XYZ; to: XYZ; color?: RGBA }[]): void {
    this.clearCitationBeams();
    if (!items.length || !this.renderer) return;

    const count = items.length;
    const pos = new Float32Array(count * 6);
    const col = new Float32Array(count * 6);

    for (let i = 0; i < count; i++) {
      const it = items[i];
      const c = it.color ?? INK.gold;
      pos[i * 6] = it.from[0];
      pos[i * 6 + 1] = it.from[1];
      pos[i * 6 + 2] = it.from[2];
      pos[i * 6 + 3] = it.to[0];
      pos[i * 6 + 4] = it.to[1];
      pos[i * 6 + 5] = it.to[2];

      for (let j = 0; j < 6; j += 3) {
        col[i * 6 + j] = c[0];
        col[i * 6 + j + 1] = c[1];
        col[i * 6 + j + 2] = c[2];
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
    });

    const mesh = new THREE.LineSegments(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 4;
    this.scene.add(mesh);
    this.citationBeams = mesh;
  }

  clearCitationBeams(): void {
    if (!this.citationBeams) return;
    this.scene.remove(this.citationBeams);
    this.citationBeams.geometry.dispose();
    (this.citationBeams.material as THREE.Material).dispose();
    this.citationBeams = null;
  }

  setControlsEnabled(on: boolean): void {
    if (this.controls) {
      this.controls.enabled = on;
      // The cosmos overview wants its gentle auto-orbit; a scripted travel /
      // presentation camera must be the ONLY writer, so autoRotate follows the
      // same switch. An always-on autoRotate is a hidden second writer that
      // slowly drifts the camera off a solved pose (measured as late-stage
      // overshoot that grows with dwell time).
      this.controls.autoRotate = on;
    }
    // an explicit state change with no flight in progress retires any pending
    // restore, so a later flight can't resurrect an old enabled/autoRotate
    if (!this.camRaf) this.camPrev = null;
  }

  /**
   * Star selection treatment: brightness/size boost on the star (aSel
   * attribute) + a thin animated billboard ring around it. Monochrome
   * (single accent colour), matches the icon language.
   */
  setSelection(idx: number | null): void {
    if (!this.selAttr || !this.geo) return;
    const arr = this.selAttr.array as Float32Array;
    if (this.selIdx >= 0 && this.selIdx < this.n) arr[this.selIdx] = 0;
    this.selIdx = idx ?? -1;
    if (this.selIdx >= 0 && this.selIdx < this.n) {
      arr[this.selIdx] = 1;
      this.selAttr.needsUpdate = true;
      this.ensureRing();
      const ring = this.selRing!;
      // radius is screen-space sized per frame in animateRing(); just place it
      ring.scale.setScalar(1);
      ring.position.set(this.pos[this.selIdx * 3], this.pos[this.selIdx * 3 + 1], this.pos[this.selIdx * 3 + 2]);
      ring.visible = true;
    } else {
      this.selAttr.needsUpdate = true;
      if (this.selRing) this.selRing.visible = false;
    }
  }

  /** Project a node to CSS px (for the screen-space selection tooltip). */
  projectNode(idx: number): { x: number; y: number; visible: boolean } | null {
    if (!this.camera || !this.canvas || idx < 0 || idx >= this.n) return null;
    this.projectAll();
    return {
      x: this.screen[idx * 2],
      y: this.screen[idx * 2 + 1],
      visible: this.screen[idx * 2] > -50 && this.screen[idx * 2] < (this.canvas.clientWidth || 1) + 50,
    };
  }

  private ensureRing(): void {
    if (this.selRing || !this.renderer) return;
    // thin judged-on-screen annulus: thickness is a fraction of the radius
    const g = new THREE.RingGeometry(0.8, 1.0, 64);
    const m = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.49, 0.95, 1.0),
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
    });
    const ring = new THREE.Mesh(g, m);
    ring.visible = false;
    ring.frustumCulled = false;
    ring.renderOrder = 4; // over the starfield, under the HTML tooltip
    this.scene.add(ring);
    this.selRing = ring;
  }

  /**
   * Animate the selection ring: billboard to camera, slow spin, subtle breathe.
   * Sized in SCREEN px (a fixed world radius is sub-pixel on a zoomed-out view —
   * the star sprite is only a few px there, so the ring would never be seen):
   * radius hugs the sprite but never drops below RING_MIN_PX.
   */
  private animateRing(t: number): void {
    if (!this.selRing || !this.selRing.visible || this.selIdx < 0 || !this.ptsMat) return;
    const r = this.selRing;
    r.quaternion.copy(this.camera!.quaternion);
    const size = (this.geo!.getAttribute("aSize") as THREE.BufferAttribute).getX(this.selIdx);
    // the vertex shader boosts the selected sprite: aSize * (1 + aSel*1.1)
    const uScale = this.ptsMat.uniforms.uScale.value as number;
    const dist = Math.max(0.1, this.camera!.position.distanceTo(r.position));
    const spritePx = (size * 2.1 * uScale) / dist;
    // hug the star sprite (the shader clamps sprites at 90 device px, so the
    // nominal size can overshoot) but never below RING_MIN_PX
    const targetPx = Math.max(RING_MIN_PX * this.dpr, Math.min(spritePx, 90) * 0.62) * (1 + 0.05 * Math.sin(t * 2.6));
    r.scale.setScalar((targetPx * dist) / uScale);
    r.rotateZ(t * 0.6);
    (r.material as THREE.MeshBasicMaterial).opacity = 0.7 + 0.2 * Math.sin(t * 2.6);
  }

  setHalos(items: { idx: number; color?: RGBA; size?: number }[] | null): void {
    this.clearHalos();
    if (!items || !items.length || !this.renderer || !this.pos.length) return;
    for (const item of items) {
      if (item.idx < 0 || item.idx >= this.n) continue;
      const g = new THREE.RingGeometry(0.75, 1.25, 48);
      const col = item.color ?? INK.cyan;
      const m = new THREE.MeshBasicMaterial({
        color: new THREE.Color(col[0], col[1], col[2]),
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: false,
      });
      const mesh = new THREE.Mesh(g, m);
      mesh.frustumCulled = false;
      mesh.renderOrder = 4;
      mesh.position.set(this.pos[item.idx * 3], this.pos[item.idx * 3 + 1], this.pos[item.idx * 3 + 2]);
      this.scene.add(mesh);
      this.halos.push({ idx: item.idx, color: col, mesh, size: item.size ?? 1 });
    }
  }

  clearHalos(): void {
    for (const h of this.halos) {
      this.scene.remove(h.mesh);
      h.mesh.geometry.dispose();
      (h.mesh.material as THREE.Material).dispose();
    }
    this.halos = [];
  }

  private animateHalos(t: number): void {
    if (!this.halos.length || !this.camera || !this.ptsMat) return;
    const uScale = this.pointScale();
    for (const h of this.halos) {
      h.mesh.quaternion.copy(this.camera.quaternion);
      const dist = Math.max(0.1, this.camera.position.distanceTo(h.mesh.position));
      const basePx = Math.max(16, 20 * h.size);
      const targetPx = basePx * this.dpr * (1 + 0.16 * Math.sin(t * 3.6 + h.idx * 0.5));
      h.mesh.scale.setScalar((targetPx * dist) / uScale);
      h.mesh.rotateZ(t * 0.4);
      (h.mesh.material as THREE.MeshBasicMaterial).opacity = 0.65 + 0.3 * Math.sin(t * 3.6 + h.idx * 0.5);
    }
  }

  pulseCluster(indices: number[], color: RGBA, fly = true): void {
    if (!indices.length || !this.n) return;
    const valid = indices.filter((i) => i >= 0 && i < this.n);
    if (!valid.length) return;
    this.spotlight(new Set(valid));
    const paints = new Map<number, { color: RGBA; size?: number }>();
    valid.forEach((i) => paints.set(i, { color, size: 2.8 }));
    this.paint(paints);
    this.setHalos(valid.map((idx) => ({ idx, color, size: 1.3 })));
    if (fly && this.camera) {
      let sx = 0, sy = 0, sz = 0;
      for (const i of valid) {
        sx += this.pos[i * 3];
        sy += this.pos[i * 3 + 1];
        sz += this.pos[i * 3 + 2];
      }
      const cx = sx / valid.length;
      const cy = sy / valid.length;
      const cz = sz / valid.length;
      let maxDist = 0;
      for (const i of valid) {
        const dx = this.pos[i * 3] - cx;
        const dy = this.pos[i * 3 + 1] - cy;
        const dz = this.pos[i * 3 + 2] - cz;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > maxDist) maxDist = d;
      }
      const fitDist = Math.max(18, Math.min(130, maxDist * 2.4));
      this.flyToNode([cx, cy, cz], fitDist, 1000);
    }
  }

  cameraNow(): { pos: XYZ; target: XYZ } {
    const c = this.camera!;
    const t = this.controls?.target ?? new THREE.Vector3();
    return {
      pos: [c.position.x, c.position.y, c.position.z],
      target: [t.x, t.y, t.z],
    };
  }

  viewInfo(): { fov: number; aspect: number } {
    return { fov: this.camera?.fov ?? 55, aspect: this.camera?.aspect ?? 1.6 };
  }

  /**
   * THE camera flight. Every camera move goes through here so there is
   * exactly one writer per frame and one easing convention.
   *
   * - The arrival pose is solved by the caller (`toPos`/`toTgt`) and never
   *   changes mid-flight, so the eased 0→1 parameter cannot overshoot.
   * - OrbitControls is taken out of the loop for the whole flight: damping /
   *   pan / dolly residuals are drained first (with `enableDamping` off,
   *   OrbitControls.update() zeroes them — its own else-branch), autoRotate
   *   is suspended and `enabled` is false so no input can inject more. The
   *   tween is the only thing that calls update(), once per frame.
   * - At t=1 the previous control state is restored and re-synced exactly
   *   once, leaving the target on the node so OrbitControls orbits THE NODE.
   */
  private tweenCamera(toPosV: THREE.Vector3, toTgtV: THREE.Vector3, ms: number, ease: CamEase): void {
    if (!this.camera) return;
    this.stopCamRaf();
    const cam = this.camera;
    const ctl = this.controls;
    const fromPos = cam.position.clone();
    const fromTgt = ctl ? ctl.target.clone() : new THREE.Vector3();
    // remember the control state only on the FIRST flight of a chain, so a
    // second fly-to mid-flight cannot latch controls disabled forever
    if (ctl && this.camPrev === null) this.camPrev = { enabled: ctl.enabled, auto: ctl.autoRotate };
    if (ctl) {
      const damp = ctl.enableDamping;
      ctl.enableDamping = false;
      ctl.update();
      ctl.enableDamping = damp;
      ctl.autoRotate = false;
      ctl.enabled = false;
    }
    const finish = () => {
      this.camBusy = false;
      const prev = this.camPrev;
      this.camPrev = null;
      if (ctl && prev) {
        ctl.enabled = prev.enabled;
        ctl.autoRotate = prev.auto;
        ctl.update();
      }
      this.screenDirty = true;
    };
    if (ms <= 0) {
      cam.position.copy(toPosV);
      if (ctl) ctl.target.copy(toTgtV);
      cam.lookAt(toTgtV);
      finish();
      return;
    }
    this.camBusy = true;
    const t0 = performance.now();
    const step = (t: number) => {
      const p = Math.min(1, (t - t0) / ms);
      const e = applyEase(p, ease);
      cam.position.lerpVectors(fromPos, toPosV, e);
      if (ctl) {
        ctl.target.lerpVectors(fromTgt, toTgtV, e);
        ctl.update();
      } else {
        cam.lookAt(toTgtV);
      }
      this.screenDirty = true;
      if (p < 1) {
        this.camRaf = requestAnimationFrame(step);
      } else {
        this.camRaf = 0;
        finish();
      }
    };
    this.camRaf = requestAnimationFrame(step);
  }

  cameraSet(pos: XYZ, target: XYZ): void {
    this.tweenCamera(
      new THREE.Vector3(pos[0], pos[1], pos[2]),
      new THREE.Vector3(target[0], target[1], target[2]),
      0,
      "linear",
    );
  }

  cameraFly(pos: XYZ, target: XYZ, ms = 1400, ease: "out" | "inout" | "linear" = "inout"): void {
    this.tweenCamera(
      new THREE.Vector3(pos[0], pos[1], pos[2]),
      new THREE.Vector3(target[0], target[1], target[2]),
      ms,
      ease,
    );
  }

  /**
   * THE node focus. The arrival pose is solved ONCE: target → the node,
   * camera → the node plus the current line of sight × `dist`. The node ends
   * dead-centre, the requested distance is honoured exactly, and — because
   * the eased parameter is monotonic — the camera can never overshoot or
   * arrive somewhere else.
   */
  flyToNode(p: XYZ, dist = 9.5, ms = 850): void {
    const cam = this.camera;
    if (!cam) return;
    const tgt = this.controls?.target ?? new THREE.Vector3();
    const arrive = solveArrival(
      [p[0], p[1], p[2]],
      [cam.position.x, cam.position.y, cam.position.z],
      [tgt.x, tgt.y, tgt.z],
      dist,
    );
    this.tweenCamera(
      new THREE.Vector3(arrive.pos[0], arrive.pos[1], arrive.pos[2]),
      new THREE.Vector3(arrive.tgt[0], arrive.tgt[1], arrive.tgt[2]),
      ms,
      "inout",
    );
  }

  /**
   * Instant camera placement on the surface of a sphere around `center`.
   *
   * Deliberately does NOT cancel an in-flight tween: a user fly-to must win
   * over cinematic camera choreography. The choreography itself must stop
   * driving the camera when the user takes over (the caller owns that — see
   * TravelView's cancellable camera track).
   */
  orbitAround(center: XYZ, dist: number, azimuth: number, elevation: number): void {
    if (!this.camera) return;
    const ce = Math.cos(elevation);
    const pos: XYZ = [
      center[0] + dist * ce * Math.sin(azimuth),
      center[1] + dist * Math.sin(elevation),
      center[2] + dist * ce * Math.cos(azimuth),
    ];
    this.tweenCamera(new THREE.Vector3(pos[0], pos[1], pos[2]), new THREE.Vector3(center[0], center[1], center[2]), 0, "linear");
  }

  /** 2D-compat flight: hover the camera above (x, y, 0) at the zoom-implied distance. */
  flyTo(x: number, y: number, zoom: number, ms = 1400): void {
    const dist = this.distForZoom(zoom);
    this.tweenCamera(new THREE.Vector3(x, y, dist), new THREE.Vector3(x, y, 0), ms, "inout");
  }

  fit(padding = 0.12): void {
    if (!this.n || !this.camera) return;
    // bounding sphere over all stars
    const sphere = new THREE.Sphere();
    new THREE.Box3().setFromBufferAttribute(
      new THREE.BufferAttribute(this.pos, 3),
    ).getBoundingSphere(sphere);
    const fov = (this.camera.fov * Math.PI) / 180;
    const dist = (sphere.radius / Math.sin(fov / 2)) * (1 + padding) + sphere.radius * 0.1;
    this.fitDist = Math.max(2, dist);
    if (this.ptsMat) {
      this.ptsMat.uniforms.uFogRef.value = this.fitDist;
    }
    this.cameraSet([0, this.fitDist * 0.42, this.fitDist], [0, 0, 0]);
  }

  /** Camera distance that yields `zoom` (CSS px per world unit at the target). */
  private distForZoom(zoom: number): number {
    const h = this.canvas?.clientHeight || 600;
    const tanFov = Math.tan(((this.camera?.fov ?? 55) * Math.PI) / 180 / 2);
    return Math.max(1.5, h / (2 * Math.max(1e-4, zoom) * tanFov));
  }

  currentZoom(): number {
    if (!this.camera) return 1;
    const d = this.camera.position.distanceTo(this.controls?.target ?? new THREE.Vector3());
    const h = this.canvas?.clientHeight || 600;
    const tanFov = Math.tan(((this.camera?.fov ?? 55) * Math.PI) / 180 / 2);
    return h / (2 * Math.max(0.5, d) * tanFov);
  }

  destroy(): void {
    this.stopCamRaf();
    this.camBusy = false;
    this.camPrev = null;
    cancelAnimationFrame(this.hoverRaf);
    cancelAnimationFrame(this.pulseRaf);
    this.renderer?.setAnimationLoop(null);
    this.ro?.disconnect();
    this.ro = null;
    if (this.canvas) {
      this.canvas.removeEventListener("pointermove", this.onMove);
      this.canvas.removeEventListener("pointerleave", this.onLeave);
      this.canvas.removeEventListener("click", this.onClick);
      this.canvas.removeEventListener("pointerdown", this.onDown);
    }
    this.clearMarkers();
    this.clearHalos();
    this.clearContextLines();
    this.clearTrail();
    if (this.selRing) {
      this.scene.remove(this.selRing);
      this.selRing.geometry.dispose();
      (this.selRing.material as THREE.Material).dispose();
      this.selRing = null;
    }
    this.controls?.dispose();
    this.controls = null;
    this.geo?.dispose();
    this.ptsMat?.dispose();
    this.edgeGeo?.dispose();
    this.edgeMat?.dispose();
    this.renderer?.dispose();
    this.renderer = null;
    this.canvas = null;
    this.camera = null;
  }

  // ---- internals ----

  private stopCamRaf(): void {
    cancelAnimationFrame(this.camRaf);
    this.camRaf = 0;
  }

  private pointScale(): number {
    const h = (this.canvas?.clientHeight || 600) * this.dpr;
    const fov = (this.camera?.fov ?? 55) * (Math.PI / 180);
    return h / (2 * Math.tan(fov / 2));
  }

  private resize(): void {
    const c = this.canvas;
    if (!c || !this.renderer || !this.camera) return;
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const s = this.pointScale();
    if (this.ptsMat) this.ptsMat.uniforms.uScale.value = s;
    for (const m of this.markers.values()) m.mat.uniforms.uScale.value = s;
  }

  /** Gentle clock driving twinkle + marker pulse (one uniform write per frame). */
  private startPulseClock(): void {
    this.pulseT0 = performance.now();
    const tick = (now: number) => {
      const t = (now - this.pulseT0) / 1000;
      if (this.ptsMat) this.ptsMat.uniforms.uTime.value = t;
      for (const m of this.markers.values()) m.mat.uniforms.uPulse.value = t;
      this.pulseRaf = requestAnimationFrame(tick);
    };
    this.pulseRaf = requestAnimationFrame(tick);
  }

  /** Alpha buffer from hidden families + spotlight + explicit paints. */
  private recomputeAlphas(): void {
    if (!this.alphaAttr || !this.n) return;
    const arr = this.alphaAttr.array as Float32Array;
    for (let i = 0; i < this.n; i++) {
      let a = 1;
      if (this.hidden.has(familyOf(this.types[i]))) a = 0.07;
      if (this.spot && !this.spot.has(i)) a = Math.min(a, SPOT_DIM);
      const p = this.paints.get(i);
      if (p) a = 1; // explicit highlights win over dim/ghost
      arr[i] = a;
    }
    this.alphaAttr.needsUpdate = true;
  }

  private recomputeColorsSizes(): void {
    if (!this.colAttr || !this.sizeAttr || !this.n) return;
    const col = this.colAttr.array as Float32Array;
    const siz = this.sizeAttr.array as Float32Array;
    for (let i = 0; i < this.n; i++) {
      col[i * 3] = this.baseCol[i * 3];
      col[i * 3 + 1] = this.baseCol[i * 3 + 1];
      col[i * 3 + 2] = this.baseCol[i * 3 + 2];
      siz[i] = this.baseSize[i];
    }
    for (const [idx, p] of this.paints) {
      if (idx < 0 || idx >= this.n) continue;
      col[idx * 3] = p.color[0];
      col[idx * 3 + 1] = p.color[1];
      col[idx * 3 + 2] = p.color[2];
      siz[idx] = p.size ?? this.baseSize[idx] * 2.2;
    }
    this.colAttr.needsUpdate = true;
    this.sizeAttr.needsUpdate = true;
  }

  // ---- picking (precision screen-space) ----

  private projectAll(): void {
    if (!this.screenDirty || !this.camera || !this.canvas || this.n === 0) return;
    if (this.screen.length !== this.n * 2) this.screen = new Float32Array(this.n * 2);
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    const v = new THREE.Vector3();
    for (let i = 0; i < this.n; i++) {
      v.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
      v.project(this.camera);
      this.screen[i * 2] = (v.x * 0.5 + 0.5) * w;
      this.screen[i * 2 + 1] = (-v.y * 0.5 + 0.5) * h;
    }
    this.screenDirty = false;
  }

  private pick(clientX: number, clientY: number): number | null {
    if (this.n === 0 || !this.canvas) return null;
    this.projectAll();
    const rect = this.canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    // pick radius in screen px (positions are already projected)
    const R = 9;
    let best = -1;
    let bestD = R * R;
    for (let i = 0; i < this.n; i++) {
      if (this.hidden.has(familyOf(this.types[i]))) continue;
      const dx = this.screen[i * 2] - mx;
      const dy = this.screen[i * 2 + 1] - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best >= 0 ? best : null;
  }

  private onMove = (e: PointerEvent) => {
    // pick only when a consumer needs it: the hover callback (cosmos view) or
    // the edge-hover highlight (adaptive edge mode). Keeps travel/presentation
    // (small subgraphs, no hover callback) from scanning on every move.
    if ((!this.onHover && !this.edgeAdaptive) || this.hoverRaf) return;
    const cx = e.clientX;
    const cy = e.clientY;
    this.hoverRaf = requestAnimationFrame(() => {
      this.hoverRaf = 0;
      const idx = this.pick(cx, cy);
      if (this.onHover) this.onHover(idx === null ? null : this.selOf(idx));
      // reuse the SAME pick (no extra raycast) for the edge highlight
      if (this.edgeAdaptive) this.updateEdgeHover(idx);
    });
  };

  /** Edge-hover from the existing pick, throttled to >= 30 ms. */
  private updateEdgeHover(idx: number | null): void {
    if (idx === this.hoverIdx) return;
    const now = performance.now();
    if (now - this.lastHoverT < 30) return;
    this.lastHoverT = now;
    this.setEdgeHover(idx);
  }

  private onLeave = () => {
    if (this.onHover) this.onHover(null);
    if (this.edgeAdaptive) this.setEdgeHover(null);
  };

  private onDown = (e: PointerEvent) => {
    // Remember where the gesture started: a click that ends a drag must NOT
    // select (the 2D fallback had the same guard — without it every orbit
    // drag ended with a bogus "click" that flew the camera somewhere else).
    this.downAt = { x: e.clientX, y: e.clientY };
    // a user grab always wins over any in-flight camera tween
    if (this.camRaf) {
      this.stopCamRaf();
      this.camBusy = false;
      const prev = this.camPrev;
      this.camPrev = null;
      if (this.controls) {
        this.controls.enabled = prev ? prev.enabled : true;
        this.controls.update();
      }
    }
  };

  private onClick = (e: MouseEvent) => {
    if (!this.onSelect) return;
    // ignore clicks that are really the end of an orbit/zoom drag
    if (this.downAt) {
      const dx = e.clientX - this.downAt.x;
      const dy = e.clientY - this.downAt.y;
      if (dx * dx + dy * dy > 36) return;
    }
    const idx = this.pick(e.clientX, e.clientY);
    // empty space deselects; a star selects
    this.onSelect(idx === null ? null : this.selOf(idx));
  };

  private selOf(idx: number): SelectedNode {
    return { gpuId: idx, graphIdx: idx, name: this.names[idx] ?? `#${idx}` };
  }
}

/** Does this renderer support the 3D journey extensions? */
export function asJourney3D(c: unknown): (Journey3D & CosmosLike) | null {
  return c instanceof Cosmos3D ? (c as unknown as Journey3D & CosmosLike) : null;
}

// re-export so existing imports keep working
export { INK };
