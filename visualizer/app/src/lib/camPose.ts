// Pure camera-pose math for fly-to flights. No three.js import so it can be
// unit-tested directly (node scripts/flyto-pose.test.mjs).
//
// A "fly to a node" is fully described by its ARRIVAL POSE:
//   target = the node            (so the node ends dead-centre and
//                                 OrbitControls then orbits the node)
//   camera = node + dir * dist   (dir = current line of sight, normalised)
// Both endpoints are solved ONCE, then position and target are interpolated
// with the same single eased 0→1 parameter. That is overshoot-free by
// construction: the eased parameter is monotonic in [0,1], so the camera path
// is a straight segment that stops exactly on the arrival pose and never
// travels past it.

export type Vec3 = [number, number, number];

export type CamEase = "out" | "inout" | "linear";

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function applyEase(t: number, ease: CamEase): number {
  const c = t <= 0 ? 0 : t >= 1 ? 1 : t;
  return ease === "out" ? easeOutCubic(c) : ease === "inout" ? easeInOutCubic(c) : c;
}

export function lerp3(a: Vec3, b: Vec3, e: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e];
}

export interface ArrivalPose {
  /** camera position at t=1 */
  pos: Vec3;
  /** controls.target at t=1 — exactly the node */
  tgt: Vec3;
  /** |pos − tgt| — the honoured flight distance */
  dist: number;
}

/**
 * Solve the arrival pose for a fly-to-node.
 *
 * `dir` is the direction the camera sits relative to its target — i.e. the
 * current line of sight — so the flight is a dive along the existing view
 * rather than a cut. When the camera is (numerically) on its target, a
 * sensible up-ish default is used instead of dividing by ~0.
 */
export function solveArrival(node: Vec3, camPos: Vec3, camTarget: Vec3, dist: number): ArrivalPose {
  let dx = camPos[0] - camTarget[0];
  let dy = camPos[1] - camTarget[1];
  let dz = camPos[2] - camTarget[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-4 || !Number.isFinite(len)) {
    dx = 0;
    dy = 0.35;
    dz = 1;
    const dl = Math.hypot(dx, dy, dz);
    dx /= dl;
    dy /= dl;
    dz /= dl;
  } else {
    dx /= len;
    dy /= len;
    dz /= len;
  }
  const d = dist > 0 && Number.isFinite(dist) ? dist : 9.5;
  return {
    pos: [node[0] + dx * d, node[1] + dy * d, node[2] + dz * d],
    tgt: [node[0], node[1], node[2]],
    dist: d,
  };
}

/**
 * The ONE cinematic look direction used by every journey keyframe
 * (up-and-behind +z), normalised. All stage poses share it, so consecutive
 * keyframes differ only by translation: the linear (monotone) interpolation
 * between two poses is a straight segment and can never swing the camera
 * through a wide arc or past the arrival.
 */
export const CAM_DIR: Vec3 = (() => {
  const l = Math.hypot(0.18, 0.42, 0.94) || 1;
  return [0.18 / l, 0.42 / l, 0.94 / l] as Vec3;
})();

export function centroid3(points: Vec3[]): Vec3 | null {
  if (!points.length) return null;
  let x = 0;
  let y = 0;
  let z = 0;
  for (const p of points) {
    x += p[0];
    y += p[1];
    z += p[2];
  }
  return [x / points.length, y / points.length, z / points.length];
}

/** Largest distance from `center` to any point (0 for a single point). */
export function boundingRadius(points: Vec3[], center: Vec3): number {
  let r = 0;
  for (const p of points) {
    const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    if (d > r) r = d;
  }
  return r;
}

/**
 * Camera distance that fits a sphere of `radius` (centred on the target) into
 * the frustum. Uses the tighter of the vertical/horizontal half-angles so the
 * content fits on both axes for any canvas aspect. Radius 0 (single point)
 * collapses to `minDist`.
 */
export function fitDistance(
  radius: number,
  fovDeg: number,
  aspect: number,
  padding = 0.15,
  minDist = 6,
): number {
  const fov = fovDeg > 0 && Number.isFinite(fovDeg) ? fovDeg : 55;
  const a = aspect > 0 && Number.isFinite(aspect) ? aspect : 1.6;
  const vHalf = (fov * Math.PI) / 180 / 2;
  const hHalf = Math.atan(Math.tan(vHalf) * a);
  const half = Math.max(0.02, Math.min(vHalf, hHalf));
  const r = Number.isFinite(radius) ? Math.max(0, radius) : 0;
  const d = (r / Math.sin(half)) * (1 + padding) + r * 0.1;
  return Math.max(minDist, d);
}

export interface FocusPose extends ArrivalPose {
  /** bounding radius of the framed focus set (0 for a single point) */
  radius: number;
}

/**
 * Solve the cinematic arrival pose that FRAMES a set of focus points:
 * target = their centroid, camera = centroid + CAM_DIR × distance-to-fit.
 * The pose is the single source of truth for a journey keyframe — playback
 * interpolates straight to it with one eased parameter (no overshoot).
 */
export function solveFocusPose(
  points: Vec3[],
  fovDeg: number,
  aspect: number,
  padding = 0.15,
  minDist = 6,
): FocusPose {
  const c = centroid3(points) ?? ([0, 0, 0] as Vec3);
  const r = boundingRadius(points, c);
  const d = fitDistance(r, fovDeg, aspect, padding, minDist);
  return {
    pos: [c[0] + CAM_DIR[0] * d, c[1] + CAM_DIR[1] * d, c[2] + CAM_DIR[2] * d],
    tgt: c,
    dist: d,
    radius: r,
  };
}
