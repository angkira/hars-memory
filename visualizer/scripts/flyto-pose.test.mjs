// Unit tests for the pure fly-to pose math (app/src/lib/camPose.ts).
// Run: node scripts/flyto-pose.test.mjs   (exit 1 on failure)
//
// Node's type-stripping loads the .ts module directly — no bundler needed.
import {
  applyEase,
  lerp3,
  solveArrival,
  CAM_DIR,
  centroid3,
  boundingRadius,
  fitDistance,
  solveFocusPose,
} from "../app/src/lib/camPose.ts";
import { stageCamera } from "../app/src/lib/journeyCam.ts";

let failures = 0;
const ok = (cond, msg) => {
  console.log(`${cond ? "ok  " : "FAIL"}  ${msg}`);
  if (!cond) failures++;
};
const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const vClose = (a, b, eps = 1e-9) => a.every((v, i) => close(v, b[i], eps));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// ---- easing
for (const ease of ["inout", "out", "linear"]) {
  ok(close(applyEase(0, ease), 0), `${ease}: e(0)=0`);
  ok(close(applyEase(1, ease), 1), `${ease}: e(1)=1`);
  let mono = true;
  let prev = -Infinity;
  for (let i = 0; i <= 100; i++) {
    const v = applyEase(i / 100, ease);
    if (v < prev - 1e-12) mono = false;
    if (v < -1e-12 || v > 1 + 1e-12) mono = false;
    prev = v;
  }
  ok(mono, `${ease}: monotonic and within [0,1]`);
  ok(close(applyEase(-5, ease), 0), `${ease}: clamps t<0`);
  ok(close(applyEase(5, ease), 1), `${ease}: clamps t>1`);
}

// ---- solveArrival: the arrival pose is EXACT
{
  const node = [10, -20, 3];
  const camPos = [0, 0, 100];
  const camTgt = [0, 0, 0];
  const req = 28;
  const a = solveArrival(node, camPos, camTgt, req);
  ok(vClose(a.tgt, node), "arrival: target === node (node ends dead-centre)");
  ok(close(a.dist, req), `arrival: requested distance honoured (${a.dist} === ${req})`);
  ok(close(dist(a.pos, a.tgt), req), "arrival: |camera − target| === requested distance");
  // camera sits along the current line of sight
  const dir = [camPos[0] - camTgt[0], camPos[1] - camTgt[1], camPos[2] - camTgt[2]];
  const len = Math.hypot(...dir);
  const want = node.map((v, i) => v + (dir[i] / len) * req);
  ok(vClose(a.pos, want, 1e-9), "arrival: camera = node + normalised line-of-sight × dist");
}

// ---- solveArrival: degenerate camera-on-target uses a sane default, never NaN
{
  const a = solveArrival([1, 2, 3], [1, 2, 3], [1, 2, 3], 9.5);
  ok(a.pos.every(Number.isFinite) && a.tgt.every(Number.isFinite), "degenerate: finite pose");
  ok(close(dist(a.pos, a.tgt), 9.5), "degenerate: distance still honoured");
}

// ---- solveArrival: bad distance falls back (never 0 / NaN)
{
  const a = solveArrival([0, 0, 0], [0, 0, 5], [0, 0, 0], NaN);
  ok(Number.isFinite(a.dist) && a.dist > 0, "bad dist: finite positive fallback");
}

// ---- the flight cannot overshoot and lands exactly on the pose
for (const ease of ["inout", "out", "linear"]) {
  const node = [-70.8, -13.4, -5.1];
  const camPos = [39.4, 5, 17.9];
  const camTgt = [0, 0, 0];
  const req = 30;
  const A = solveArrival(node, camPos, camTgt, req);
  const fromPos = camPos;
  const fromTgt = camTgt;
  const total = dist(fromPos, A.pos);
  const axis = [
    (A.pos[0] - fromPos[0]) / total,
    (A.pos[1] - fromPos[1]) / total,
    (A.pos[2] - fromPos[2]) / total,
  ];
  let maxAhead = -Infinity;
  let prevAhead = -Infinity;
  let backwards = false;
  let finalPos = null;
  let finalTgt = null;
  for (let i = 0; i <= 200; i++) {
    const e = applyEase(i / 200, ease);
    const p = lerp3(fromPos, A.pos, e);
    const t = lerp3(fromTgt, A.tgt, e);
    const ahead = (p[0] - fromPos[0]) * axis[0] + (p[1] - fromPos[1]) * axis[1] + (p[2] - fromPos[2]) * axis[2];
    if (ahead < prevAhead - 1e-9) backwards = true;
    prevAhead = ahead;
    maxAhead = Math.max(maxAhead, ahead);
    finalPos = p;
    finalTgt = t;
  }
  ok(!backwards, `${ease}: camera never reverses along its travel axis`);
  ok(maxAhead <= total + 1e-9, `${ease}: never travels past the arrival pose (no overshoot)`);
  ok(vClose(finalPos, A.pos), `${ease}: final camera === solved arrival`);
  ok(vClose(finalTgt, node), `${ease}: final target === node`);
  ok(close(dist(finalPos, node), req), `${ease}: final camera→node === requested distance`);
  // centered: look direction from the camera is exactly toward the node
  const look = [finalTgt[0] - finalPos[0], finalTgt[1] - finalPos[1], finalTgt[2] - finalPos[2]];
  const toNode = [node[0] - finalPos[0], node[1] - finalPos[1], node[2] - finalPos[2]];
  const par = Math.abs(
    (look[0] * toNode[0] + look[1] * toNode[1] + look[2] * toNode[2]) /
      (Math.hypot(...look) * Math.hypot(...toNode) || 1),
  );
  ok(close(par, 1, 1e-9), `${ease}: camera looks exactly at the node (dead-centre)`);
}

// ---- keyframe-sequence math -------------------------------------------------
// The journey stage camera is a sequence of explicit poses {pos, target}.
// These tests pin the guarantees: fitDistance frames the content, each key's
// arrival pose is exact, and the path between two consecutive keys can never
// leave the bounding sphere of their endpoints or overshoot the arrival.

// tiny deterministic PRNG so the synthetic graph is reproducible
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const scatter = (n, cx, cy, cz, r) =>
  Array.from({ length: n }, () => {
    const th = rnd() * Math.PI * 2;
    const ph = Math.acos(2 * rnd() - 1);
    const rr = r * (0.3 + 0.7 * rnd());
    return [cx + rr * Math.sin(ph) * Math.cos(th), cy + rr * Math.sin(ph) * Math.sin(th), cz + rr * Math.cos(ph)];
  });

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Project `p` through a pinhole camera at `pos` looking at `tgt` (up = +y). */
function projectNDC(pos, tgt, p, fovDeg, aspect) {
  const f = norm(sub(tgt, pos));
  let right = cross(f, [0, 1, 0]);
  if (Math.hypot(...right) < 1e-6) right = [1, 0, 0];
  right = norm(right);
  const up = cross(right, f);
  const v = sub(p, pos);
  const z = dot(v, f);
  const x = dot(v, right);
  const y = dot(v, up);
  const tanV = Math.tan((fovDeg * Math.PI) / 180 / 2);
  if (z <= 1e-6) return [Infinity, Infinity];
  return [x / (z * tanV * aspect), y / (z * tanV)];
}

// ---- fitDistance: monotone in radius, honours minDist
{
  const a = fitDistance(0, 55, 1.6, 0.15, 20);
  ok(close(a, 20), `fitDistance: radius 0 -> minDist (${a})`);
  let mono = true;
  let prev = -Infinity;
  for (let r = 0; r <= 200; r += 5) {
    const d = fitDistance(r, 55, 1.6, 0.15, 20);
    if (d < prev - 1e-9) mono = false;
    prev = d;
  }
  ok(mono, "fitDistance: monotone non-decreasing in radius");
  ok(Number.isFinite(fitDistance(NaN, NaN, NaN, 0.15, 6)), "fitDistance: bad inputs stay finite");
}

// ---- solveFocusPose: frames EVERY focus point inside the frustum
{
  const pts = scatter(40, 12, -6, 3, 60);
  const fov = 55;
  const aspect = 1.9;
  const pose = solveFocusPose(pts, fov, aspect, 0.16, 24);
  ok(vClose(pose.tgt, centroid3(pts)), "solveFocusPose: target === centroid of focus");
  ok(close(pose.dist, Math.hypot(...sub(pose.pos, pose.tgt))), "solveFocusPose: |pos-target| === dist");
  let worst = 0;
  for (const p of pts) {
    const ndc = projectNDC(pose.pos, pose.tgt, p, fov, aspect);
    worst = Math.max(worst, Math.abs(ndc[0]), Math.abs(ndc[1]));
  }
  ok(worst <= 1, `solveFocusPose: all focus points framed (worst |NDC| = ${worst.toFixed(3)} <= 1)`);
}

// ---- stageCamera: explicit poses solved from the stage's focus
const ctx = {
  query: [28, -13, -22],
  vecCluster: [29, -13, -23],
  seeds: [-61, 13, 3],
  bm25: [35, -9, -7],
  scene: [-42, 12, 6],
  fov: 55,
  aspect: 1.9,
  vecHits: scatter(12, 29, -13, -23, 40),
  seedPts: scatter(90, -61, 13, 3, 55),
  bm25Hits: scatter(12, 35, -9, -7, 28),
  fusionHits: scatter(14, 31, -14, -12, 24),
  ctxPts: scatter(20, 20, 0, 0, 35),
  scenePts: scatter(350, -42, 12, 6, 64),
};
const focusFor = (s) => {
  const wq = (ps) => (ctx.query ? [ctx.query, ...ps] : ps);
  switch (s) {
    case 0: return ctx.query ? [ctx.query] : [];
    case 1: return wq(ctx.vecHits);
    case 2: return wq(ctx.seedPts);
    case 3: return wq(ctx.bm25Hits);
    case 4: return wq(ctx.fusionHits);
    case 5: return wq(ctx.fusionHits);
    case 6: return wq(ctx.ctxPts);
    default: return wq(ctx.scenePts);
  }
};

const keys = [];
for (let s = 0; s < 8; s++) {
  const k = stageCamera(s, ctx);
  keys.push(k);
  const focus = focusFor(s);
  ok(k.pos.every(Number.isFinite) && k.target.every(Number.isFinite), `stage ${s}: finite pose`);
  ok(vClose(k.target, centroid3(focus), 1e-9), `stage ${s}: target === focus centroid`);
  ok(close(Math.hypot(...sub(k.pos, k.target)), k.dist, 1e-9), `stage ${s}: |pos-target| === dist`);
  ok(close(dist(k.pos, k.target), k.dist, 1e-9), `stage ${s}: honoured distance (${k.dist.toFixed(1)})`);
  let worst = 0;
  for (const p of focus) {
    const ndc = projectNDC(k.pos, k.target, p, ctx.fov, ctx.aspect);
    worst = Math.max(worst, Math.abs(ndc[0]), Math.abs(ndc[1]));
  }
  ok(worst <= 1, `stage ${s}: focus content framed (worst |NDC| = ${worst.toFixed(3)})`);
}

// ---- consecutive keys: monotone segment, no overshoot, inside bounding sphere
for (let s = 0; s < 7; s++) {
  const A = keys[s];
  const B = keys[s + 1];
  const total = dist(A.pos, B.pos);
  const axis = total > 1e-9 ? sub(B.pos, A.pos).map((v) => v / total) : [0, 0, 0];
  const mid = [(A.pos[0] + B.pos[0]) / 2, (A.pos[1] + B.pos[1]) / 2, (A.pos[2] + B.pos[2]) / 2];
  const R = total / 2;
  for (const ease of ["inout", "out", "linear"]) {
    let prevAhead = -Infinity;
    let backwards = false;
    let maxAhead = -Infinity;
    let maxFromMid = -Infinity;
    let finalPos = null;
    let finalTgt = null;
    for (let i = 0; i <= 200; i++) {
      const e = applyEase(i / 200, ease);
      const p = lerp3(A.pos, B.pos, e);
      const t = lerp3(A.target, B.target, e);
      const ahead = dot(sub(p, A.pos), axis);
      if (ahead < prevAhead - 1e-9) backwards = true;
      prevAhead = ahead;
      maxAhead = Math.max(maxAhead, ahead);
      maxFromMid = Math.max(maxFromMid, dist(p, mid));
      finalPos = p;
      finalTgt = t;
    }
    ok(!backwards, `key ${s}->${s + 1} ${ease}: monotone along travel axis`);
    ok(maxAhead <= total + 1e-9, `key ${s}->${s + 1} ${ease}: no overshoot past arrival`);
    ok(maxFromMid <= R + 1e-9, `key ${s}->${s + 1} ${ease}: path inside bounding sphere of endpoints`);
    ok(vClose(finalPos, B.pos), `key ${s}->${s + 1} ${ease}: lands exactly on next pose`);
    ok(vClose(finalTgt, B.target), `key ${s}->${s + 1} ${ease}: target lands exactly on next target`);
  }
}

// ---- legacy fallback (no fov/aspect): PresentationView path unchanged
{
  const legacyCtx = {
    query: [28, -13, -22],
    vecCluster: [29, -13, -23],
    seeds: [-61, 13, 3],
    bm25: [35, -9, -7],
    scene: [0, 0, 0],
  };
  for (let s = 0; s < 8; s++) {
    const k = stageCamera(s, legacyCtx);
    ok(k.pos.every(Number.isFinite) && k.target.every(Number.isFinite), `legacy stage ${s}: finite`);
    ok(k.dist > 0 && Number.isFinite(k.dist), `legacy stage ${s}: positive distance`);
  }
  const k0 = stageCamera(0, legacyCtx);
  ok(vClose(k0.target, legacyCtx.query), "legacy stage 0: target === query");
}

if (failures) {
  console.error(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log("\nflyto-pose: all green");
