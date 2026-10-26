// Per-stage camera keyframes for the 3D journey.
//
// Every stage is an explicit ARRIVAL POSE {pos, target} solved from the stage's
// focus content: `target` = centroid of the focus points (world coords),
// `pos` = target + CAM_DIR × a distance that FITS those points into the
// frustum. Playback flies pose→pose through the canonical camera tween (one
// eased parameter), so the path is a monotone straight segment that stops
// exactly on the arrival pose — it can never overshoot or leave the bounding
// sphere of two consecutive poses. Scrubbing jumps straight to the pose.
//
// The rich focus sets are optional: callers that only pass the legacy centroid
// fields (e.g. PresentationView) keep the tuned fixed distances.
import { centroid3, solveFocusPose } from "./camPose.ts";
import type { XYZ } from "./cosmos";

export interface CamKey {
  pos: XYZ;
  target: XYZ;
  /** honoured camera→target distance of the solved arrival pose */
  dist: number;
  /** bounding radius of the framed focus content (0 for a single point) */
  radius: number;
}

export interface StageCamCtx {
  query: XYZ | null;
  vecCluster: XYZ | null;
  seeds: XYZ | null;
  bm25: XYZ | null;
  scene: XYZ | null; // centroid of everything visited (pullback target)
  // ---- richer focus sets (world coords). When `fov` + `aspect` are supplied
  //      the distance is SOLVED to frame these; otherwise legacy distances.
  fov?: number;
  aspect?: number;
  vecHits?: XYZ[];
  seedPts?: XYZ[];
  bm25Hits?: XYZ[];
  fusionHits?: XYZ[];
  ctxPts?: XYZ[];
  scenePts?: XYZ[];
}

/** Legacy fixed distances (used when fov/aspect are not supplied). */
const LEGACY_D = [30, 44, 58, 46, 52, 40, 34, 118];
/** Per-stage framing padding when solving (answer stage pulls back a touch more). */
const PAD = [0.35, 0.16, 0.18, 0.18, 0.18, 0.18, 0.22, 0.24];
/** Per-stage minimum framing distance (single-point stages sit at this). */
const MIN_D = [26, 26, 30, 26, 26, 26, 24, 40];

function addAll(a: XYZ[] | undefined): XYZ[] {
  return a && a.length ? a : [];
}

/** The world-space focus content each stage frames. */
function focusFor(stage: number, ctx: StageCamCtx): XYZ[] {
  const withQ = (ps: XYZ[]): XYZ[] => (ctx.query ? [ctx.query, ...ps] : ps);
  switch (stage) {
    case 0:
      return ctx.query ? [ctx.query] : [];
    case 1:
      return withQ(addAll(ctx.vecHits));
    case 2:
      return withQ(addAll(ctx.seedPts));
    case 3:
      return withQ(addAll(ctx.bm25Hits));
    case 4:
      return withQ(addAll(ctx.fusionHits));
    case 5:
      return withQ(addAll(ctx.fusionHits));
    case 6:
      return withQ(addAll(ctx.ctxPts));
    default:
      return withQ(addAll(ctx.scenePts));
  }
}

export function stageCamera(stage: number, ctx: StageCamCtx): CamKey {
  const s = ((Math.floor(stage) % 8) + 8) % 8;
  const focus = focusFor(s, ctx);
  const target: XYZ = centroid3(focus) ?? ctx.query ?? ctx.scene ?? [0, 0, 0];

  // Solved path: explicit pose framing the stage's focus content.
  if (focus.length && ctx.fov && ctx.aspect) {
    const pose = solveFocusPose(focus, ctx.fov, ctx.aspect, PAD[s], MIN_D[s]);
    return { pos: pose.pos, target: pose.tgt, dist: pose.dist, radius: pose.radius };
  }

  // Legacy fallback: fixed offset (historic look) for callers that don't pass
  // fov/aspect. Kept byte-for-byte so PresentationView is unaffected.
  const d = LEGACY_D[s] ?? 118;
  const pos: XYZ = [target[0] + d * 0.18, target[1] + d * 0.42, target[2] + d * 0.94];
  return {
    pos,
    target,
    dist: Math.hypot(pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]),
    radius: 0,
  };
}
