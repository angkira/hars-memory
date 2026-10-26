// Shared renderer surface for Memory Travel.
//
// History: this used to host the GraphGPU (WebGPU) renderer. WebGPU is
// BLOCKED on this machine (Chrome/Dawn refuses the RADV Vulkan driver) and
// is flaky across Linux boxes in general, so the primary engine is now
// Three.js over WebGL2 (lib/cosmos3d.ts) with Canvas2D (lib/cosmos2d.ts) as
// the last-resort fallback. What survived here is everything BOTH renderers
// share: coordinate mapping, highlight palette, and the structural union
// every view consumes (AnyCosmos in cosmos2d.ts).

import type { CosmosGraph } from "./types";

export type RGBA = [number, number, number, number];
export type XYZ = [number, number, number];

/** UMAP [-1,1] -> world scale. Marker positions must use toWorld()/toWorld3(). */
export const WORLD_SCALE = 90;
/** UMAP z is flattened a little: galaxy disc reads better than a sphere. */
export const Z_FLATTEN = 0.55;

export function toWorld(xy: [number, number]): [number, number] {
  return [xy[0] * WORLD_SCALE, -xy[1] * WORLD_SCALE];
}

export function toWorld3(xyz: XYZ | null | undefined): XYZ {
  if (!xyz) return [0, 0, 0];
  return [xyz[0] * WORLD_SCALE, -xyz[1] * WORLD_SCALE, (xyz[2] ?? 0) * WORLD_SCALE * Z_FLATTEN];
}

/**
 * Highlight/semantic palette — deliberately vivid (journey stages, search
 * hits, selection). The BASE node colours live in palette.ts and are muted
 * deep tones; additive blending brightens them naturally.
 */
export const INK: Record<string, RGBA> = {
  base: [0.23, 0.3, 0.48, 1],
  dim: [0.13, 0.16, 0.28, 1],
  cyan: [0.0, 0.95, 1.0, 1],
  vector: [0.0, 0.95, 1.0, 1],
  gold: [1.0, 0.78, 0.25, 1],
  amber: [1.0, 0.72, 0.14, 1],
  bm25: [1.0, 0.72, 0.14, 1],
  overlap: [0.22, 1.0, 0.58, 1],
  magenta: [1.0, 0.35, 0.85, 1],
  violet: [0.75, 0.4, 1.0, 1],
  green: [0.35, 1.0, 0.55, 1],
  orange: [1.0, 0.55, 0.2, 1],
  white: [1.0, 1.0, 1.0, 1],
  query: [1.0, 0.28, 0.58, 1],
};

export interface SelectedNode {
  gpuId: number;
  graphIdx: number;
  name: string;
}

/**
 * The surface both renderers expose. Views (Cosmos/Travel) only touch this,
 * so everything works unchanged on either backend. Positions are 3D (XYZ);
 * the 2D fallback flattens to xy.
 */
export interface CosmosLike {
  readonly ready: boolean;
  readonly linksLoaded: number;
  init(canvas: HTMLCanvasElement): Promise<void>;
  /** Load the graph at layout positions (world coords via toWorld3). */
  load(
    graph: CosmosGraph,
    posByIdx: (XYZ | null)[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<void>;
  clear(): void;
  /** Legend toggles: hidden families draw near-invisible and stop answering picks. */
  setHiddenTypes(types: Set<string>): void;
  /** Ad-hoc glowing marker (query point, chunk hits). xy are world coords. */
  addMarker(label: string, x: number, y: number, size: number, color: RGBA): number;
  moveMarker(id: number, x: number, y: number, size?: number): void;
  removeMarker(id: number): void;
  /** Recolour/resize node subsets by graph index. Pass null to reset all. */
  paint(paints: Map<number, { color: RGBA; size?: number }> | null): void;
  /** Dim everything except the given indices (search spotlight). */
  spotlight(keep: Set<number> | null): void;
  /** Selection treatment: boost + ring on the star; null clears. */
  setSelection(idx: number | null): void;
  /** Project a node to CSS px (screen-space tooltip placement). */
  projectNode(idx: number): { x: number; y: number; visible: boolean } | null;
  /**
   * Fixed base edge opacity (journey/presentation). Calling this also takes
   * the renderer OUT of zoom-adaptive mode — the travel/presentation views
   * want a stable haze, the cosmos overview wants the distance-driven curve.
   */
  setEdgeOpacity(v: number): void;
  /**
   * Cosmos overview mode: edge opacity follows camera distance to the graph
   * (subtle haze when far, clearly readable lines up close). One uniform (3D)
   * or one idle-frame path repaint (2D) — never a geometry rebuild.
   */
  setEdgeZoomAdaptive(on: boolean): void;
  /** Master on/off for the whole edge layer (HUD toggle). Cheap: draw skip. */
  setEdgesEnabled(on: boolean): void;
  /**
   * Selection focus: edges incident to `idx` light up, its 1-hop neighbours'
   * edges at medium strength, everything else dims. null clears. Attribute
   * driven — one per-edge strength buffer, one draw call, no geometry rebuild.
   */
  setEdgeFocus(idx: number | null): void;
  /** Hover focus: same treatment, milder, while a star is hovered. */
  setEdgeHover(idx: number | null): void;
  /** Current effective base edge opacity (0..1) for the HUD intensity hint. */
  edgeIntensity(): number;
  /** Glowing billboard halos around candidate nodes. */
  setHalos(items: { idx: number; color?: RGBA; size?: number }[] | null): void;
  clearHalos(): void;
  /** Highlight and pulse a cluster of nodes with camera flight. */
  pulseCluster(nodeIndices: number[], color: RGBA, fly?: boolean): void;
  /** Eased camera flight. x/y world coords; zoom is renderer-relative. */
  flyTo(x: number, y: number, zoom: number, ms?: number): void;
  fit(padding?: number): void;
  currentZoom(): number;
  destroy(): void;
  onSelect: ((s: SelectedNode | null) => void) | null;
  onHover: ((s: SelectedNode | null) => void) | null;
}
