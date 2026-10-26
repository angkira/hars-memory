// Canvas2D fallback for Memory Travel — API-compatible with Cosmos3D.
//
// Auto-switch rule (see components/CosmosCanvas.tsx): the app tries the
// WebGL2/Three.js starfield first; when WebGL2 is missing or init fails, it
// constructs this class on the SAME canvas instead. Travellers of the
// Cosmos3D/Cosmos2D union only touch the shared surface below, so
// Travel/Cosmos views work unchanged on either path. Positions arrive as
// 3D (XYZ); the 2D plane simply drops z.
//
// Performance principles (this is what keeps 47k nodes interactive in 2D):
// - Static frames only: every state change sets a dirty flag; one rAF
//   coalesces redraws. No per-frame loop, no physics.
// - Nodes are batched by fill color (one fillStyle per group) and drawn as
//   1-2px rects at overview zoom — no per-node shadowBlur (the perf killer).
// - Full edge draw (54k segments, one stroked path, low alpha) happens on
//   idle static frames; while the user is actively panning/zooming and the
//   edge count is large, edges are skipped and repainted ~150ms after the
//   gesture ends.
// - DPR-aware backing store; pan/zoom/click-select with mouse reusing the
//   same world mapping as the 3D path (toWorld / WORLD_SCALE in cosmos.ts).
import { WORLD_SCALE, type CosmosLike, type RGBA, type SelectedNode, type XYZ } from "./cosmos";
import { applyEase } from "./camPose";
import { colorForType, familyOf } from "./palette";
import type { CosmosGraph } from "./types";

function css(c: RGBA): string {
  const [r, g, b, a] = c;
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;
}

interface Marker {
  id: number;
  label: string;
  x: number;
  y: number;
  size: number;
  color: RGBA;
}

const BG = "#03040c";
/** World-unit -> px factor for node rects. Tuned so base nodes are 1-2px at fit zoom. */
const NODE_K = 0.06;
/** Above this many edges, skip edges while a pan/zoom gesture is active. */
const EDGE_INTERACTION_CUTOFF = 5000;

// ---- zoom-adaptive edge opacity (2D parity with cosmos3d) ------------------
// Same intent as the 3D curve, keyed on the 2D camera scale (px per world
// unit) relative to the fit scale: overview (1×) stays the familiar 0.09 haze,
// a close-up (>= 8×) reaches 0.42 so incident edges read clearly. Log-smoothstep
// between — monotone, no pops. Focus multiplies these by the per-edge weight.
const EDGE2D_OVERVIEW = 0.09;
const EDGE2D_CLOSE = 0.42;
const EDGE2D_OVERVIEW_SCALE = 1;
const EDGE2D_CLOSE_SCALE = 8;
/** Unrelated edges keep this fraction of the base while a focus is active. */
const EDGE2D_DIM = 0.3;
/** Incident-edge boost (× base). */
const EDGE2D_BOOST = 1.15;

function edgeAlphaForScale(scale: number, fitScale: number): number {
  const s = scale / Math.max(1e-6, fitScale);
  if (s <= EDGE2D_OVERVIEW_SCALE) return EDGE2D_OVERVIEW;
  if (s >= EDGE2D_CLOSE_SCALE) return EDGE2D_CLOSE;
  const u = Math.log(s) / Math.log(EDGE2D_CLOSE_SCALE);
  return EDGE2D_OVERVIEW + (EDGE2D_CLOSE - EDGE2D_OVERVIEW) * u * u * (3 - 2 * u);
}

export class Cosmos2D implements CosmosLike {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private ro: ResizeObserver | null = null;
  private dpr = 1;

  // graph data (world coords)
  private n = 0;
  private wx = new Float32Array(0);
  private wy = new Float32Array(0);
  private baseSize = new Float32Array(0);
  private baseCss: string[] = [];
  private types: string[] = [];
  private edges: [number, number][] = [];
  private edgeAlpha = 0.09;
  private names: string[] = [];

  // edge focus/hover + zoom-adaptive opacity (2D parity with cosmos3d)
  private edgeAdaptive = false;
  private edgeEnabled = true;
  private fitScale = 5;
  private edgeFocusIdx = -1;
  private edgeHoverIdx = -1;
  private adj: number[][] = [];
  private edgeS = new Int32Array(0);
  private edgeT = new Int32Array(0);

  // highlight state
  private paints = new Map<number, { color: RGBA; size?: number }>();
  private paintsResetAll = true; // paint(null) was last called / default
  private spot: Set<number> | null = null;
  /** entity families switched off in the legend — drawn near-invisible, unpickable */
  private hidden: Set<string> = new Set();

  private markers = new Map<number, Marker>();
  private nextMarker = 1;
  private selIdx = -1;

  // camera: world center + px-per-world-unit
  private cx = 0;
  private cy = 0;
  private scale = 5;

  private dirty = false;
  private raf = 0;
  private interacting = false;
  private interactTimer = 0;
  private flyRaf = 0;

  onSelect: ((s: SelectedNode | null) => void) | null = null;
  onHover: ((s: SelectedNode | null) => void) | null = null;

  private hoverRaf = 0;
  private lastHoverT = 0;

  get ready(): boolean {
    return this.canvas !== null;
  }

  get linksLoaded(): number {
    return this.edges.length;
  }

  async init(canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) throw new Error("cosmos2d init failed at stage=canvas: no canvas element");
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("cosmos2d init failed at stage=context2d: 2D context unavailable");
    this.canvas = canvas;
    this.ctx = ctx;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.resize();
    this.ro = new ResizeObserver(() => {
      this.resize();
      this.requestDraw();
    });
    this.ro.observe(canvas);
    canvas.addEventListener("pointerdown", this.onDown);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    canvas.addEventListener("click", this.onClick);
    canvas.addEventListener("pointermove", this.onMove);
    canvas.addEventListener("pointerleave", this.onLeave);
    canvas.style.touchAction = "none";
    canvas.style.cursor = "grab";
    this.requestDraw();
  }

  /** Load the graph at 3D positions, flattened to the 2D plane (z dropped). */
  async load(
    graph: CosmosGraph,
    posByIdx: (XYZ | null)[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<void> {
    if (!this.canvas) throw new Error("cosmos2d not initialised");
    const n = graph.nodes.length;
    this.n = n;
    this.wx = new Float32Array(n);
    this.wy = new Float32Array(n);
    this.baseSize = new Float32Array(n);
    this.baseCss = new Array(n);
    this.types = new Array(n);
    this.names = graph.nodes.map((nd) => nd.id);
    for (let i = 0; i < n; i++) {
      const nd = graph.nodes[i];
      const p = posByIdx[i];
      const [x, y] = p
        ? [p[0], p[1]]
        : [(Math.random() - 0.5) * WORLD_SCALE, (Math.random() - 0.5) * WORLD_SCALE];
      this.wx[i] = x;
      this.wy[i] = y;
      const size = 3.2 + Math.min(9, Math.sqrt(nd.deg) * 1.6);
      this.baseSize[i] = size;
      this.baseCss[i] = colorForType(nd.t); // colour IS the entity family (legend parity)
      this.types[i] = nd.t;
      if (onProgress && i % 8000 === 0) {
        onProgress(i, n);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    onProgress?.(n, n);
    this.edges = graph.edges.filter(([s, t]) => s >= 0 && t >= 0 && s < n && t < n);
    // adjacency + endpoint arrays once, so focus changes are O(deg) not O(E)
    this.edgeS = new Int32Array(this.edges.length);
    this.edgeT = new Int32Array(this.edges.length);
    this.adj = Array.from({ length: n }, () => [] as number[]);
    this.edges.forEach(([s, t], k) => {
      this.edgeS[k] = s;
      this.edgeT[k] = t;
      this.adj[s].push(k);
      if (t !== s) this.adj[t].push(k);
    });
    this.edgeFocusIdx = -1;
    this.edgeHoverIdx = -1;
    this.paints.clear();
    this.paintsResetAll = true;
    this.spot = null;
    // debug/verification handle (mirrors cosmos3d): only a renderer with DATA
    // loaded claims it, so the hidden duplicate canvas can't shadow it
    (window as unknown as { __cosmos2d?: Cosmos2D }).__cosmos2d = this;
    this.fit(0.12);
  }

  clear(): void {
    this.n = 0;
    this.wx = new Float32Array(0);
    this.wy = new Float32Array(0);
    this.baseSize = new Float32Array(0);
    this.baseCss = [];
    this.types = [];
    this.edges = [];
    this.adj = [];
    this.edgeS = new Int32Array(0);
    this.edgeT = new Int32Array(0);
    this.names = [];
    this.paints.clear();
    this.paintsResetAll = true;
    this.spot = null;
    this.markers.clear();
    this.requestDraw();
  }

  /**
   * Legend toggles: families here draw near-invisible and stop answering picks.
   * Explicit highlights (paint/spotlight) still win over hiding.
   */
  setHiddenTypes(types: Set<string>): void {
    this.hidden = types;
    this.requestDraw();
  }

  addMarker(label: string, x: number, y: number, size: number, color: RGBA): number {
    const id = this.nextMarker++;
    this.markers.set(id, { id, label, x, y, size, color });
    this.requestDraw();
    return id;
  }

  moveMarker(id: number, x: number, y: number, size?: number): void {
    const m = this.markers.get(id);
    if (!m) return;
    m.x = x;
    m.y = y;
    if (size !== undefined) m.size = size;
    this.requestDraw();
  }

  removeMarker(id: number): void {
    if (this.markers.delete(id)) this.requestDraw();
  }

  paint(paints: Map<number, { color: RGBA; size?: number }> | null): void {
    if (paints === null) {
      this.paints.clear();
      this.paintsResetAll = true;
    } else {
      this.paintsResetAll = false;
      for (const [idx, p] of paints) this.paints.set(idx, p);
    }
    this.requestDraw();
  }

  spotlight(keep: Set<number> | null): void {
    this.spot = keep;
    this.requestDraw();
  }

  /** Selection in 2D: the star gets a halo + ring when next drawn. */
  setSelection(idx: number | null): void {
    this.selIdx = idx ?? -1;
    this.requestDraw();
  }

  /** Screen position of a node (CSS px within the canvas). */
  projectNode(idx: number): { x: number; y: number; visible: boolean } | null {
    if (idx < 0 || idx >= this.n) return null;
    const [sx, sy] = this.toScreen(this.wx[idx], this.wy[idx]);
    const w = this.canvas?.clientWidth || 1;
    return { x: sx, y: sy, visible: sx > -50 && sx < w + 50 };
  }

  setEdgeOpacity(v: number): void {
    // fixed base (journey/presentation) also cancels zoom-adaptive mode
    this.edgeAlpha = v;
    this.edgeAdaptive = false;
    this.requestDraw();
  }

  setEdgeZoomAdaptive(on: boolean): void {
    this.edgeAdaptive = on;
    this.requestDraw();
  }

  setEdgesEnabled(on: boolean): void {
    this.edgeEnabled = on;
    this.requestDraw();
  }

  setEdgeFocus(idx: number | null): void {
    const v = idx ?? -1;
    if (v === this.edgeFocusIdx) return;
    this.edgeFocusIdx = v;
    this.requestDraw();
  }

  setEdgeHover(idx: number | null): void {
    const v = idx ?? -1;
    if (v === this.edgeHoverIdx) return;
    this.edgeHoverIdx = v;
    this.requestDraw();
  }

  edgeIntensity(): number {
    if (!this.edgeEnabled) return 0;
    return this.edgeAdaptive ? edgeAlphaForScale(this.scale, this.fitScale) : this.edgeAlpha;
  }

  /** Edge index -> highlight weight (1 incident, 0.5 one hop) for the focused
   *  and hovered nodes; null when nothing is focused. Incident wins. */
  private edgeHighlights(): Map<number, number> | null {
    const foc = this.edgeFocusIdx;
    const hov = this.edgeHoverIdx;
    if (foc < 0 && hov < 0) return null;
    const w = new Map<number, number>();
    const add = (f: number) => {
      if (f < 0 || f >= this.n) return;
      const inc = this.adj[f] ?? [];
      for (const k of inc) if ((w.get(k) ?? 0) < 1) w.set(k, 1);
      for (const k of inc) {
        const other = this.edgeS[k] === f ? this.edgeT[k] : this.edgeS[k];
        if (other === f) continue;
        for (const k2 of this.adj[other]) if (!w.has(k2)) w.set(k2, 0.5);
      }
    };
    add(foc);
    add(hov);
    return w;
  }

  /** Eased camera flight (lerped center + zoom via rAF — no physics).
   *  The path is a straight interpolation of both endpoints with one eased
   *  0→1 parameter, so it cannot overshoot; it stops exactly on (x, y, zoom)
   *  with the node dead-centre. Any user gesture cancels it (see onDown /
   *  onWheel) so an in-flight camera can never fight the pointer. */
  flyTo(x: number, y: number, zoom: number, ms = 1400): void {
    cancelAnimationFrame(this.flyRaf);
    const fromX = this.cx;
    const fromY = this.cy;
    const fromZ = this.scale;
    const toZ = Math.max(0.05, zoom);
    if (ms <= 0) {
      this.cx = x;
      this.cy = y;
      this.scale = toZ;
      this.requestDraw();
      return;
    }
    const t0 = performance.now();
    const step = (t: number) => {
      const p = Math.min(1, (t - t0) / ms);
      const e = applyEase(p, "inout");
      this.cx = fromX + (x - fromX) * e;
      this.cy = fromY + (y - fromY) * e;
      this.scale = fromZ + (toZ - fromZ) * e;
      this.requestDraw();
      if (p < 1) this.flyRaf = requestAnimationFrame(step);
      else this.flyRaf = 0;
    };
    this.flyRaf = requestAnimationFrame(step);
  }

  /** A user gesture takes the camera away from any in-flight flight. */
  private stopFly(): void {
    cancelAnimationFrame(this.flyRaf);
    this.flyRaf = 0;
  }

  setHalos(_items: { idx: number; color?: RGBA; size?: number }[] | null): void {
    // 2D fallback handles halos via spotlight and paint
  }

  clearHalos(): void {}

  pulseCluster(indices: number[], color: RGBA, fly = true): void {
    if (!indices.length || !this.n) return;
    this.spotlight(new Set(indices));
    const paints = new Map<number, { color: RGBA; size?: number }>();
    indices.forEach((i) => paints.set(i, { color, size: 3 }));
    this.paint(paints);
    if (fly) {
      let sx = 0, sy = 0;
      let count = 0;
      for (const i of indices) {
        if (i >= 0 && i < this.n) {
          sx += this.wx[i];
          sy += this.wy[i];
          count++;
        }
      }
      if (count > 0) this.flyTo(sx / count, sy / count, this.fitScale * 2.5, 1000);
    }
  }

  fit(padding = 0.12): void {
    if (!this.canvas || this.n === 0) return;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < this.n; i++) {
      const x = this.wx[i];
      const y = this.wy[i];
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    const bw = Math.max(1e-6, x1 - x0);
    const bh = Math.max(1e-6, y1 - y0);
    this.scale = Math.min(w / bw, h / bh) * (1 - padding);
    this.cx = (x0 + x1) / 2;
    this.cy = (y0 + y1) / 2;
    this.fitScale = this.scale; // reference for the zoom-adaptive curve
    this.requestDraw();
  }

  currentZoom(): number {
    return this.scale;
  }

  destroy(): void {
    cancelAnimationFrame(this.raf);
    cancelAnimationFrame(this.flyRaf);
    cancelAnimationFrame(this.hoverRaf);
    window.clearTimeout(this.interactTimer);
    this.ro?.disconnect();
    this.ro = null;
    if (this.canvas) {
      this.canvas.removeEventListener("pointerdown", this.onDown);
      this.canvas.removeEventListener("wheel", this.onWheel);
      this.canvas.removeEventListener("click", this.onClick);
      this.canvas.removeEventListener("pointermove", this.onMove);
      this.canvas.removeEventListener("pointerleave", this.onLeave);
    }
    this.canvas = null;
    this.ctx = null;
  }

  // ---- coordinate mapping (screen <-> world) ----
  private toScreen(x: number, y: number): [number, number] {
    const c = this.canvas!;
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    return [(x - this.cx) * this.scale + w / 2, (y - this.cy) * this.scale + h / 2];
  }

  // ---- interaction ----
  private drag: { px: number; py: number; moved: number } | null = null;

  private onDown = (e: PointerEvent) => {
    this.stopFly(); // a user grab always wins over an in-flight fly-to
    this.drag = { px: e.clientX, py: e.clientY, moved: 0 };
    this.canvas?.setPointerCapture(e.pointerId);
    if (this.canvas) this.canvas.style.cursor = "grabbing";
    const up = (ev: PointerEvent) => {
      this.drag = null;
      if (this.canvas) this.canvas.style.cursor = "grab";
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      this.endInteractSoon();
      void ev;
    };
    const move = (ev: PointerEvent) => {
      if (!this.drag) return;
      const dx = ev.clientX - this.drag.px;
      const dy = ev.clientY - this.drag.py;
      this.drag.px = ev.clientX;
      this.drag.py = ev.clientY;
      this.drag.moved += Math.abs(dx) + Math.abs(dy);
      this.cx -= dx / this.scale;
      this.cy -= dy / this.scale;
      this.beginInteract();
      this.requestDraw();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.stopFly(); // user input cancels an in-flight fly-to
    const c = this.canvas;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const next = Math.min(4000, Math.max(0.05, this.scale * factor));
    // zoom to cursor: keep the world point under the cursor fixed
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    const wx = this.cx + (mx - w / 2) / this.scale;
    const wy = this.cy + (my - h / 2) / this.scale;
    this.scale = next;
    this.cx = wx - (mx - w / 2) / next;
    this.cy = wy - (my - h / 2) / next;
    this.beginInteract();
    this.requestDraw();
  };

  private onClick = (e: MouseEvent) => {
    if (this.drag && this.drag.moved > 6) return; // was a drag, not a click
    const idx = this.pick(e.clientX, e.clientY);
    if (!this.onSelect) return;
    this.onSelect(idx === null ? null : { gpuId: idx, graphIdx: idx, name: this.names[idx] ?? `#${idx}` });
  };

  private onMove = (e: PointerEvent) => {
    if (this.drag || (!this.onHover && !this.edgeAdaptive) || this.hoverRaf) return;
    const cx = e.clientX;
    const cy = e.clientY;
    this.hoverRaf = requestAnimationFrame(() => {
      this.hoverRaf = 0;
      const idx = this.pick(cx, cy);
      this.onHover?.(
        idx === null ? null : { gpuId: idx, graphIdx: idx, name: this.names[idx] ?? `#${idx}` },
      );
      // reuse the SAME pick for the edge highlight, throttled to >= 30 ms
      if (this.edgeAdaptive && idx !== this.edgeHoverIdx && performance.now() - this.lastHoverT >= 30) {
        this.lastHoverT = performance.now();
        this.setEdgeHover(idx);
      }
    });
  };

  private onLeave = () => {
    if (this.onHover) this.onHover(null);
    if (this.edgeAdaptive) this.setEdgeHover(null);
  };

  /** Nearest node within 12px of the cursor (47k scan per click is <5ms). */
  private pick(clientX: number, clientY: number): number | null {
    const c = this.canvas;
    if (!c || this.n === 0) return null;
    const rect = c.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    const R = 12;
    let best = -1;
    let bestD = R * R;
    for (let i = 0; i < this.n; i++) {
      if (this.hidden.has(familyOf(this.types[i]))) continue; // legend-hidden: unpickable
      const [sx, sy] = this.toScreen(this.wx[i], this.wy[i]);
      const dx = sx - mx;
      const dy = sy - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best >= 0 ? best : null;
  }

  private beginInteract(): void {
    this.interacting = true;
    window.clearTimeout(this.interactTimer);
  }

  private endInteractSoon(): void {
    window.clearTimeout(this.interactTimer);
    this.interactTimer = window.setTimeout(() => {
      this.interacting = false;
      this.requestDraw(); // repaint with edges now the gesture ended
    }, 150);
  }

  // ---- drawing ----
  private resize(): void {
    const c = this.canvas;
    if (!c) return;
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    c.width = Math.round(w * this.dpr);
    c.height = Math.round(h * this.dpr);
  }
  private requestDraw(): void {
    if (!this.canvas || this.dirty) return;
    this.dirty = true;
    this.raf = requestAnimationFrame(() => {
      this.dirty = false;
      this.draw();
    });
  }

  private draw(): void {
    const c = this.canvas;
    const ctx = this.ctx;
    if (!c || !ctx) return;
    const w = c.clientWidth || 1;
    const h = c.clientHeight || 1;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    if (this.n === 0) return;

    const base = this.edgeEnabled
      ? (this.edgeAdaptive ? edgeAlphaForScale(this.scale, this.fitScale) : this.edgeAlpha)
      : 0;
    const highlights = this.edgeHighlights();
    const showEdges = base > 0.001 && !(this.interacting && this.edges.length > EDGE_INTERACTION_CUTOFF);
    if (showEdges && this.edges.length > 0) {
      // pass 1: every edge at the (possibly dimmed) base alpha — ONE stroked
      // path for all 54k segments; the raster is only repainted on idle static
      // frames, so a pan/zoom gesture never re-strokes them (see draw()).
      const dim = highlights ? EDGE2D_DIM : 1;
      ctx.strokeStyle = `rgba(120,150,255,${(base * dim).toFixed(4)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k < this.edges.length; k++) {
        const [s, t] = this.edges[k];
        const ax = (this.wx[s] - this.cx) * this.scale + w / 2;
        const ay = (this.wy[s] - this.cy) * this.scale + h / 2;
        // cheap viewport cull on the segment start (ends may still cross in)
        if (ax < -50 || ay < -50 || ax > w + 50 || ay > h + 50) {
          const bx = (this.wx[t] - this.cx) * this.scale + w / 2;
          const by = (this.wy[t] - this.cy) * this.scale + h / 2;
          if (bx < -50 || by < -50 || bx > w + 50 || by > h + 50) continue;
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
        } else {
          ctx.moveTo(ax, ay);
          ctx.lineTo((this.wx[t] - this.cx) * this.scale + w / 2, (this.wy[t] - this.cy) * this.scale + h / 2);
        }
      }
      ctx.stroke();
      // pass 2: focused/hovered subset over-stroked brighter (incident bright,
      // 1-hop medium). Small subset — never the full 54k during a gesture.
      if (highlights) {
        const incident: number[] = [];
        const hop: number[] = [];
        for (const [k, wt] of highlights) (wt >= 1 ? incident : hop).push(k);
        const stroke = (ids: number[], alpha: number) => {
          if (!ids.length) return;
          ctx.strokeStyle = `rgba(188,210,255,${Math.min(0.92, alpha).toFixed(4)})`;
          ctx.beginPath();
          for (const k of ids) {
            const s = this.edgeS[k];
            const t = this.edgeT[k];
            ctx.moveTo((this.wx[s] - this.cx) * this.scale + w / 2, (this.wy[s] - this.cy) * this.scale + h / 2);
            ctx.lineTo((this.wx[t] - this.cx) * this.scale + w / 2, (this.wy[t] - this.cy) * this.scale + h / 2);
          }
          ctx.stroke();
        };
        stroke(hop, base * 0.7);
        stroke(incident, base * EDGE2D_BOOST);
      }
    }

    // Nodes, batched by fill color: one fillStyle per group, fillRect each.
    const groups = new Map<string, number[]>();
    const dim = "rgba(18,23,42,1)";
    const ghost = "rgba(30,36,58,0.18)"; // legend-hidden family: barely-there ghost
    const push = (color: string, i: number) => {
      const g = groups.get(color);
      if (g) g.push(i);
      else groups.set(color, [i]);
    };
    if (this.paintsResetAll && this.spot === null) {
      for (let i = 0; i < this.n; i++) {
        push(this.hidden.has(familyOf(this.types[i])) ? ghost : this.baseCss[i], i);
      }
    } else {
      for (let i = 0; i < this.n; i++) {
        if (this.spot && !this.spot.has(i)) {
          push(dim, i);
          continue;
        }
        const p = this.paints.get(i);
        push(p ? css(p.color) : this.hidden.has(familyOf(this.types[i])) ? ghost : this.baseCss[i], i);
      }
    }
    for (const [color, idxs] of groups) {
      ctx.fillStyle = color;
      for (let k = 0; k < idxs.length; k++) {
        const i = idxs[k];
        const p = this.paintsResetAll ? undefined : this.paints.get(i);
        const want = p?.size ?? this.baseSize[i];
        let s = want * this.scale * NODE_K;
        if (s < 1) s = 1;
        else if (s > 26) s = 26;
        const x = (this.wx[i] - this.cx) * this.scale + w / 2;
        const y = (this.wy[i] - this.cy) * this.scale + h / 2;
        if (x < -30 || y < -30 || x > w + 30 || y > h + 30) continue;
        ctx.fillRect(x - s / 2, y - s / 2, s, s);
      }
    }

    // selection ring: thin animated halo around the selected star
    if (this.selIdx >= 0 && this.selIdx < this.n) {
      const x = (this.wx[this.selIdx] - this.cx) * this.scale + w / 2;
      const y = (this.wy[this.selIdx] - this.cy) * this.scale + h / 2;
      const t = (performance.now() % 2400) / 2400;
      const r = 10 + 2 * Math.sin((performance.now() / 1000) * 2.6);
      ctx.save();
      ctx.strokeStyle = "rgba(125, 243, 255, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 5]);
      ctx.lineDashOffset = -t * 44;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Markers (query point, chunk hits): glowing-ish dot via layered circles + label.
    for (const m of this.markers.values()) {
      const [sx, sy] = this.toScreen(m.x, m.y);
      if (sx < -60 || sy < -60 || sx > w + 60 || sy > h + 60) continue;
      let r = m.size * this.scale * NODE_K;
      if (r < 4) r = 4;
      else if (r > 30) r = 30;
      const col = css(m.color);
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 1.9, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 0.8, 0, Math.PI * 2);
      ctx.fill();
      // label on a dark pill so it stays crisp over the starfield
      if (!m.label.trim()) continue;
      ctx.font = `${Math.max(10, Math.min(14, r + 4))}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const tw = ctx.measureText(m.label).width;
      const ly = sy - r - 10;
      ctx.fillStyle = "rgba(3,4,12,0.78)";
      ctx.beginPath();
      ctx.roundRect(sx - tw / 2 - 5, ly - 8, tw + 10, 16, 8);
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#eef4ff";
      ctx.fillText(m.label, sx, ly);
    }
  }
}

/** Either renderer. Structural union — both expose the same shared surface. */
export type AnyCosmos = import("./cosmos3d").Cosmos3D | Cosmos2D;
