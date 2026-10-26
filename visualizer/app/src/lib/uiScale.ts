/**
 * UI scale — one knob for 4K monitors / projector legibility.
 *
 * MECHANISM: the CSS `zoom` property on `#root` (styled in ui-scale.css).
 *
 *   index.css is a px-only sheet (347 px values, no rem/em font sizes), so the
 *   usual `:root { font-size }` trick would scale nothing; `zoom` is the one
 *   property that scales a px layout AND its text, and because it scales each
 *   containing block it keeps every absolutely-positioned panel (topbar, query
 *   bar, sidebars, presentation chrome) anchored to the same visual viewport.
 *   The query bar's `width: 50%` still resolves to half the *visual* viewport.
 *
 * CANVAS CAVEAT (measured, Chromium 1187 / 141-class): both renderers size
 * their backing store from `canvas.clientWidth/clientHeight` (cosmos3d.ts
 * `resize()`, cosmos2d.ts `resize()`). Under an ancestor `zoom: Z` that reports
 * the *layout* size (visual / Z), so a zoomed canvas is a smaller buffer
 * stretched Z× — blurred at Z > 1. Measured: at Z = 1.45 a 1440 px-wide canvas
 * reported clientWidth 993 and produced a 993 px backing store over a 1440 px
 * visual area (1.45× soft). Neither renderer is ours to edit and faking
 * `window.devicePixelRatio` globally is not acceptable, so we compensate in
 * CSS: `.canvaswrap { zoom: 1 / Z }` (ui-scale.css) makes the *cumulative*
 * zoom of every canvas exactly 1. The canvas keeps its full-viewport layout
 * size, clientWidth is the true viewport width and the backing store is
 * pixel-exact (verified 1440x900 backing over a 1440x900 visual area at
 * Z = 1.45, i.e. crisp). Only the full-bleed canvas layers opt out of the
 * scaling; every DOM overlay around them still scales uniformly.
 */

/** localStorage key — an explicit choice here always wins over the auto default. */
export const UI_SCALE_KEY = "memory-travel.ui-scale";
export const UI_SCALE_MIN = 0.75;
export const UI_SCALE_MAX = 1.6;
export const UI_SCALE_STEP = 0.05;

/**
 * Auto default from the viewport width, largest match first.
 * Width (not DPR) is the right signal: a 4K panel at 100% OS scaling already
 * reports ~3840 innerWidth, while the same panel at 200% reports ~1920 CSS px
 * and is *already* comfortably sized by the OS — folding DPR in would over-
 * scale HiDPI laptops. DPR is logged for support, not used to pick a tier.
 */
const AUTO_TIERS: ReadonlyArray<readonly [number, number]> = [
  [3300, 1.45],
  [2500, 1.3],
  [2000, 1.15],
];

export function autoScaleForWidth(width: number): number {
  for (const [min, scale] of AUTO_TIERS) if (width >= min) return scale;
  return 1;
}

/** Auto default for the current viewport. */
export function autoScaleForViewport(): number {
  return autoScaleForWidth(typeof window === "undefined" ? 0 : window.innerWidth || 0);
}

/** Round to the 0.05 grid so range-input float dust (1.4500000000000002) can't leak. */
function snap(scale: number): number {
  return Math.round(Math.round(scale / UI_SCALE_STEP) * UI_SCALE_STEP * 100) / 100;
}

export function clampScale(scale: number): number {
  return snap(Math.min(UI_SCALE_MAX, Math.max(UI_SCALE_MIN, scale)));
}

/** Stored explicit choice, or null when absent/out of range/corrupt. */
export function readStoredScale(): number | null {
  try {
    const raw = localStorage.getItem(UI_SCALE_KEY);
    if (raw === null) return null;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < UI_SCALE_MIN || n > UI_SCALE_MAX) return null;
    return snap(n);
  } catch {
    return null; // private mode / disabled storage — fall back to auto
  }
}

function writeStoredScale(scale: number | null): void {
  try {
    if (scale === null) localStorage.removeItem(UI_SCALE_KEY);
    else localStorage.setItem(UI_SCALE_KEY, String(scale));
  } catch { /* private mode etc — the scale still applies for this session */ }
}

// ---- tiny store (single source of truth for every subscriber) --------------

type Listener = (scale: number, auto: boolean) => void;

let current = 1;
let auto = true;
const listeners = new Set<Listener>();

function notify(): void {
  for (const fn of listeners) fn(current, auto);
}

/**
 * Apply the scale to the live document. Writes the two custom properties
 * ui-scale.css reads: the scale itself and its reciprocal (CSS `calc()` cannot
 * divide by a custom property, so the inverse is computed here). Set on
 * documentElement so both `#root` and the counter-zoomed `.canvaswrap` inherit.
 */
function apply(scale: number): void {
  const el = document.documentElement;
  el.style.setProperty("--mt-ui-scale", String(scale));
  el.style.setProperty("--mt-ui-scale-inv", String(1 / scale));
  el.dataset.uiScale = String(scale);
}

export function getUiScale(): number {
  return current;
}

export function isAutoScale(): boolean {
  return auto;
}

export function subscribeUiScale(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Set an explicit scale (slider). Persists by default — an explicit choice wins over auto. */
export function setUiScale(scale: number, opts: { persist?: boolean } = {}): number {
  current = clampScale(scale);
  auto = false;
  if (opts.persist !== false) writeStoredScale(current);
  apply(current);
  notify();
  return current;
}

/** Drop the explicit choice and go back to the auto default for this viewport. */
export function resetUiScale(): number {
  writeStoredScale(null);
  auto = true;
  current = autoScaleForViewport();
  apply(current);
  notify();
  return current;
}

let resizeRaf = 0;

/** Re-pick the auto default when the window crosses a tier — only while in auto mode. */
function onViewportResize(): void {
  if (!auto) return; // an explicit choice is sticky across resizes
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    const next = autoScaleForViewport();
    if (next === current) return;
    current = next;
    apply(current);
    notify();
  });
}

/**
 * Boot step: resolve the scale (stored choice, else the viewport default) and
 * apply it BEFORE the first paint — main.tsx calls this synchronously before
 * `createRoot().render()`, so there is no unscaled flash. Logs the decision
 * once (support needs to know what a given screenshot's scale was).
 */
export function initUiScale(): number {
  if (!zoomSupported()) {
    // ui-scale.css is wrapped in @supports (zoom: 1): without element zoom the
    // app simply stays at 100% instead of getting a mis-sized canvas.
    console.warn("[memory-travel] CSS `zoom` unsupported — the UI scale control is inert on this engine");
  }
  const stored = readStoredScale();
  const useAuto = stored === null;
  auto = useAuto;
  current = useAuto ? autoScaleForViewport() : stored;
  apply(current);
  window.addEventListener("resize", onViewportResize);
  const w = window.innerWidth;
  const h = window.innerHeight;
  const dpr = window.devicePixelRatio || 1;
  console.info(
    `[memory-travel] UI scale ${Math.round(current * 100)}% (${auto ? "auto from viewport" : "saved preference"})` +
      ` — viewport ${w}x${h} css px @ dpr ${dpr} (${Math.round(w * dpr)}x${Math.round(h * dpr)} device px,` +
      ` screen ${window.screen?.width ?? "?"}x${window.screen?.height ?? "?"})`,
  );
  return current;
}

/** True when the engine understands element `zoom` (Chromium/Safari/Gecko 126+). */
export function zoomSupported(): boolean {
  return typeof CSS !== "undefined" && typeof CSS.supports === "function" && CSS.supports("zoom", "1");
}
