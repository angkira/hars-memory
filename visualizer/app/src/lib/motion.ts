// Motion preferences. CSS handles the dissolves/staggers; this covers the two
// JS-side timings that CSS cannot: the presentation's camera flights and the
// minimum time the streamed progress rows stay on screen.

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
