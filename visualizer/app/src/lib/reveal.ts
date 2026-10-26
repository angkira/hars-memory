// Minimal rAF timeline for the Travel stage reveals.
//
// A reveal plan schedules `at(t, fn)` cues and `tween(t, dur, fn)` size/
// position animations; start() plays them on one rAF loop. `speed` divides
// every delay (playback-rate control); `instant` fires everything in order
// with tweens jumped to their end — that's what scrubbing uses.

export type TweenEase = "out" | "inout" | "linear";

export interface Timeline {
  /** Fire fn once, `delayMs` from timeline start (pre-speed). */
  at(delayMs: number, fn: () => void): void;
  /** Run fn(easedT) each frame from delayMs to delayMs+durMs (pre-speed). */
  tween(delayMs: number, durMs: number, fn: (t: number) => void, ease?: TweenEase): void;
  start(): void;
  cancel(): void;
  /** Total planned duration (post-speed), for scheduling hold times. */
  durationMs(): number;
}

export function easeout(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function easeinout(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

interface Cue {
  at: number; // post-speed ms
  fn: () => void;
  done: boolean;
}

interface Tween {
  from: number;
  dur: number;
  fn: (t: number) => void;
  ease: TweenEase;
  done: boolean;
}

export function makeTimeline(speed: number, instant = false): Timeline {
  const cues: Cue[] = [];
  const tweens: Tween[] = [];
  let raf = 0;
  let t0 = 0;
  let planned = 0;

  const applyEase = (t: number, e: TweenEase) =>
    e === "out" ? easeout(t) : e === "inout" ? easeinout(t) : t;

  return {
    at(delayMs, fn) {
      const at = Math.max(0, delayMs) / speed;
      cues.push({ at, fn, done: false });
      planned = Math.max(planned, at);
    },
    tween(delayMs, durMs, fn, ease = "out") {
      const from = Math.max(0, delayMs) / speed;
      const dur = Math.max(1, durMs) / speed;
      tweens.push({ from, dur, fn, ease, done: false });
      planned = Math.max(planned, from + dur);
    },
    start() {
      if (instant) {
        // ordered jump-to-end: cues then tweens in plan order
        cues.sort((a, b) => a.at - b.at);
        for (const c of cues) {
          c.done = true;
          c.fn();
        }
        for (const tw of tweens) {
          tw.done = true;
          tw.fn(1);
        }
        return;
      }
      cancelAnimationFrame(raf);
      t0 = performance.now();
      const tick = (now: number) => {
        const t = now - t0;
        for (const c of cues) {
          if (!c.done && t >= c.at) {
            c.done = true;
            c.fn();
          }
        }
        for (const tw of tweens) {
          if (tw.done) continue;
          if (t >= tw.from + tw.dur) {
            tw.done = true;
            tw.fn(1);
          } else if (t >= tw.from) {
            tw.fn(applyEase((t - tw.from) / tw.dur, tw.ease));
          }
        }
        if (cues.some((c) => !c.done) || tweens.some((w) => !w.done)) {
          raf = requestAnimationFrame(tick);
        }
      };
      raf = requestAnimationFrame(tick);
    },
    cancel() {
      cancelAnimationFrame(raf);
      raf = 0;
    },
    durationMs() {
      return planned;
    },
  };
}
