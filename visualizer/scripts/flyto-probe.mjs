// Fly-to camera probe (Memory Travel).
//
// Measures, in a real headless browser, where the camera ends up after every
// code path that moves it toward a node, and whether the node is framed:
//
//   search-Enter · node-card "fly to" · canvas node click ·
//   travel journey-log row · travel journey-mode stage keyframes ·
//   presentation chapter flies
//
// For each run it reports: overshoot past the solved arrival pose, closest
// approach to the node, final distance to the node, final controls.target
// offset from the node, and the node's final screen position (NDC ≈ 0,0 means
// centred). Analysis is done against the node the app ACTUALLY selected
// (read from the node card), so a wrong-node pick can't masquerade as a
// camera bug.
//
// Real-time waits only — NEVER --virtual-time-budget (it zeroes canvases and
// starves rAF; see docs/renderer.md).
//
// Usage:
//   node scripts/flyto-probe.mjs [--url=http://127.0.0.1:5188] [--out=file.json]
//                                [--json] [--2d] [--scenarios=search,card,click,travel,pres]
//   PW_ROOT=/path/to/dir-with-node_modules/playwright   (autodiscovered otherwise)
//   --2d forces the Canvas2D fallback (getContext('webgl2') is nulled) and
//   reports the node's on-screen offset via the selection tooltip.
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DATA = join(ROOT, "app", "public", "data");
const args = new Map(process.argv.slice(2).map((a) => {
  const m = /^--([^=]+)(?:=(.*))?$/.exec(a);
  return m ? [m[1], m[2] ?? "true"] : [a, "true"];
}));
const URL_ = args.get("url") ?? "http://127.0.0.1:5188";
const OUT = args.get("out") ?? "";
const AS_JSON = args.has("json");
const FORCE_2D = args.has("2d");
const ONLY = args.get("scenarios") ? new Set(args.get("scenarios").split(",")) : null;
const DWELL = Number(args.get("dwell") ?? 1600) || 1600;
const want = (s) => !ONLY || ONLY.has(s);

function loadPlaywright() {
  const candidates = [
    process.env.PW_ROOT && join(process.env.PW_ROOT, "package.json"),
    "/home/angkira/Project/knotgate/node_modules/playwright/package.json",
    "/home/angkira/Project/software/cortex/gui/frontend/node_modules/playwright/package.json",
  ].filter(Boolean);
  for (const c of candidates) {
    try { return createRequire(c)("playwright"); } catch { /* next */ }
  }
  throw new Error("playwright not found; set PW_ROOT");
}

// ---- unambiguous targets: the search must select exactly this node, so the
// node's name may not be a prefix of any other name (Enter picks the first
// startsWith match).
const graph = JSON.parse(readFileSync(join(DATA, "graph.json"), "utf8"));
const names = JSON.parse(readFileSync(join(DATA, "node_index.json"), "utf8"));
const degSorted = graph.nodes.map((n, i) => ({ i, id: n.id, deg: n.deg })).sort((a, b) => b.deg - a.deg);
const isPrefixOfOther = (nm) => names.some((x) => x !== nm && x.toLowerCase().startsWith(nm.toLowerCase()));
const uniq = (rec) => !isPrefixOfOther(rec.id);
const picked = [];
const pick = (from, minDeg, maxDeg) => {
  const hit = from.find((d) => d.deg >= minDeg && d.deg <= maxDeg && uniq(d) && !picked.includes(d.i));
  if (hit) picked.push(hit.i);
  return hit;
};
const TARGETS = [
  { label: "hub", ...pick(degSorted, 30, Infinity) },
  { label: "mid", ...pick(degSorted, 4, 29) },
  { label: "leaf", ...pick(degSorted, 1, 1) },
].filter((t) => t && t.i !== undefined);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- in-page instrumentation (installed before the app boots)
const INSTRUMENT = () => {
  window.__probe = {
    samples: [], sampling: false, node: [0, 0, 0], cam: null,
    projNDC(cam, x, y, z) {
      const e = cam.matrixWorldInverse.elements, p = cam.projectionMatrix.elements;
      const vx = e[0] * x + e[4] * y + e[8] * z + e[12];
      const vy = e[1] * x + e[5] * y + e[9] * z + e[13];
      const vz = e[2] * x + e[6] * y + e[10] * z + e[14];
      const vw = e[3] * x + e[7] * y + e[11] * z + e[15];
      const cx = p[0] * vx + p[4] * vy + p[8] * vz + p[12] * vw;
      const cy = p[1] * vx + p[5] * vy + p[9] * vz + p[13] * vw;
      const cw = p[3] * vx + p[7] * vy + p[11] * vz + p[15] * vw;
      return [cx / cw, cy / cw];
    },
  };
  const loop = () => {
    const P = window.__probe;
    if (P.sampling) {
      const c = P.cam || window.__cosmos3d;
      if (c && c.camera && c.controls) {
        const cam = c.camera;
        cam.updateMatrixWorld();
        cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
        const [nx, ny, nz] = P.node;
        const px = cam.position.x, py = cam.position.y, pz = cam.position.z;
        const tx = c.controls.target.x, ty = c.controls.target.y, tz = c.controls.target.z;
        P.samples.push({
          t: performance.now(),
          pos: [px, py, pz], tgt: [tx, ty, tz],
          ndc: P.projNDC(cam, nx, ny, nz),
        });
      } else {
        // Canvas2D fallback: the camera IS the world centre + px-per-unit scale
        const d = window.__cosmos2d;
        if (d && typeof d.cx === "number") {
          const [nx, ny] = P.node;
          const w = d.canvas?.clientWidth || 1, h = d.canvas?.clientHeight || 1;
          const sx = (nx - d.cx) * d.scale + w / 2;
          const sy = (ny - d.cy) * d.scale + h / 2;
          P.samples.push({
            t: performance.now(),
            pos: [d.cx, d.cy, 0], tgt: [d.cx, d.cy, 0],
            ndc: [(sx / w) * 2 - 1, (sy / h) * 2 - 1],
            scale: d.scale,
          });
        }
      }
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
};

const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * Camera quality of a recorded trajectory against the node the app selected.
 *  overshoot      how far past the final camera pose the camera travelled
 *                 along its own travel axis (0 = it never overshot)
 *  minDist        closest the camera ever got to the node
 *  finalDist      camera→node distance at rest (the requested dist, e.g. 28)
 *  tgtOff         |controls.target − node| at rest (0 = orbiting the node)
 *  finalNDC       node's screen position at rest (0,0 = dead centre)
 */
function analyze(samples, node) {
  if (!samples.length) return { frames: 0 };
  const s0 = samples[0], sN = samples[samples.length - 1];
  const u = [sN.pos[0] - s0.pos[0], sN.pos[1] - s0.pos[1], sN.pos[2] - s0.pos[2]];
  const uLen = Math.hypot(...u) || 1e-9;
  const uH = u.map((v) => v / uLen);
  let maxAhead = -Infinity, minDist = Infinity, maxAbsNDC = 0;
  for (const s of samples) {
    maxAhead = Math.max(maxAhead, (s.pos[0] - s0.pos[0]) * uH[0] + (s.pos[1] - s0.pos[1]) * uH[1] + (s.pos[2] - s0.pos[2]) * uH[2]);
    minDist = Math.min(minDist, dist3(s.pos, node));
    maxAbsNDC = Math.max(maxAbsNDC, Math.hypot(s.ndc[0], s.ndc[1]));
  }
  return {
    frames: samples.length,
    durationMs: Math.round(sN.t - s0.t),
    travelLen: uLen,
    overshoot: Math.max(0, maxAhead - uLen),
    minDist,
    finalDist: dist3(sN.pos, node),
    tgtOff: dist3(sN.tgt, node),
    finalNDC: sN.ndc,
    finalNDCabs: Math.hypot(sN.ndc[0], sN.ndc[1]),
    maxAbsNDC,
    startDist: dist3(s0.pos, node),
  };
}

async function main() {
  const pw = loadPlaywright();
  const browser = await pw.chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const logs = [];
  page.on("console", (m) => { if (m.type() === "error") logs.push(m.text()); });
  page.on("pageerror", (e) => logs.push(`PAGEERROR ${e.message}`));
  await page.addInitScript(INSTRUMENT);
  if (FORCE_2D) {
    await page.addInitScript(() => {
      const orig = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
        if (type === "webgl2" || type === "webgl" || type === "experimental-webgl") return null;
        return orig.call(this, type, ...rest);
      };
    });
  }

  const results = {
    url: URL_, mode: FORCE_2D ? "canvas2d (forced)" : "auto",
    targets: TARGETS.map((t) => ({ label: t.label, name: t.id, idx: t.i, deg: t.deg })),
    scenarios: {}, logs,
  };

  await page.goto(URL_, { waitUntil: "domcontentloaded" });
  if (FORCE_2D) {
    await page.waitForSelector(".modebadge", { timeout: 60000 });
    await page.waitForFunction(() => !document.querySelector(".panel.left .loading"), null, { timeout: 60000 }).catch(() => {});
  } else {
    await page.waitForFunction(() => { const c = window.__cosmos3d; return !!(c && c.n > 0 && document.querySelector(".hud")); }, null, { timeout: 120000 });
    await page.waitForFunction(() => !document.querySelector(".panel.left .loading"), null, { timeout: 120000 }).catch(() => {});
  }
  await sleep(1500);
  const is3d = !FORCE_2D && await page.evaluate(() => !!window.__cosmos3d?.camera);
  results.is3d = is3d;

  // ---- page helpers
  const startSample = (world) => page.evaluate((w) => { const P = window.__probe; P.samples = []; P.node = w; P.sampling = true; }, world);
  const stopSample = () => page.evaluate(() => { const P = window.__probe; P.sampling = false; return P.samples; });
  const cardTitle = () => page.evaluate(() => document.querySelector(".nodecard h3")?.textContent ?? null);
  const worldOfName = (nm) => page.evaluate((n) => {
    const c = window.__probe.cam || window.__cosmos3d;
    if (c && c.names && c.pos) {
      const i = c.names.indexOf(n); if (i < 0) return null;
      return { idx: i, world: [c.pos[i * 3], c.pos[i * 3 + 1], c.pos[i * 3 + 2]] };
    }
    const d = window.__cosmos2d;
    if (d && d.names) {
      const i = d.names.indexOf(n); if (i < 0) return null;
      return { idx: i, world: [d.wx[i], d.wy[i], 0] };
    }
    return null;
  }, nm);
  const projectNDC = async (idx) => {
    const r = await page.evaluate((i) => {
      const c = window.__probe.cam || window.__cosmos3d;
      const canvas = c?.canvas ?? document.querySelector(".canvaswrap:not(.hidden) canvas");
      const p = c?.projectNode ? c.projectNode(i) : null;
      const w = canvas?.clientWidth || 1, h = canvas?.clientHeight || 1;
      return p ? { x: p.x, y: p.y, w, h } : null;
    }, idx);
    return r ? [(r.x / r.w) * 2 - 1, (r.y / r.h) * 2 - 1] : null;
  };
  const probe2DState = () => page.evaluate(() => {
    const d = window.__cosmos2d;
    if (!d) return null;
    // nearest node to the current world centre = the node the camera flew to
    let best = -1, bestD = Infinity;
    for (let i = 0; i < d.n; i++) {
      const dd = (d.wx[i] - d.cx) ** 2 + (d.wy[i] - d.cy) ** 2;
      if (dd < bestD) { bestD = dd; best = i; }
    }
    return { idx: best, name: d.names[best], dist: Math.sqrt(bestD), scale: d.scale, cx: d.cx, cy: d.cy, w: d.canvas?.clientWidth || 1, h: d.canvas?.clientHeight || 1 };
  });
  const canvasBox = () => page.evaluate(() => {
    const c = document.querySelector(".canvaswrap:not(.hidden) canvas");
    if (!c) return null;
    const r = c.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });

  /** run a scenario: reset, start sampling, trigger, settle, resolve the node
   *  the app selected, measure. `trigger` returns nothing; `expectName` is the
   *  intended node name (used for the sampling NDC). */
  async function run(trigger, expectName, pre) {
    if (pre) await pre();
    const w = expectName ? (await worldOfName(expectName))?.world ?? [0, 0, 0] : [0, 0, 0];
    await startSample(w);
    await trigger();
    await sleep(1500);
    const s = await stopSample();
    const selTitle = await cardTitle();
    const resolved = selTitle ? await worldOfName(selTitle) : null;
    const node = resolved?.world ?? w;
    const a = analyze(s, node);
    a.selected = selTitle;
    a.matched = !expectName || selTitle === expectName;
    if (is3d && resolved) {
      const ndc = await projectNDC(resolved.idx);
      if (ndc) { a.finalNDC = ndc; a.finalNDCabs = Math.hypot(ndc[0], ndc[1]); a.finalNDCsrc = "projectNode"; }
    }
    if (!is3d) {
      // 2D: the node the camera actually centred is the nearest to the world
      // centre; its screen offset from the canvas centre is the centring check
      const st = await probe2DState();
      if (st) {
        a.selected2D = st.name;
        a.centreOffWorld = st.dist;
        a.finalScale = st.scale;
        const d = await page.evaluate((i) => {
          const d2 = window.__cosmos2d;
          const w = d2.canvas.clientWidth, h = d2.canvas.clientHeight;
          const sx = (d2.wx[i] - d2.cx) * d2.scale + w / 2;
          const sy = (d2.wy[i] - d2.cy) * d2.scale + h / 2;
          return [(sx / w) * 2 - 1, (sy / h) * 2 - 1];
        }, st.idx);
        a.finalNDC = d;
        a.finalNDCabs = Math.hypot(d[0], d[1]);
        a.finalNDCsrc = "cosmos2d";
      }
    }
    return a;
  }

  const resetFit = () => page.evaluate(() => {
    const c = window.__probe.cam || window.__cosmos3d;
    if (c) { c.fit(0.12); return; }
    window.__cosmos2d?.fit(0.12);
  });
  const search = async (nm) => {
    await page.fill(".searchwrap input.search", nm);
    await page.waitForSelector(".results button", { timeout: 5000 }).catch(() => {});
    await sleep(150);
    await page.press(".searchwrap input.search", "Enter");
  };
  // real user gestures that leave OrbitControls damping residuals behind
  const orbitDrag = async () => {
    const b = await canvasBox();
    const cx = b.x + b.w * 0.62, cy = b.y + b.h * 0.5;
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(cx - i * 22, cy + i * 9);
    await page.mouse.up();
  };
  const wheelZoom = async (dy) => {
    const b = await canvasBox();
    await page.mouse.move(b.x + b.w * 0.6, b.y + b.h * 0.5);
    await page.mouse.wheel(0, dy);
  };
  const rightDrag = async () => {
    const b = await canvasBox();
    const cx = b.x + b.w * 0.62, cy = b.y + b.h * 0.5;
    await page.mouse.move(cx, cy);
    await page.mouse.down({ button: "right" });
    for (let i = 1; i <= 6; i++) await page.mouse.move(cx - i * 18, cy - i * 6);
    await page.mouse.up({ button: "right" });
  };
  const findClickableNode = async (idx) => {
    // return node index whose projection sits in a panel-free, canvas area
    return page.evaluate((i) => {
      const c = window.__cosmos3d;
      const p = c.projectNode(i);
      if (!p) return null;
      const el = document.elementFromPoint(p.x, p.y);
      return el?.tagName === "CANVAS" ? { idx: i, x: p.x, y: p.y } : null;
    }, idx);
  };

  // ---------- search-Enter ----------
  if (want("search")) {
    results.scenarios.search = [];
    for (const t of TARGETS) {
      process.stderr.write(`[search] ${t.label} ${t.id} #${t.i} deg=${t.deg}\n`);
      results.scenarios.search.push({ label: t.label, name: t.id, deg: t.deg, ...(await run(() => search(t.id), t.id, resetFit)) });
      await page.evaluate(() => { const i = document.querySelector(".searchwrap input.search"); if (i) i.value = ""; });
    }
  }

  // ---------- node-card "fly to" ----------
  if (want("card")) {
    results.scenarios.cardFly = [];
    for (const t of TARGETS) {
      process.stderr.write(`[card] ${t.label} ${t.id}\n`);
      await resetFit(); await sleep(300);
      await search(t.id); await sleep(1300);
      results.scenarios.cardFly.push({
        label: t.label, name: t.id, deg: t.deg,
        ...(await run(() => page.click(".nodecard .row button:nth-child(1)"), t.id,
          async () => {
            await page.evaluate(() => {
              const c = window.__probe.cam || window.__cosmos3d;
              if (c && c.cameraSet) c.cameraSet([0, 220, 470], [0, 0, 0]);
            });
            await sleep(400);
          })),
      });
    }
  }

  // ---------- canvas node click ----------
  if (want("click") && is3d) {
    results.scenarios.click = [];
    for (const t of TARGETS) {
      process.stderr.write(`[click] ${t.label} ${t.id}\n`);
      await resetFit(); await sleep(300);
      // centre the node with a plain (z-aware) camera set so it is on screen
      await page.evaluate((idx) => {
        const c = window.__cosmos3d;
        const p = [c.pos[idx * 3], c.pos[idx * 3 + 1], c.pos[idx * 3 + 2]];
        c.cameraSet([p[0] + 40, p[1] + 60, p[2] + 120], p);
      }, t.i);
      await sleep(700);
      const hit = await findClickableNode(t.i);
      if (!hit) { results.scenarios.click.push({ label: t.label, name: t.id, skipped: "node not over a clickable canvas point" }); continue; }
      // sampling node = whatever we click at; resolution of the true target
      // happens from the node card after the flight
      const a = await run(async () => { await page.mouse.click(hit.x, hit.y); }, null);
      results.scenarios.click.push({ label: t.label, name: t.id, ...a });
    }
  }

  // ---------- fly-to right after real user gestures (dirty OrbitControls) ----------
  // This is the realistic path: the user orbits/zooms, then clicks a node
  // while damping residuals are still settling. flyToNode only sets
  // controls.enabled=false, but the render loop AND the tween both keep
  // calling controls.update(), which applies those residuals.
  if (want("dirty") && is3d) {
    for (const mode of ["orbit", "wheel", "pan"]) {
      results.scenarios[`dirty-${mode}`] = [];
      for (const t of TARGETS) {
        process.stderr.write(`[dirty-${mode}] ${t.label} ${t.id}\n`);
        await resetFit(); await sleep(400);
        const a = await run(async () => {
          if (mode === "orbit") await orbitDrag();
          else if (mode === "wheel") { await wheelZoom(-600); }
          else await rightDrag();
          await sleep(60); // user clicks almost immediately
          if (mode === "wheel") await search(t.id);
          else await search(t.id);
        }, t.id);
        results.scenarios[`dirty-${mode}`].push({ label: t.label, name: t.id, ...a });
        await page.evaluate(() => { const i = document.querySelector(".searchwrap input.search"); if (i) i.value = ""; });
      }
    }
  }

  // ---------- live-answer focus (real memory_recall → focusNode) ----------
  if (want("live") && is3d) {
    results.scenarios.liveAnswer = [];
    try {
      // pin the cosmos renderer: the recall switches to the Travel tab and
      // __cosmos3d gets re-pointed when the travel canvas loads
      await page.evaluate(() => { window.__probe.cam = window.__cosmos3d; });
      await page.evaluate(() => { const P = window.__probe; P.samples = []; P.sampling = true; });
      await page.fill(".querybar input", "What is demo contrastive alignment training?");
      await page.press(".querybar input", "Enter");
      const settled = await page
        .waitForFunction(() => document.querySelector(".querybar .qok") || document.querySelector(".querybar .qerr"), null, { timeout: 180000 })
        .then(() => true)
        .catch(() => false);
      await sleep(1800);
      const s = await stopSample();
      if (!settled) {
        results.scenarios.liveAnswer.push({ label: "live", skipped: "live recall did not settle within 180s (bridge-dependent)" });
      } else {
        const note = await page.textContent(".querybar .qok").catch(() => null);
        const err = await page.textContent(".querybar .qerr").catch(() => null);
        const info = await page.evaluate(() => {
          const c = window.__probe.cam;
          const t = c.controls.target;
          let best = -1, bd = Infinity;
          for (let i = 0; i < c.n; i++) {
            const dd = (c.pos[i * 3] - t.x) ** 2 + (c.pos[i * 3 + 1] - t.y) ** 2 + (c.pos[i * 3 + 2] - t.z) ** 2;
            if (dd < bd) { bd = dd; best = i; }
          }
          const world = [c.pos[best * 3], c.pos[best * 3 + 1], c.pos[best * 3 + 2]];
          const cam = c.camera;
          cam.updateMatrixWorld();
          cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
          return { name: c.names[best], world, ndc: window.__probe.projNDC(cam, world[0], world[1], world[2]) };
        });
        const a = analyze(s, info.world);
        a.label = "live";
        a.name = info.name;
        a.note = (note || err || "").slice(0, 90);
        a.finalNDC = info.ndc;
        a.finalNDCabs = Math.hypot(info.ndc[0], info.ndc[1]);
        a.finalNDCsrc = "camera-matrices";
        results.scenarios.liveAnswer.push(a);
      }
      await page.evaluate(() => { window.__probe.cam = null; });
    } catch (e) {
      results.scenarios.liveAnswer.push({ error: String(e).slice(0, 160) });
    }
  }

  // ---------- 2D fallback: user gesture must cancel the fly (no drift) ----------
  if (want("drift") && !is3d) {
    results.scenarios.drift2D = [];
    try {
      await resetFit();
      await sleep(400);
      const target = await page.evaluate(() => {
        const d = window.__cosmos2d;
        let best = -1, bd = -1;
        for (let i = 0; i < d.n; i++) {
          const dd = (d.wx[i] - d.cx) ** 2 + (d.wy[i] - d.cy) ** 2;
          if (dd > bd) { bd = dd; best = i; }
        }
        return { world: [d.wx[best], d.wy[best], 0] };
      });
      await startSample(target.world);
      // the exact call focusNode makes in the 2D branch
      await page.evaluate((w) => {
        const d = window.__cosmos2d;
        d.flyTo(w[0], w[1], d.currentZoom() * 3.2, 1400);
      }, target.world);
      await sleep(300);
      await orbitDrag(); // user gesture mid-flight
      const after = await probe2DState();
      await sleep(900);
      const later = await probe2DState();
      const s = await stopSample();
      const drift = Math.hypot(later.cx - after.cx, later.cy - after.cy);
      results.scenarios.drift2D.push({
        label: "2d-drag", frames: s.length,
        centreAtDrag: [after.cx, after.cy], centreAfter: [later.cx, later.cy],
        driftAfterDrag: drift, finalScale: later.scale,
        note: drift < 0.5 ? "fly cancelled on input (no drift)" : "STILL DRIFTING",
      });
    } catch (e) {
      results.scenarios.drift2D.push({ error: String(e).slice(0, 160) });
    }
  }

  // ---------- travel: journey-log row + journey-mode stage keyframes ----------
  if (want("travel") && is3d) {
    results.scenarios.journeyLog = [];
    results.scenarios.travelStages = [];
    try {
      await page.click("header.topbar nav button:nth-child(2)");
      await page.waitForFunction(() => { const c = window.__cosmos3d; return !!(c && c.n > 0 && document.querySelector(".jlog") && document.querySelector(".scrub")); }, null, { timeout: 90000 });
      // stage 0 has no log rows yet — advance so stage-1 entries stream in
      await page.waitForSelector(".scrub button", { timeout: 30000 });
      await page.click(".scrub > button:nth-child(3)");
      await sleep(1200);
      await page.waitForSelector(".jlogrow", { timeout: 20000 });
      const rows = await page.$$(".jlogrow");
      for (let k = 0; k < Math.min(3, rows.length); k++) {
        const rows2 = await page.$$(".jlogrow");
        const info = await rows2[k].evaluate((el) => ({ name: el.querySelector(".jlogname")?.textContent ?? "", title: el.getAttribute("title") }));
        if (!info.title) { results.scenarios.journeyLog.push({ label: `row${k}`, name: info.name, skipped: "chunk row (opens inspector)" }); continue; }
        process.stderr.write(`[journeyLog] row${k} ${info.name}\n`);
        results.scenarios.journeyLog.push({ label: `row${k}`, name: info.name, ...(await run(() => rows2[k].click(), info.name)) });
      }
    } catch (e) {
      results.scenarios.journeyLog.push({ error: String(e).slice(0, 160) });
    }
    try {
      // journey-mode stage keyframes. Each click advances one stage; the stage
      // driver flies the camera to that stage's solved keyframe pose. We
      // measure, per transition: overshoot past the arrival pose along the
      // travel axis, the honoured distance to the stage's focus target, and how
      // well the camera frames that focus (the focus is the stage's key
      // content: query / vector cluster / seeds / bm25 centroid …).
      await page.waitForSelector(".scrub button", { timeout: 20000 });
      // deterministically start at stage 0 so labels match real stages
      await page.focus(".timeline");
      await page.keyboard.press("Home");
      await sleep(700);
      // stage 0 is current; sweep all 8 stages by pressing "next".
      for (let i = 0; i < 8; i++) {
        const pre = await page.evaluate(() => { const c = window.__cosmos3d; return [c.camera.position.toArray(), c.controls.target.toArray()]; });
        await page.evaluate(() => { window.__probe.samples = []; window.__probe.node = [0, 0, 0]; window.__probe.sampling = true; });
        await page.click(".scrub > button:nth-child(3)");
        await sleep(DWELL);
        const s = await stopSample();
        const stageNo = (i + 1) % 8;
        const endTgt = s[s.length - 1]?.tgt ?? [0, 0, 0];
        // the stage focus IS the final controls.target (the keyframe target);
        // measure the trajectory against it for overshoot + honoured distance.
        const a = analyze(s, endTgt);
        const framed = await page.evaluate((t) => {
          const c = window.__cosmos3d;
          const cam = c.camera;
          cam.updateMatrixWorld();
          cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
          const ndc = window.__probe.projNDC(cam, t[0], t[1], t[2]);
          return { ndc, abs: Math.hypot(ndc[0], ndc[1]), tgtOff: Math.hypot(c.controls.target.x - t[0], c.controls.target.y - t[1], c.controls.target.z - t[2]) };
        }, endTgt);
        results.scenarios.travelStages.push({
          label: `stage${stageNo}`, stage: stageNo, frames: s.length,
          overshoot: a.overshoot, minDist: a.minDist, finalDist: a.finalDist,
          finalNDC: framed.ndc, finalNDCabs: framed.abs, tgtOff: framed.tgtOff,
          maxAbsNDC: a.maxAbsNDC,
          startPos: s[0]?.pos, endPos: s[s.length - 1]?.pos, endTgt,
          travel: a.travelLen,
          prePos: pre?.[0], preTgt: pre?.[1],
        });
      }
    } catch (e) {
      results.scenarios.travelStages.push({ error: String(e).slice(0, 160) });
    }
  }

  // ---------- presentation chapter flies ----------
  if (want("pres") && is3d) {
    results.scenarios.presentation = [];
    try {
      await page.click("header.topbar nav button:nth-child(1)");
      await sleep(300);
      await page.click(".panel.left button.primary");
      await page.waitForSelector(".pres", { timeout: 20000 });
      for (let i = 0; i < 3; i++) {
        await page.keyboard.press("ArrowRight");
        await sleep(60);
        const pre = await page.evaluate(() => { const c = window.__cosmos3d; return c ? [c.camera.position.toArray(), c.controls.target.toArray()] : null; });
        await page.evaluate(() => { window.__probe.samples = []; window.__probe.node = [0, 0, 0]; window.__probe.sampling = true; });
        await sleep(1600);
        const s = await stopSample();
        results.scenarios.presentation.push({ chapter: i + 3, frames: s.length, startPos: s[0]?.pos, endPos: s[s.length - 1]?.pos, endTgt: s[s.length - 1]?.tgt, travel: s.length ? dist3(s[0].pos, s[s.length - 1].pos) : null, prePos: pre?.[0], preTgt: pre?.[1] });
      }
      await page.keyboard.press("Escape");
      await sleep(400);
    } catch (e) {
      results.scenarios.presentation.push({ error: String(e).slice(0, 160) });
    }
  }

  await browser.close();
  if (OUT) writeFileSync(OUT, JSON.stringify(results, null, 2));
  if (AS_JSON) console.log(JSON.stringify(results, null, 2));
  else printReport(results);
}

const fmtN = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

function printReport(r) {
  console.log(`\nFlyTo probe — ${r.url}  [${r.mode}]  is3d=${r.is3d}`);
  console.log("targets:", r.targets.map((t) => `${t.label}=${t.name}(deg ${t.deg}, #${t.idx})`).join("  "));
  const NODE_COLS = [
    { h: "label", w: 6, f: (x) => x.label ?? "" },
    { h: "name", w: 24, f: (x) => (x.name ?? "").slice(0, 24) },
    { h: "frames", w: 6, f: (x) => x.frames ?? "skip" },
    { h: "overshoot", w: 9, f: (x) => fmtN(x.overshoot) },
    { h: "minDist", w: 8, f: (x) => fmtN(x.minDist) },
    { h: "finalDist", w: 9, f: (x) => fmtN(x.finalDist) },
    { h: "tgtOff", w: 7, f: (x) => fmtN(x.tgtOff) },
    { h: "finalNDC", w: 16, f: (x) => (x.finalNDC ? `(${fmtN(x.finalNDC[0])},${fmtN(x.finalNDC[1])})` : "—") },
    { h: "maxNDC", w: 7, f: (x) => fmtN(x.maxAbsNDC) },
    { h: "sel", w: 4, f: (x) => (x.matched === false ? "WRONG" : x.selected ? "ok" : "") },
    { h: "note", w: 22, f: (x) => x.skipped ?? (x.error ? String(x.error).slice(0, 20) : "") },
  ];
  const CAM_COLS = [
    { h: "label", w: 9, f: (x) => x.label ?? x.chapter ?? "" },
    { h: "frames", w: 6, f: (x) => x.frames },
    { h: "startPos", w: 24, f: (x) => x.startPos?.map((n) => n.toFixed(1)).join(",") },
    { h: "endPos", w: 24, f: (x) => x.endPos?.map((n) => n.toFixed(1)).join(",") },
    { h: "endTgt", w: 24, f: (x) => x.endTgt?.map((n) => n.toFixed(1)).join(",") },
    { h: "travel", w: 8, f: (x) => fmtN(x.travel) },
    { h: "note", w: 20, f: (x) => (x.error ? String(x.error).slice(0, 20) : "") },
  ];
  const table = (title, rows, cols) => {
    console.log(`\n== ${title} ==`);
    if (!rows.length) { console.log("  (none)"); return; }
    console.log(cols.map((c) => c.h.padEnd(c.w)).join(" "));
    console.log(cols.map((c) => "-".repeat(c.w)).join(" "));
    for (const row of rows) console.log(cols.map((c) => String(c.f(row)).padEnd(c.w)).join(" "));
  };
  const STAGE_COLS = [
    { h: "label", w: 8, f: (x) => x.label ?? "" },
    { h: "frames", w: 6, f: (x) => x.frames ?? "skip" },
    { h: "overshoot", w: 9, f: (x) => fmtN(x.overshoot) },
    { h: "minDist", w: 8, f: (x) => fmtN(x.minDist) },
    { h: "finalDist", w: 9, f: (x) => fmtN(x.finalDist) },
    { h: "tgtOff", w: 8, f: (x) => fmtN(x.tgtOff) },
    { h: "finalNDC", w: 16, f: (x) => (x.finalNDC ? `(${fmtN(x.finalNDC[0])},${fmtN(x.finalNDC[1])})` : "—") },
    { h: "travel", w: 8, f: (x) => fmtN(x.travel) },
    { h: "note", w: 20, f: (x) => (x.error ? String(x.error).slice(0, 20) : "") },
  ];
  const DRIFT_COLS = [
    { h: "label", w: 10, f: (x) => x.label ?? "" },
    { h: "frames", w: 6, f: (x) => x.frames },
    { h: "centre@drag", w: 22, f: (x) => x.centreAtDrag?.map((n) => n.toFixed(1)).join(",") },
    { h: "centre@end", w: 22, f: (x) => x.centreAfter?.map((n) => n.toFixed(1)).join(",") },
    { h: "drift", w: 7, f: (x) => fmtN(x.driftAfterDrag) },
    { h: "scale", w: 7, f: (x) => fmtN(x.finalScale) },
    { h: "note", w: 30, f: (x) => x.note ?? (x.error ? String(x.error).slice(0, 30) : "") },
  ];
  for (const [k, v] of Object.entries(r.scenarios)) {
    if (!v.length) { table(k, v, NODE_COLS); continue; }
    if (k === "travelStages") table(k, v, STAGE_COLS);
    else if (k === "presentation") table(k, v, CAM_COLS);
    else if (k === "drift2D") table("2D fallback: user gesture cancels the fly (no drift)", v, DRIFT_COLS);
    else table(k, v, NODE_COLS);
  }
  if (r.logs?.length) console.log("\nconsole errors:", r.logs.slice(0, 6));
  console.log("");
}

main().catch((e) => { console.error(e); process.exit(1); });
