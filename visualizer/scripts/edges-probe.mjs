// Edge-visibility probe (Memory Travel).
//
// Real headless browser, REAL-TIME waits only (never --virtual-time-budget:
// it zeroes canvases / starves rAF). Captures the edge-visibility acceptance
// set and measures rAF frame deltas:
//
//   overview (default edges)              -> edges-overview[-before].png
//   five-step continuous zoom-in series   -> edges-zoom-<tag>-<d>.png  (+opacities)
//   selected node, incident edges lit     -> edges-selected.png
//   hover highlight                        -> edges-hover.png
//   edges toggled OFF                      -> edges-off.png
//
// Usage:
//   node scripts/edges-probe.mjs --url=http://127.0.0.1:5190 --tag=after
//   node scripts/edges-probe.mjs --url=http://127.0.0.1:5189 --tag=before --mode=overview
//   PW_ROOT=/dir/with/playwright/node_modules  (autodiscovered otherwise)
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "docs", "screenshots");
const args = new Map(process.argv.slice(2).map((a) => {
  const m = /^--([^=]+)(?:=(.*))?$/.exec(a);
  return m ? [m[1], m[2] ?? "true"] : [a, "true"];
}));
const URL_ = args.get("url") ?? "http://127.0.0.1:5190";
const TAG = args.get("tag") ?? "after";
const MODE = args.get("mode") ?? "full";
const FORCE_2D = args.has("force2d");
const VIEW = { width: 1400, height: 880 };

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// unique-prefix hub target (search Enter must select exactly this node)
const graph = JSON.parse(readFileSync(join(ROOT, "app", "public", "data", "graph.json"), "utf8"));
const names = JSON.parse(readFileSync(join(ROOT, "app", "public", "data", "node_index.json"), "utf8"));
const coords = JSON.parse(readFileSync(join(ROOT, "app", "public", "data", "coords_entities.json"), "utf8"));
const isPrefixOfOther = (nm) => names.some((x) => x !== nm && x.toLowerCase().startsWith(nm.toLowerCase()));
const byDeg = graph.nodes.map((n, i) => ({ i, id: n.id, deg: n.deg })).sort((a, b) => b.deg - a.deg);
const HUB = byDeg.find((d) => !isPrefixOfOther(d.id) && d.deg >= 5);
const MID = byDeg.find((d) => d.deg >= 3 && d.deg <= 12 && !isPrefixOfOther(d.id));
// low-degree leaf for the hover shot: its few incident edges read clearly
const LEAF = [...byDeg].reverse().find((d) => d.deg >= 2 && d.deg <= 4 && !isPrefixOfOther(d.id)) ?? MID;
// peripheral mid-degree node: sits in the sparse halo, so a hover starburst
// reads against black instead of drowning in the dense core
const rById = new Map();
coords.ids.forEach((id, i) => { const p = coords.xyz?.[i] ?? coords.xy[i]; if (p) rById.set(id, Math.hypot(p[0], p[1])); });
const PERI = byDeg
  .filter((d) => d.deg >= 4 && d.deg <= 20 && !isPrefixOfOther(d.id) && rById.has(d.id))
  .sort((a, b) => (rById.get(b.id) ?? 0) - (rById.get(a.id) ?? 0))[0] ?? MID;

const INSTRUMENT = () => {
  window.__ft = { t: [], on: false };
  const loop = (ts) => {
    if (window.__ft.on) window.__ft.t.push(ts);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
};

async function sampleFrames(page, ms) {
  await page.evaluate(() => { window.__ft.t = []; window.__ft.on = true; });
  await sleep(ms);
  return page.evaluate(() => {
    window.__ft.on = false;
    const t = window.__ft.t;
    const d = [];
    for (let i = 1; i < t.length; i++) d.push(t[i] - t[i - 1]);
    d.sort((a, b) => a - b);
    const mean = d.reduce((a, b) => a + b, 0) / Math.max(1, d.length);
    return {
      frames: t.length,
      mean: +mean.toFixed(2),
      median: +(d[Math.floor(d.length / 2)] ?? 0).toFixed(2),
      p95: +(d[Math.floor(d.length * 0.95)] ?? 0).toFixed(2),
      min: +(d[0] ?? 0).toFixed(2),
      max: +(d[d.length - 1] ?? 0).toFixed(2),
    };
  });
}

const shot = (page, name) => page.screenshot({ path: join(OUT, name), fullPage: false });

async function main() {
  const pw = loadPlaywright();
  const browser = await pw.chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({ viewport: VIEW, deviceScaleFactor: 1 });
  const errs = [];
  page.on("pageerror", (e) => errs.push(`PAGEERROR ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
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

  const report = { url: URL_, tag: TAG, mode: MODE, hub: HUB, mid: MID, leaf: LEAF, peri: PERI, screenshots: [], numbers: {}, errors: errs };

  await page.goto(URL_, { waitUntil: "domcontentloaded" });
  if (FORCE_2D) {
    await page.waitForFunction(() => !!window.__cosmos2d, null, { timeout: 120000 });
    await page.waitForFunction(() => !document.querySelector(".panel.left .loading"), null, { timeout: 120000 }).catch(() => {});
    await sleep(1500);
    report.numbers.mode2d = await page.evaluate(() => ({
      adaptive: typeof window.__cosmos2d.setEdgeZoomAdaptive === "function",
      focus: typeof window.__cosmos2d.setEdgeFocus === "function",
      toggle: !!document.querySelector(".edgebtn"),
    }));
    await shot(page, `edges-2d-overview-${TAG}.png`);
    report.screenshots.push(`edges-2d-overview-${TAG}.png`);
    report.numbers.overviewIntensity2d = await page.evaluate(() => window.__cosmos2d.edgeIntensity());
    // zoom in (instant fly, no gesture -> the raster repaints once)
    await page.evaluate(() => { const d = window.__cosmos2d; d.flyTo(d.cx, d.cy, d.scale * 6, 0); });
    await sleep(600);
    report.numbers.zoomIntensity2d = await page.evaluate(() => window.__cosmos2d.edgeIntensity());
    await shot(page, `edges-2d-zoom-${TAG}.png`);
    report.screenshots.push(`edges-2d-zoom-${TAG}.png`);
    // selection focus via the real UI path (search -> select -> setEdgeFocus)
    await page.evaluate(() => { window.__cosmos2d.fit(0.12); });
    await sleep(400);
    await page.click(".panel.left .searchwrap input");
    await page.fill(".panel.left .searchwrap input", PERI.id);
    await page.waitForSelector(".results button", { timeout: 8000 }).catch(() => {});
    await page.keyboard.press("Enter");
    await sleep(2200);
    report.numbers.selection2d = await page.evaluate(() => ({
      focus: window.__cosmos2d.edgeFocusIdx,
      card: document.querySelector(".nodecard h3")?.textContent ?? null,
    }));
    await shot(page, `edges-2d-selected-${TAG}.png`);
    report.screenshots.push(`edges-2d-selected-${TAG}.png`);
    // toggle off
    await page.click(".edgebtn");
    await sleep(500);
    report.numbers.toggleOff2d = await page.evaluate(() => ({
      intensity: window.__cosmos2d.edgeIntensity(),
      stored: localStorage.getItem("mt.edges.v1"),
    }));
    await shot(page, `edges-2d-off-${TAG}.png`);
    report.screenshots.push(`edges-2d-off-${TAG}.png`);
    console.log(JSON.stringify(report, null, 2));
    await browser.close();
    return;
  }

  await page.waitForFunction(() => {
    const c = window.__cosmos3d;
    return !!(c && c.n > 0 && document.querySelector(".hud"));
  }, null, { timeout: 120000 });
  await page.waitForFunction(() => !document.querySelector(".panel.left .loading"), null, { timeout: 120000 }).catch(() => {});
  await sleep(1800); // settle fit + auto-orbit + opacity ease

  const hasAdaptive = await page.evaluate(() => typeof window.__cosmos3d?.setEdgeZoomAdaptive === "function");
  const uiHasToggle = await page.evaluate(() => !!document.querySelector(".edgebtn"));
  report.numbers.hasAdaptive = hasAdaptive;
  report.numbers.uiToggle = uiHasToggle;

  const setDist = (d, idx) => page.evaluate(({ d, idx }) => {
    const c = window.__cosmos3d;
    const x = idx >= 0 ? c.pos[idx * 3] : 0;
    const y = idx >= 0 ? c.pos[idx * 3 + 1] : 0;
    const z = idx >= 0 ? c.pos[idx * 3 + 2] : 0;
    c.camera.position.set(x, y + 0.15 * d, z + d);
    c.controls.target.set(x, y, z);
    c.controls.update();
    return c.edgeIntensity ? c.edgeIntensity() : null;
  }, { d, idx });

  const intensity = () => page.evaluate(() => window.__cosmos3d?.edgeIntensity?.() ?? null);

  // ---- (a) overview, default edges -------------------------------------
  await shot(page, `edges-overview-${TAG}.png`);
  report.screenshots.push(`edges-overview-${TAG}.png`);
  report.numbers.overviewOpacity = await intensity();
  report.numbers.framesOverview = await sampleFrames(page, 2500);
  report.numbers.renderBenchOverview = await page.evaluate((n) => {
    const c = window.__cosmos3d;
    const r = c.renderer, s = c.scene, cam = c.camera;
    const time = (count) => { const t0 = performance.now(); for (let i = 0; i < count; i++) r.render(s, cam); return (performance.now() - t0) / count; };
    time(3);
    const on = time(n);
    c.edges.visible = false; time(3);
    const off = time(n);
    c.edges.visible = true;
    return { framesPerSample: n, msEdgesOn: +on.toFixed(2), msEdgesOff: +off.toFixed(2), msEdgeLayer: +(on - off).toFixed(2) };
  }, 15);

  // ---- (b) five-step continuous zoom-in series -------------------------
  const steps = [360, 150, 55, 20, 9.5];
  const zoom = [];
  for (const d of steps) {
    await setDist(d, HUB.i);
    await sleep(450);
    const op = await intensity();
    const name = `edges-zoom-${TAG}-d${d}.png`;
    await shot(page, name);
    report.screenshots.push(name);
    zoom.push({ d, opacity: op });
  }
  report.numbers.zoomSeries = zoom;
  report.numbers.framesZoomed = await sampleFrames(page, 2500);

  if (MODE === "overview") {
    report.numbers.framesEdgesOff = await sampleFrames(page, 1200);
    console.log(JSON.stringify(report, null, 2));
    await browser.close();
    return;
  }

  // ---- (c) selected node: incident edges lit ---------------------------
  // peripheral node -> its starburst reads clearly against the dimmed field
  await page.evaluate(() => { const c = window.__cosmos3d; c.fit(0.12); c.controls.update(); });
  await sleep(500);
  await page.click(".panel.left .searchwrap input");
  await page.fill(".panel.left .searchwrap input", PERI.id);
  await page.waitForSelector(".results button", { timeout: 8000 }).catch(() => {});
  await page.keyboard.press("Enter");
  await sleep(2200); // fly-to (850ms) + focus ease
  const focusReadback = await page.evaluate((i) => ({
    sel: window.__cosmos3d.selIdx,
    focus: window.__cosmos3d.focusIdx,
    uniform: window.__cosmos3d.edgeMat?.uniforms?.uFocus?.value ?? null,
    card: document.querySelector(".nodecard h3")?.textContent ?? null,
  }), PERI.i);
  report.numbers.selection = focusReadback;
  await shot(page, "edges-selected.png");
  report.screenshots.push("edges-selected.png");

  // deselect, then zoom to a PERIPHERAL node (no selection) and hover it, so
  // its incident edges form a clear starburst against the sparse field
  await page.click(".panel.left .nodecard .ghostbtn");
  await sleep(700);
  await page.evaluate((i) => {
    const c = window.__cosmos3d;
    const x = c.pos[i * 3], y = c.pos[i * 3 + 1], z = c.pos[i * 3 + 2];
    c.camera.position.set(x, y, z + 45);
    c.controls.target.set(x, y, z);
    c.controls.update();
  }, PERI.i);
  await sleep(800);
  const hovered = await page.evaluate((i) => {
    const p = window.__cosmos3d.projectNode(i);
    return p ? { x: p.x, y: p.y } : null;
  }, PERI.i);
  if (hovered) {
    await page.mouse.move(hovered.x, hovered.y, { steps: 8 });
    await sleep(120);
    await page.mouse.move(hovered.x + 2, hovered.y + 1, { steps: 2 }); // fresh pointermove frame
    await sleep(600);
  }
  const hoverReadback = await page.evaluate(() => ({
    hoverIdx: window.__cosmos3d.hoverIdx,
    uniform: window.__cosmos3d.edgeMat?.uniforms?.uHover?.value ?? null,
    tip: !!document.querySelector(".hovertip"),
  }));
  report.numbers.hover = hoverReadback;
  await shot(page, "edges-hover.png");
  report.screenshots.push("edges-hover.png");

  // ---- synchronous render benchmark (stable per-frame cost, edges on/off)
  report.numbers.renderBenchClose = await page.evaluate((n) => {
    const c = window.__cosmos3d;
    const r = c.renderer, s = c.scene, cam = c.camera;
    const time = (count) => { const t0 = performance.now(); for (let i = 0; i < count; i++) r.render(s, cam); return (performance.now() - t0) / count; };
    time(3); // warm
    const on = time(n);
    c.edges.visible = false; time(3);
    const off = time(n);
    c.edges.visible = true;
    return { framesPerSample: n, msEdgesOn: +on.toFixed(2), msEdgesOff: +off.toFixed(2), msEdgeLayer: +(on - off).toFixed(2) };
  }, 15);

  // ---- (e) toggle OFF (+ persistence) ----------------------------------
  await page.mouse.move(10, 10);
  await page.click(".edgebtn");
  await sleep(700);
  const offReadback = await page.evaluate(() => ({
    label: document.querySelector(".edgebtn")?.textContent?.trim(),
    pressed: document.querySelector(".edgebtn")?.getAttribute("aria-pressed"),
    intensity: window.__cosmos3d?.edgeIntensity?.(),
    visible: window.__cosmos3d?.edges?.visible,
    stored: localStorage.getItem("mt.edges.v1"),
  }));
  report.numbers.toggleOff = offReadback;
  await shot(page, "edges-off.png");
  report.screenshots.push("edges-off.png");
  report.numbers.framesEdgesOff = await sampleFrames(page, 2500);

  // restore ON for cleanliness
  await page.click(".edgebtn");
  await sleep(400);

  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}

main().catch((e) => {
  console.error("edges-probe failed:", e);
  process.exit(1);
});
