// verify-ui — headless, REAL-TIME verification of the deployed dev server.
//
// Drives the app through CDP (no puppeteer/playwright dependency — Node's
// global WebSocket): loads the page, waits for the connection chip to resolve,
// screenshots the cosmos tab, the travel tab, a REAL live query typed through
// the query bar, the About card, and an offline (bridge-blocked) pass. Also
// collects console errors. Never uses --virtual-time-budget: every wait is
// wall-clock polling of the actual DOM state.
//
// Usage:  node scripts/verify-ui.mjs [--base http://127.0.0.1:5174]
// Env:    VERIFY_QUESTION — the live question to ask (default below).
// Screenshots land in docs/screenshots/verify-*.png; exit 1 on any failure.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = join(ROOT, "docs", "screenshots");
const BASE = (() => {
  const i = process.argv.indexOf("--base");
  return i >= 0 ? process.argv[i + 1] : "http://127.0.0.1:5174";
})();
const QUESTION =
  process.env.VERIFY_QUESTION ??
  "What is the VEA expert adapter and which datasets are used for training?";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Minimal CDP client over the Node global WebSocket
// ---------------------------------------------------------------------------

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.handlers = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(typeof ev.data === "string" ? ev.data : String(ev.data));
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(`${msg.error.message} (${msg.error.data ?? ""})`)) : resolve(msg.result);
      } else if (msg.method) {
        (this.handlers.get(msg.method) ?? []).forEach((fn) => fn(msg.params));
      }
    });
  }

  static async connect(wsUrl) {
    const ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", () => reject(new Error(`ws connect failed: ${wsUrl}`)), { once: true });
    });
    return new Cdp(ws);
  }

  send(method, params = {}) {
    const id = ++this.id;
    const p = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.ws.send(JSON.stringify({ id, method, params }));
    return p;
  }

  on(event, fn) {
    if (!this.handlers.has(event)) this.handlers.set(event, []);
    this.handlers.get(event).push(fn);
  }
}

// ---------------------------------------------------------------------------
// Chrome lifecycle
// ---------------------------------------------------------------------------

async function launchChrome({ blockApi = false } = {}) {
  const profile = mkdtempSync(join(tmpdir(), "opencode", "mt-verify-"));
  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--hide-scrollbars",
    "--window-size=1600,900",
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    "about:blank",
  ];
  const chrome = spawn("google-chrome", args, { stdio: ["ignore", "ignore", "pipe"] });
  const wsEndpoint = await new Promise((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("chrome did not report a DevTools endpoint")), 20000);
    chrome.stderr.on("data", (d) => {
      buf += d.toString();
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    chrome.on("exit", (code) => reject(new Error(`chrome exited early (${code}): ${buf.slice(0, 400)}`)));
  });
  // The browser-level endpoint: fetch the page target's ws URL.
  const port = new URL(wsEndpoint).port;
  const r = await fetch(`http://127.0.0.1:${port}/json/list`);
  const targets = (await r.json()).filter((tg) => tg.type === "page");
  const cdp = await Cdp.connect(targets[0].webSocketDebuggerUrl);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Log.enable");

  const consoleErrors = [];
  cdp.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error" || p.type === "warning") {
      consoleErrors.push(p.args.map((a) => a.value ?? a.description ?? a.type).join(" ").slice(0, 300));
    }
  });
  cdp.on("Runtime.exceptionThrown", (p) => {
    consoleErrors.push(String(p.exceptionDetails?.exception?.description ?? p.exceptionDetails?.text ?? "exception").slice(0, 300));
  });
  cdp.on("Log.entryAdded", (p) => {
    if (p.entry.level === "error") consoleErrors.push(`${p.entry.source}: ${p.entry.text} ${p.entry.url ?? ""}`.slice(0, 300));
  });

  if (blockApi) {
    // Simulate a FOREIGN deployment with no reachable memory: fail the dev
    // proxy paths (/api/*) AND the baked direct fallbacks (our box's 8098/8099
    // — exactly what a firewall on another server would do). Static /data
    // stays reachable: the demo dataset must still open behind its button.
    const patterns = [
      { urlPattern: `${BASE.replace(/\*/g, "\\*")}/api/*`, requestStage: "Request" },
      { urlPattern: "http://memory-host:*/*", requestStage: "Request" },
    ];
    await cdp.send("Fetch.enable", { patterns });
    cdp.on("Fetch.requestPaused", async (p) => {
      try {
        await cdp.send("Fetch.failRequest", { requestId: p.requestId, errorReason: "ConnectionFailed" });
      } catch { /* already handled */ }
    });
  }

  // record every request URL (for the source-of-data network assertions)
  const requests = [];
  await cdp.send("Network.enable");
  cdp.on("Network.requestWillBeSent", (p) => {
    const u = p?.request?.url;
    if (u && !/\.(png|svg|woff2?|css|js|map)(\?|$)/.test(u)) requests.push(u);
  });

  const cleanup = () => {
    try { chrome.kill("SIGKILL"); } catch { /* noop */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* noop */ }
  };
  return { cdp, consoleErrors, cleanup, requests };
}

// ---------------------------------------------------------------------------
// DOM helpers (all waits are real-time polls, never virtual time)
// ---------------------------------------------------------------------------

async function evaluate(cdp, expr) {
  const r = await cdp.send("Runtime.evaluate", {
    expression: expr,
    returnByValue: true,
    awaitPromise: true,
  });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
}

async function waitExpr(cdp, expr, timeoutMs, pollMs = 250) {
  const t0 = Date.now();
  for (;;) {
    const ok = await evaluate(cdp, expr).catch(() => false);
    if (ok) return true;
    if (Date.now() - t0 > timeoutMs) return false;
    await sleep(pollMs);
  }
}

async function screenshot(cdp, file) {
  const r = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(SHOTS, file), Buffer.from(r.data, "base64"));
  console.log(`shot: docs/screenshots/${file}`);
}

const pageText = () => evaluate(cdp, "document.body.innerText");

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

mkdirSync(SHOTS, { recursive: true });
const results = [];
const skips = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "ok" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

// ---- pass 1: online --------------------------------------------------------
const { cdp, consoleErrors, cleanup, requests } = await launchChrome();
try {
  const loaded = new Promise((res) => cdp.on("Page.loadEventFired", res));
  await cdp.send("Page.navigate", { url: `${BASE}/` });
  await loaded;
  // real-time grace: static data fetch + star build + memory_status roundtrip
  await waitExpr(cdp, "!!document.querySelector('.hudconn')", 20000);
  await waitExpr(cdp, `document.querySelector('.hudconn') && !document.querySelector('.hudconn').textContent.includes("connecting")`, 20000);
  await sleep(4000); // let the starfield finish painting before the shot

  const chip = await evaluate(cdp, "document.querySelector('.hudconn')?.textContent ?? null");
  const dot = await evaluate(cdp, "document.querySelector('.conndot')?.className ?? null");
  check("connection chip resolves to the live index", !!chip && !chip.includes("connecting") && !chip.includes("offline"), `chip="${chip}" dot="${dot}"`);
  check("no HARS branding anywhere on the page", !/\bhars\b/i.test(await pageText()));

  // The graph may be gated by the source probe (backend must serve the graph
  // export on the trace service). When the app says "not connected" the live
  // query cannot run — that is a backend deployment dependency, not a UI bug:
  // record it and skip the graph-dependent checks with the reason.
  const notConnected = await evaluate(
    cdp,
    `!!document.querySelector(".errcard") && /not connected/i.test(document.querySelector(".errcard")?.textContent ?? "")`,
  );
  if (notConnected) {
    skips.push("graph export API (:8098 /stats,/graph,/coords) not deployed — live query needs the graph; the app honestly shows 'not connected' + explicit demo button");
    console.log("skip: live query + travel tab (graph export API not deployed; app shows the honest not-connected state)");
  }

  // data-source badge: in the live state it must say WHICH index is rendered
  const badgeCls = await evaluate(cdp, `document.querySelector(".srcbadge")?.className ?? null`);
  const badgeTxt = await evaluate(cdp, `document.querySelector(".srcbadge .srclabel")?.textContent ?? null`);
  if (!notConnected) {
    check("source badge resolves to LIVE", /\blive\b|жив/i.test(badgeCls ?? ""), `class="${badgeCls}" label="${badgeTxt}"`);
    check("live badge names the served index_id", !!badgeTxt && /:\s*\S/.test(badgeTxt ?? ""), `label="${badgeTxt}"`);
    // network assertions: the graph came from the export API, NOT baked files
    const hitStats = requests.some((u) => /\/api\/graph-stats\/stats|:8098\/stats/.test(u));
    const hitGraph = requests.some((u) => /\/api\/graph-stats\/graph|:8098\/graph(\?|$)/.test(u));
    const hitCoords = requests.some((u) => /\/api\/graph-stats\/coords|:8098\/coords/.test(u));
    const hitBaked = requests.some((u) => /\/data\/(graph|coords_entities|coords_chunks|node_index)\.json/.test(u));
    check("live mode fetched /stats from the export API", hitStats, requests.filter((u) => /stats/.test(u)).slice(0, 3).join(" | "));
    check("live mode fetched /graph from the export API", hitGraph, requests.filter((u) => /graph/.test(u)).slice(0, 3).join(" | "));
    check("live mode fetched /coords from the export API", hitCoords);
    check("live mode did NOT fetch baked data files", !hitBaked, requests.filter((u) => /\/data\//.test(u)).slice(0, 3).join(" | "));
  }

  await screenshot(cdp, "verify-cosmos.png");

  // ---- travel tab + real live query (only when the graph is live)
  if (!notConnected) {
    await evaluate(cdp, `document.querySelectorAll("header.topbar nav button")[1].click()`);
    await sleep(3500); // subgraph compile + canvas paint
    const travelTitle = await evaluate(cdp, "document.querySelector('.panel h1')?.textContent ?? null");
    check("travel tab renders", travelTitle != null, `h1="${travelTitle}"`);
    await screenshot(cdp, "verify-travel.png");

    await evaluate(cdp, `document.querySelectorAll("header.topbar nav button")[0].click()`);
    await sleep(500);
    const typed = await evaluate(cdp, `(() => {
      const input = document.querySelector(".querybar input[type=text]");
      if (!input) return false;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(input, ${JSON.stringify(QUESTION)});
      input.dispatchEvent(new Event("input", { bubbles: true }));
      return true;
    })()`);
    check("query bar accepted the typed question", typed);
    await evaluate(cdp, `document.querySelector(".querybar button.primary")?.click()`);
    // fast bridges answer in under a second: poll for EITHER the in-flight
    // phase hint OR the completed note (both prove the live path ran)
    const started = await waitExpr(
      cdp,
      `!!document.querySelector(".querybar .qprogress") || !!document.querySelector(".querybar .qok") || !!document.querySelector(".querybar .qerr")`,
      8000,
      100,
    );
    check("live recall started (phase hint or answer visible)", started);
    // wait for the ask button to come back = recall finished (up to 130s)
    const done = await waitExpr(
      cdp,
      `!!document.querySelector(".querybar button.primary") && !document.querySelector(".querybar .qprogress")`,
      130000,
      2000,
    );
    check("live recall finished within 130s", done);
    await sleep(1500);
    const note = await evaluate(cdp, `document.querySelector(".querybar .qok")?.textContent ?? document.querySelector(".querybar .qerr")?.textContent ?? null`);
    const totalRow = await evaluate(
      cdp,
      `(() => {
        const rows = [...document.querySelectorAll(".qstep")];
        const total = rows.find((r) => r.textContent.includes("Total") || r.textContent.includes("Итого"));
        return total?.textContent ?? null;
      })()`,
    );
    check("live answer note shown", !!note && !/unreachable|failed/i.test(note), `note="${(note ?? "").slice(0, 160)}"`);
    check("pipeline total row measured", !!totalRow && /s\b|мс|ms/.test(totalRow), `total="${(totalRow ?? "").slice(0, 160)}"`);
    // Branding check after a live query: the trace DATA may legitimately quote
    // documents that mention anything (retrieved memory content is not UI
    // copy) — so this targets the app chrome only.
    const chromeText = await evaluate(
      cdp,
      `["header.topbar", ".querybar", ".hud", ".panel .sub", ".panel .metarow"].map((s) => document.querySelector(s)?.textContent ?? "").join("\\n")`,
    );
    check("no HARS branding in the app chrome after live query", !/\bhars\b/i.test(chromeText), chromeText.match(/.{0,40}\bhars\b.{0,40}/i)?.[0] ?? "");
    await screenshot(cdp, "verify-live-query.png");
  } else {
    await screenshot(cdp, "verify-not-connected.png");
  }

  // ---- about card (also triggers a fresh memory_status fetch)
  await evaluate(cdp, `document.querySelector("header.topbar button.doclink").click()`);
  await sleep(2000);
  const connTable = await evaluate(cdp, "document.querySelector('.conntable')?.innerText ?? null");
  check("about card shows the connection table", !!connTable, (connTable ?? "").replace(/\n/g, " | ").slice(0, 220));
  check("connection table names the live index", !!connTable && connTable.includes((chip ?? "").split(" ")[0] ?? "?"));
  await screenshot(cdp, "verify-about.png");

  // In the not-connected state the source probe's 404s are the EXPECTED
  // honest behaviour (backend not deployed yet) — not console noise.
  const finalConsole = notConnected
    ? consoleErrors.filter((m) => /\/api\/(graph-stats|stats|graph|coords)|:8098/.test(m) || /favicon/i.test(m) ? false : true)
    : consoleErrors.filter((m) => !/favicon/i.test(m));
  check("console clean (no unexpected errors/warnings)", finalConsole.length === 0, finalConsole.slice(0, 4).join(" || "));
} finally {
  cleanup();
}

// ---- pass 2: offline (bridge blocked) — honest not-connected + fixtures ----
{
  const off = await launchChrome({ blockApi: true });
  try {
    const loaded = new Promise((res) => off.cdp.on("Page.loadEventFired", res));
    await off.cdp.send("Page.navigate", { url: `${BASE}/` });
    await loaded;
    const settled = await waitExpr(
      off.cdp,
      `(() => {
        const c = document.querySelector(".hudconn");
        return c && (c.classList.contains("offline") || (c.classList.contains("connected") && !c.textContent.includes("connecting")));
      })()`,
      25000,
    );
    const chip = await evaluate(off.cdp, "document.querySelector('.hudconn')?.textContent ?? null");
    const isOffline = await evaluate(off.cdp, `document.querySelector(".hudconn")?.classList.contains("offline") ?? false`);
    check("offline pass: chip reports offline", settled && isOffline, `chip="${chip}"`);
    const gate = await evaluate(off.cdp, `!!document.querySelector(".errcard") && /not connected|Нет подключения/i.test(document.querySelector(".errcard")?.textContent ?? "")`);
    check("offline pass: honest not-connected card shown", gate);
    const noStars = await evaluate(off.cdp, `(() => {
      const nums = [...document.querySelectorAll(".hudnum")].map((el) => (el.textContent ?? "").trim());
      // honest unreachable state: HUD placeholders ("…") may exist, but NO
      // rendered node/edge counts and no stars behind them
      return nums.every((v) => !/\d/.test(v)) && !document.querySelector(".canvaswrap:not(.hidden) canvas ~ *");
    })()`);
    check("offline pass: NO graph data rendered (no star counts)", noStars);
    const noBaked = off.requests.some((u) => /\/data\/(graph|coords|node_index)/.test(u)) === false;
    check("offline pass: no baked data fetched before the demo click", noBaked);
    await screenshot(off.cdp, "verify-offline.png"); // the honest GATE state
    // the fixtures path is now an explicit demo action — take it if offered
    const hasDemo = await evaluate(off.cdp, `!![...document.querySelectorAll("button")].find((b) => /open demo|демо-данные/i.test(b.textContent))`);
    if (hasDemo) {
      await evaluate(off.cdp, `[...document.querySelectorAll("button")].find((b) => /open demo|демо-данные/i.test(b.textContent)).click()`);
      const demoSettled = await waitExpr(
        off.cdp,
        `!!document.querySelector(".srcbadge.demo") && document.querySelectorAll(".hudnum").length >= 2`,
        40000,
      );
      const demoBadge = await evaluate(off.cdp, `document.querySelector(".srcbadge .srclabel")?.textContent ?? ""`);
      check("offline pass: demo snapshot loads behind the explicit button (demo badge + stars)", demoSettled, String(demoBadge).slice(0, 120));
      const demoFetch = off.requests.some((u) => /\/data\/graph\.json/.test(u));
      check("offline pass: baked files fetched only AFTER the explicit click", demoFetch);
      await sleep(4000);
      await screenshot(off.cdp, "verify-demo.png"); // the explicit DEMO state
    }
    await sleep(1500);
  } finally {
    off.cleanup();
  }
}

const failed = results.filter((r) => !r.ok);
console.log(`\nverify-ui: ${results.length - failed.length}/${results.length} checks passed`);
if (skips.length) console.log("skipped (backend dependency):", skips.join("; "));
if (failed.length) {
  console.log("failed:", failed.map((f) => f.name).join("; "));
  process.exit(1);
}
