import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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
        msg.error ? reject(new Error(`${msg.error.message}`)) : resolve(msg.result);
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

async function evalExpr(cdp, expr) {
  const res = await cdp.send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (res.exceptionDetails) {
    throw new Error(`Eval error: ${JSON.stringify(res.exceptionDetails)}`);
  }
  return res.result?.value;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const profile = mkdtempSync(join(tmpdir(), "chrome-prod-live-"));
  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--ignore-certificate-errors",
    "--window-size=1600,900",
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    "about:blank",
  ];

  console.log("Launching headless Chrome for live production verification...");
  const chrome = spawn(chromePath, args, { stdio: ["ignore", "ignore", "pipe"] });

  const wsEndpoint = await new Promise((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("chrome timeout")), 15000);
    chrome.stderr.on("data", (d) => {
      buf += d.toString();
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    chrome.on("exit", (code) => reject(new Error(`chrome exited (${code}): ${buf}`)));
  });

  const port = new URL(wsEndpoint).port;
  const r = await fetch(`http://127.0.0.1:${port}/json/list`);
  const targets = (await r.json()).filter((tg) => tg.type === "page");
  const cdp = await Cdp.connect(targets[0].webSocketDebuggerUrl);

  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");

  const errors = [];
  cdp.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") {
      errors.push(p.args.map((a) => a.value ?? a.description ?? "").join(" "));
    }
  });

  console.log("Navigating to https://knotgate.labs.jb.gg/travel/ ...");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/travel/" });

  // Wait for page and connection
  console.log("Waiting for connected status on live cluster...");
  let connected = false;
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    const badge = await evalExpr(cdp, `document.querySelector(".srclabel")?.innerText || ""`);
    if (badge && !badge.includes("connecting") && !badge.includes("поиск")) {
      console.log(`Live source badge: ${badge}`);
      connected = true;
      break;
    }
  }

  // Click How It Works
  console.log("Entering 'How it works' interactive presentation...");
  await evalExpr(cdp, `(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find(x => x.innerText.toLowerCase().includes("как это работает") || x.innerText.toLowerCase().includes("how it works"));
    if (b) b.click();
  })()`);

  await sleep(1500);

  const results = [];
  async function nextChapter() {
    await evalExpr(cdp, `(() => {
      const nextBtn = document.querySelectorAll(".pres-controls button")[2];
      if (nextBtn) nextBtn.click();
    })()`);
    await sleep(1800);
  }

  // Chapter 1: What this is
  const ch1 = await evalExpr(cdp, `({
    title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
    inPres: !!document.querySelector(".pres"),
  })`);
  results.push({ chapter: 1, ...ch1 });

  // Chapter 2: Ingest
  await nextChapter();
  const ch2 = await evalExpr(cdp, `({
    title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
  })`);
  results.push({ chapter: 2, ...ch2 });

  // Chapter 3: Embedding
  await nextChapter();
  await sleep(1500);
  const ch3 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasBeacon = !!c3d?.queryBeacon;
    const cam = c3d?.cameraNow();
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasBeacon,
      hasTraceCanvas: !!document.querySelector(".pres")?.querySelector(".canvaswrap:not(.hidden)"),
      camDist: cam ? Math.hypot(cam.pos[0] - cam.target[0], cam.pos[1] - cam.target[1], cam.pos[2] - cam.target[2]) : null,
      topCard: document.querySelector(".topcard")?.innerText?.replace(/\\n+/g, " "),
      nodesInTraceCosmos: window.__cosmosTrace?.n ?? null,
    };
  })()`);
  results.push({ chapter: 3, ...ch3 });

  // Chapter 4: Channels
  await nextChapter();
  const ch4 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      paintsCount: c3d?.paints?.size ?? 0,
      halosCount: c3d?.halos?.length ?? 0,
      topCard: document.querySelector(".topcard")?.innerText?.replace(/\\n+/g, " "),
    };
  })()`);
  results.push({ chapter: 4, ...ch4 });

  // Chapter 5: Graph Walk
  await nextChapter();
  const ch5 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasGraphWalkMesh = !!c3d?.graphWalkMesh;
    const edgesAttr = c3d?.graphWalkMesh?.geometry?.getAttribute("position")?.count ?? 0;
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasGraphWalkMesh,
      walkEdgesVerts: edgesAttr,
      halosCount: c3d?.halos?.length ?? 0,
    };
  })()`);
  results.push({ chapter: 5, ...ch5 });

  // Chapter 6: Fusion & Ranking
  await nextChapter();
  const ch6 = await evalExpr(cdp, `(() => {
    const pins = Array.from(document.querySelectorAll(".pres-pin")).map(p => ({
      num: p.querySelector(".pres-pin-num")?.innerText,
      label: p.querySelector(".pres-pin-label")?.innerText,
    }));
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      pinCount: pins.length,
      samplePins: pins.slice(0, 4),
    };
  })()`);
  results.push({ chapter: 6, ...ch6 });

  // Chapter 7: Context Assembly
  await nextChapter();
  const ch7 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasClusterHull = !!c3d?.clusterHull;
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasClusterHull,
    };
  })()`);
  results.push({ chapter: 7, ...ch7 });

  // Chapter 8: Synthesis / Grounded Answer
  await nextChapter();
  const ch8 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasTrail = !!c3d?.trail;
    const hasCitationBeams = !!c3d?.citationBeams;
    const pins = Array.from(document.querySelectorAll(".pres-pin")).map(p => ({
      num: p.querySelector(".pres-pin-num")?.innerText,
      label: p.querySelector(".pres-pin-label")?.innerText,
    }));
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasTrail,
      hasCitationBeams,
      citationPins: pins.slice(0, 3),
    };
  })()`);
  results.push({ chapter: 8, ...ch8 });

  const screenshot = await cdp.send("Page.captureScreenshot", { format: "png" });
  const shotPath = join(process.cwd(), "tmp", "prod-live-walkthrough.png");
  writeFileSync(shotPath, Buffer.from(screenshot.data, "base64"));
  console.log(`Saved live production screenshot to ${shotPath}`);

  console.log("\n================ LIVE PRODUCTION VERIFICATION RESULTS ================");
  console.log(JSON.stringify(results, null, 2));

  chrome.kill();

  if (errors.length) {
    console.error("Console errors encountered:", errors);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("FAIL:", e);
  process.exit(1);
});
