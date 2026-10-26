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
  console.log("Starting Vite dev server on port 5178...");
  const vite = spawn("npm", ["--prefix", "app", "run", "dev", "--", "--port", "5178"], {
    cwd: process.cwd(),
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, MEMORY_TRACE_URL: "http://127.0.0.1:8098" },
  });

  let serverStarted = false;
  vite.stdout.on("data", (d) => {
    const s = d.toString();
    if (s.includes("5178")) {
      serverStarted = true;
    }
  });

  const timeout = setTimeout(() => {
    if (!serverStarted) {
      console.error("Vite server failed to start in time");
      vite.kill();
      process.exit(1);
    }
  }, 10000);

  while (!serverStarted) {
    await sleep(200);
  }
  clearTimeout(timeout);

  const profile = mkdtempSync(join(tmpdir(), "chrome-stages-test-"));
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

  await cdp.send("Page.navigate", { url: "http://localhost:5178/travel/" });

  // Click demo button if present
  await sleep(1500);
  await evalExpr(cdp, `(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const demo = btns.find(x => x.innerText.toLowerCase().includes("demo") || x.innerText.toLowerCase().includes("демо"));
    if (demo) demo.click();
  })()`);

  await sleep(1500);

  // Click 'how it works' button
  console.log("Navigating into 'How it works' presentation...");
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
    await sleep(1500);
  }

  // Chapter 1: What this is
  const ch1 = await evalExpr(cdp, `({
    title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
    canvasVisible: !document.querySelector(".canvaswrap.hidden")?.contains(document.querySelector("[data-label='pres-full']")),
  })`);
  results.push({ chapter: 1, ...ch1 });

  // Move to Chapter 2: Ingest
  await nextChapter();
  const ch2 = await evalExpr(cdp, `({
    title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
    fact: document.querySelector(".pres-factcard")?.innerText,
  })`);
  results.push({ chapter: 2, ...ch2 });

  // Move to Chapter 3: Embedding
  await nextChapter();
  await sleep(1500);
  const shot3 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch3-embedding.png"), Buffer.from(shot3.data, "base64"));
  const ch3 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasBeacon = !!c3d?.queryBeacon;
    const cam = c3d?.cameraNow();
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasBeacon,
      camDist: cam ? Math.hypot(cam.pos[0] - cam.target[0], cam.pos[1] - cam.target[1], cam.pos[2] - cam.target[2]) : null,
      camPos: cam?.pos,
      camTarget: cam?.target,
      topCard: document.querySelector(".topcard")?.innerText?.replace(/\\n+/g, " "),
    };
  })()`);
  results.push({ chapter: 3, ...ch3 });

  // Move to Chapter 4: Channels (Vector vs BM25)
  await nextChapter();
  await sleep(1500);
  const shot4 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch4-channels.png"), Buffer.from(shot4.data, "base64"));
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

  // Move to Chapter 5: Graph Walk
  await nextChapter();
  await sleep(1500);
  const shot5 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch5-graphwalk.png"), Buffer.from(shot5.data, "base64"));
  const ch5 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasGraphWalkMesh = !!c3d?.graphWalkMesh;
    const edgesAttr = c3d?.graphWalkMesh?.geometry?.getAttribute("position")?.count ?? 0;
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasGraphWalkMesh,
      walkEdgesVerts: edgesAttr,
      halosCount: c3d?.halos?.length ?? 0,
      paintsCount: c3d?.paints?.size ?? 0,
    };
  })()`);
  results.push({ chapter: 5, ...ch5 });

  // Move to Chapter 6: Fusion & Ranking
  await nextChapter();
  await sleep(1500);
  const shot6 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch6-fusion.png"), Buffer.from(shot6.data, "base64"));
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

  // Move to Chapter 7: Context Assembly
  await nextChapter();
  await sleep(1500);
  const shot7 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch7-context.png"), Buffer.from(shot7.data, "base64"));
  const ch7 = await evalExpr(cdp, `(() => {
    const c3d = window.__cosmosTrace || window.__cosmos3d;
    const hasClusterHull = !!c3d?.clusterHull;
    return {
      title: document.querySelector(".pres-narrcard")?.innerText?.split("\\n")[0],
      hasClusterHull,
      halosCount: c3d?.halos?.length ?? 0,
    };
  })()`);
  results.push({ chapter: 7, ...ch7 });

  // Move to Chapter 8: Synthesis / Grounded Answer
  await nextChapter();
  await sleep(1500);
  const shot8 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(process.cwd(), "tmp", "ch8-answer.png"), Buffer.from(shot8.data, "base64"));
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

  // Capture final screenshot for visual inspection
  const screenshot = await cdp.send("Page.captureScreenshot", { format: "png" });
  const shotPath = join(process.cwd(), "tmp", "ch8-stage-view.png");
  writeFileSync(shotPath, Buffer.from(screenshot.data, "base64"));
  console.log(`Saved screenshot to ${shotPath}`);

  console.log("\n================ STAGE VISUAL INSPECTION RESULTS ================");
  console.log(JSON.stringify(results, null, 2));

  chrome.kill();
  vite.kill();

  if (errors.length) {
    console.error("Errors encountered:", errors);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("FAIL:", e);
  process.exit(1);
});
