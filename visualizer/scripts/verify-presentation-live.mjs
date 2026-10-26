import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
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
  console.log("Starting Vite dev server on port 5176...");
  const vite = spawn("npm", ["--prefix", "app", "run", "dev", "--", "--port", "5176"], {
    cwd: join(process.cwd(), "../memory-travel"),
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, MEMORY_TRACE_URL: "http://127.0.0.1:8098" },
  });

  let serverStarted = false;
  vite.stdout.on("data", (d) => {
    const s = d.toString();
    if (s.includes("http://localhost:5176") || s.includes("http://127.0.0.1:5176")) {
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

  const profile = mkdtempSync(join(tmpdir(), "chrome-pres-test-"));
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

  await cdp.send("Page.navigate", { url: "http://localhost:5176/travel/" });

  // If there's an "Open demo data" / "открыть демо" button, click it first
  await sleep(1500);
  await evalExpr(cdp, `(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const demo = btns.find(x => x.innerText.toLowerCase().includes("demo") || x.innerText.toLowerCase().includes("демо"));
    if (demo) demo.click();
  })()`);

  await sleep(1500);

  // Wait for tour button to be clickable
  console.log("Waiting for 'how it works' button to be enabled...");
  let enabled = false;
  for (let i = 0; i < 40; i++) {
    const btnInfo = await evalExpr(cdp, `(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      const b = btns.find(x => x.innerText.toLowerCase().includes("как это работает") || x.innerText.toLowerCase().includes("how it works"));
      return b ? { found: true, disabled: b.disabled, text: b.innerText } : { found: false };
    })()`);
    if (btnInfo.found && !btnInfo.disabled) {
      enabled = true;
      break;
    }
    await sleep(500);
  }

  console.log("Tour button state: enabled =", enabled);

  // Click the tour button
  await evalExpr(cdp, `(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find(x => x.innerText.toLowerCase().includes("как это работает") || x.innerText.toLowerCase().includes("how it works"));
    if (b) b.click();
  })()`);

  await sleep(1500);

  // Check presentation view
  const presState = await evalExpr(cdp, `(() => {
    const pres = document.querySelector(".pres");
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    return { inPres: !!pres, narr, fact };
  })()`);
  console.log("Pres state:", presState);

  // Next chapter: Chapter 2 (Ingest)
  console.log("Moving to Chapter 2...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1500);

  const ch2State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const hasIngest = !!document.querySelector(".ingest");
    const hasGate = !!document.querySelector(".presgate");
    return { narr, fact, hasIngest, hasGate };
  })()`);
  console.log("Chapter 2 state:", ch2State);

  // Next chapter: Chapter 3 (Embedding)
  console.log("Moving to Chapter 3...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(2500);

  const ch3State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    const hasGate = !!document.querySelector(".presgate");
    return { narr, fact, topCard, hasGate };
  })()`);
  console.log("Chapter 3 state:", ch3State);

  // Next chapter: Chapter 4 (Channels)
  console.log("Moving to Chapter 4...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1000);

  const ch4State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    return { narr, fact, topCard };
  })()`);
  console.log("Chapter 4 state:", ch4State);

  // Next chapter: Chapter 5 (Fusion)
  console.log("Moving to Chapter 5...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1000);

  const ch5State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    return { narr, fact, topCard };
  })()`);
  console.log("Chapter 5 state:", ch5State);

  // Next chapter: Chapter 6 (Rerank)
  console.log("Moving to Chapter 6...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1000);

  const ch6State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    return { narr, fact, topCard };
  })()`);
  console.log("Chapter 6 state:", ch6State);

  // Next chapter: Chapter 7 (Context)
  console.log("Moving to Chapter 7...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1000);

  const ch7State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    return { narr, fact, topCard };
  })()`);
  console.log("Chapter 7 state:", ch7State);

  // Next chapter: Chapter 8 (Synthesis)
  console.log("Moving to Chapter 8...");
  await evalExpr(cdp, `(() => {
    const nextBtn = document.querySelectorAll(".pres-controls button")[2];
    if (nextBtn) nextBtn.click();
  })()`);
  await sleep(1000);

  const ch8State = await evalExpr(cdp, `(() => {
    const narr = document.querySelector(".pres-narrcard")?.innerText;
    const fact = document.querySelector(".pres-factcard")?.innerText;
    const topCard = document.querySelector(".topcard")?.innerText;
    return { narr, fact, topCard };
  })()`);
  console.log("Chapter 8 state:", ch8State);

  console.log("Errors during run:", errors.length ? errors : "None!");

  chrome.kill();
  vite.kill();
}

main().catch((e) => {
  console.error("FAIL:", e);
  process.exit(1);
});
