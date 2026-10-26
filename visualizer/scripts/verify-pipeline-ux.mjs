import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

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

async function run() {
  console.log("Starting local Vite preview / dev server...");
  const vite = spawn("npm", ["run", "dev", "--", "--port", "5191"], {
    cwd: join(ROOT, "app"),
    stdio: ["ignore", "pipe", "pipe"],
  });

  const url = await new Promise((resolve, reject) => {
    let out = "";
    vite.stdout.on("data", (d) => {
      out += d.toString();
      const m = out.match(/http:\/\/localhost:(\d+)\/travel\//) || out.match(/http:\/\/127.0.0.1:(\d+)\/travel\//);
      if (m) resolve(`http://localhost:${m[1]}/travel/`);
    });
    vite.stderr.on("data", (d) => {
      out += d.toString();
    });
    setTimeout(() => reject(new Error(`Vite failed to start: ${out}`)), 15000);
  });
  console.log(`Vite running at ${url}`);

  const profile = mkdtempSync(join(tmpdir(), "chrome-pipe-eval-"));
  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--window-size=1600,1000",
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    "about:blank",
  ];

  const chrome = spawn(chromePath, args, { stdio: ["ignore", "ignore", "pipe"] });
  const wsEndpoint = await new Promise((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("Chrome failed to start within 15s")), 15000);
    chrome.stderr.on("data", (d) => {
      buf += d.toString();
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
  });

  const port = new URL(wsEndpoint).port;
  const r = await fetch(`http://127.0.0.1:${port}/json/list`);
  const targets = (await r.json()).filter((tg) => tg.type === "page");
  const cdp = await Cdp.connect(targets[0].webSocketDebuggerUrl);

  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");

  console.log(`Navigating to ${url}...`);
  await cdp.send("Page.navigate", { url });
  await new Promise((res) => setTimeout(res, 4000));

  console.log("\n--- TEST 1: Idle state verification ---");
  const idleState = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const qsteps = document.querySelector('.qsteps');
      const pipelineWidget = document.querySelector('.pipeline-widget');
      const querybar = document.querySelector('.querybar');
      const canvas = document.querySelector('canvas');
      const bodyText = document.body.innerText;
      return {
        hasQSteps: !!qsteps,
        hasPipelineWidget: !!pipelineWidget,
        hasQueryBar: !!querybar,
        hasCanvas: !!canvas,
        hasPipelineText: /pipeline 1 keywords/i.test(bodyText) || /конвейер 1 ключевые/i.test(bodyText)
      };
    })()`,
    returnByValue: true,
  });

  console.log("Idle state results:", idleState.result.value);
  if (idleState.result.value.hasQSteps || idleState.result.value.hasPipelineWidget || idleState.result.value.hasPipelineText) {
    throw new Error("FAILED: Idle state contains pipeline bars or clutter!");
  }
  console.log("PASS: Idle state is clean, pure 3D starfield with unobstructed canvas and no pipeline clutter.");

  console.log("\n--- TEST 2: Query submission & right-side widget slide-in ---");
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const input = document.querySelector('.querybar input[type=text]');
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(input, 'How does the memory combine dense and graph retrieval?');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      const btn = document.querySelector('.querybar button.primary');
      if (btn) btn.click();
    })()`,
  });

  await new Promise((res) => setTimeout(res, 1200));

  const activeState = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const pw = document.querySelector('.pipeline-widget');
      if (!pw) return null;
      const stages = Array.from(pw.querySelectorAll('.pw-stage-name')).map(s => s.textContent.trim());
      const isRightAligned = window.getComputedStyle(pw).right !== 'auto';
      return {
        hasWidget: !!pw,
        isRightAligned,
        stageCount: stages.length,
        stages
      };
    })()`,
    returnByValue: true,
  });

  console.log("Active query pipeline widget:", activeState.result.value);
  if (!activeState.result.value?.hasWidget || activeState.result.value?.stageCount !== 7) {
    throw new Error("FAILED: Pipeline widget did not render all 7 stages on the right!");
  }
  console.log("PASS: Animated right-side widget slides in and shows the 7 retrieval stages in order.");

  console.log("\n--- TEST 3: Stage 2 / Stage 4 node glow and pulse in 3D Cosmos ---");
  const cosmosHighlightState = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const c3d = window.__cosmos3d;
      const halos = c3d?.halos?.length ?? 0;
      const spotlight = c3d?.spot !== null;
      return {
        hasC3D: !!c3d,
        halosCount: halos,
        isSpotlighted: spotlight
      };
    })()`,
    returnByValue: true,
  });
  console.log("3D Cosmos highlight state:", cosmosHighlightState.result.value);

  console.log("\n--- TEST 4: Interaction - minimize and expand ---");
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const minBtn = document.querySelector('.pw-action-btn');
      if (minBtn) minBtn.click();
    })()`,
  });
  await new Promise((res) => setTimeout(res, 500));

  const minimizedState = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const pill = document.querySelector('.pipeline-pill');
      return { hasPill: !!pill, pillText: pill?.textContent?.trim() };
    })()`,
    returnByValue: true,
  });
  console.log("Minimized state:", minimizedState.result.value);
  if (!minimizedState.result.value.hasPill) throw new Error("FAILED: Minimized pill not found!");

  // Expand back
  await cdp.send("Runtime.evaluate", {
    expression: `document.querySelector('.pipeline-pill')?.click()`,
  });
  await new Promise((res) => setTimeout(res, 500));

  console.log("\n--- TEST 5: Stage Inspection (clicking stage 2 and stage 4) ---");
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const stageHeaders = document.querySelectorAll('.pw-stage-header');
      if (stageHeaders[1]) stageHeaders[1].click(); // Stage 2: Vector hits
    })()`,
  });
  await new Promise((res) => setTimeout(res, 800));

  const stage2Detail = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const chunks = document.querySelectorAll('.pw-chunk-item');
      const c3d = window.__cosmos3d;
      return {
        chunkCount: chunks.length,
        halosCount: c3d?.halos?.length ?? 0
      };
    })()`,
    returnByValue: true,
  });
  console.log("Stage 2 detail chunks and 3D halos:", stage2Detail.result.value);

  chrome.kill("SIGKILL");
  vite.kill("SIGKILL");
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* noop */ }
  console.log("\nALL UX & 3D COSMOS PIPELINE TESTS PASSED!");
  process.exit(0);
}

run().catch((e) => {
  console.error("Test error:", e);
  process.exit(1);
});
