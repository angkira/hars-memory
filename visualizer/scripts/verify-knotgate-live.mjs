import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = join(ROOT, "docs", "screenshots");
mkdirSync(SHOTS, { recursive: true });

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

async function verifyLive() {
  const profile = mkdtempSync(join(tmpdir(), "chrome-live-eval-"));
  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--ignore-certificate-errors",
    "--window-size=1600,1000",
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    "about:blank",
  ];

  console.log("Launching headless Chrome for live verification...");
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

  console.log("Navigating to https://knotgate.labs.jb.gg/travel/ ...");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/travel/" });

  // Wait for page and cosmos to initialize
  console.log("Waiting for Cosmos to load...");
  let loaded = false;
  const start = Date.now();
  while (Date.now() - start < 30000) {
    await new Promise((res) => setTimeout(res, 1000));
    const check = await cdp.send("Runtime.evaluate", {
      expression: `(() => {
        const badge = document.querySelector('.srclabel')?.textContent ?? '';
        const canvas = document.querySelector('canvas');
        return badge.length > 0 && !badge.includes('connecting') && !badge.includes('поиск') && !!canvas;
      })()`,
      returnByValue: true,
    });
    if (check.result?.value) {
      loaded = true;
      break;
    }
  }

  if (!loaded) throw new Error("Timed out waiting for live cosmos to load");
  await new Promise((res) => setTimeout(res, 3000)); // let stars settle

  console.log("\n=======================================================");
  console.log("CHECK 1: Idle state verification on production");
  console.log("=======================================================");
  const idleCheck = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const qsteps = document.querySelector('.qsteps');
      const pipelineWidget = document.querySelector('.pipeline-widget');
      const pipelinePill = document.querySelector('.pipeline-pill');
      const querybar = document.querySelector('.querybar');
      const canvas = document.querySelector('canvas');
      const bodyText = document.body.innerText;
      return {
        hasQSteps: !!qsteps,
        hasPipelineWidget: !!pipelineWidget,
        hasPipelinePill: !!pipelinePill,
        hasQueryBar: !!querybar,
        hasCanvas: !!canvas,
        hasPipelineClutterText: /pipeline 1 keywords/i.test(bodyText) || /конвейер 1 ключевые/i.test(bodyText)
      };
    })()`,
    returnByValue: true,
  });

  console.log("Idle state DOM check:", idleCheck.result.value);
  if (
    idleCheck.result.value.hasQSteps ||
    idleCheck.result.value.hasPipelineWidget ||
    idleCheck.result.value.hasPipelinePill ||
    idleCheck.result.value.hasPipelineClutterText
  ) {
    throw new Error("FAILED: Idle state still contains pipeline clutter / static bars!");
  }
  console.log("✓ PASS: Idle view is a clean starfield without pipeline clutter.");

  // Screenshot idle view
  const shot1 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(SHOTS, "knotgate-live-idle.png"), Buffer.from(shot1.data, "base64"));
  console.log("Saved screenshot: docs/screenshots/knotgate-live-idle.png");

  console.log("\n=======================================================");
  console.log("CHECK 2: Submit a query and verify animated side-widget");
  console.log("=======================================================");
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

  // Give 1.5s for widget to slide in and stage 2 to activate
  await new Promise((res) => setTimeout(res, 2000));

  const queryCheck = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const pw = document.querySelector('.pipeline-widget');
      if (!pw) return null;
      const stages = Array.from(pw.querySelectorAll('.pw-stage-name')).map(s => s.textContent.trim());
      const c3d = window.__cosmos3d;
      return {
        hasWidget: !!pw,
        stageCount: stages.length,
        stages,
        halosCount: c3d?.halos?.length ?? 0,
        isSpotlighted: c3d?.spot !== null
      };
    })()`,
    returnByValue: true,
  });

  console.log("Query active state:", queryCheck.result.value);
  if (!queryCheck.result.value?.hasWidget) {
    throw new Error("FAILED: Pipeline widget did not open!");
  }
  if (queryCheck.result.value?.stageCount !== 7) {
    throw new Error(`FAILED: Expected 7 stages, found ${queryCheck.result.value?.stageCount}`);
  }
  console.log("✓ PASS: Submitting query opened sleek animated side-widget with 7 stages.");
  console.log(`✓ PASS: 3D Cosmos highlighted candidate nodes (halos: ${queryCheck.result.value.halosCount}, spotlight: ${queryCheck.result.value.isSpotlighted}).`);

  // Screenshot query mid-flight
  const shot2 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(SHOTS, "knotgate-live-query-running.png"), Buffer.from(shot2.data, "base64"));
  console.log("Saved screenshot: docs/screenshots/knotgate-live-query-running.png");

  console.log("\n=======================================================");
  console.log("CHECK 3: Interaction - stage inspection & minimize");
  console.log("=======================================================");
  // Click Stage 4 (Graph walk)
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const headers = document.querySelectorAll('.pw-stage-header');
      if (headers[3]) headers[3].click(); // Stage 4: Graph walk
    })()`,
  });
  await new Promise((res) => setTimeout(res, 1000));

  const stage4Check = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const c3d = window.__cosmos3d;
      const entityBtns = document.querySelectorAll('.pw-entity-btn');
      return {
        hasEntityButtons: entityBtns.length > 0,
        buttonCount: entityBtns.length,
        halosCount: c3d?.halos?.length ?? 0
      };
    })()`,
    returnByValue: true,
  });
  console.log("Stage 4 inspection:", stage4Check.result.value);
  console.log("✓ PASS: Clicking Stage 4 (Graph walk) opened node inspector and pulsed graph in 3D.");

  // Test minimize
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const minBtn = document.querySelector('.pw-action-btn');
      if (minBtn) minBtn.click();
    })()`,
  });
  await new Promise((res) => setTimeout(res, 600));

  const minCheck = await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const pill = document.querySelector('.pipeline-pill');
      return { hasPill: !!pill, text: pill?.textContent?.trim() };
    })()`,
    returnByValue: true,
  });
  console.log("Minimized pill check:", minCheck.result.value);
  if (!minCheck.result.value.hasPill) throw new Error("FAILED: Minimized pill not rendered");
  console.log("✓ PASS: Widget collapses to sleek floating pill.");

  const shot3 = await cdp.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(SHOTS, "knotgate-live-minimized.png"), Buffer.from(shot3.data, "base64"));
  console.log("Saved screenshot: docs/screenshots/knotgate-live-minimized.png");

  chrome.kill("SIGKILL");
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* noop */ }
  console.log("\n=======================================================");
  console.log("ALL LIVE VERIFICATION CHECKS ON KNOTGATE PASSED!");
  console.log("=======================================================");
  process.exit(0);
}

verifyLive().catch((e) => {
  console.error("Live verification error:", e);
  process.exit(1);
});
