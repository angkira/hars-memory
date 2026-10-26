// Headless browser verification using CDP directly via native Node.js WebSocket
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

async function main() {
  const profile = mkdtempSync(join(tmpdir(), "chrome-test-"));
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

  console.log("Launching Chrome...");
  const chrome = spawn(chromePath, args, { stdio: ["ignore", "ignore", "pipe"] });

  const wsEndpoint = await new Promise((resolve, reject) => {
    let buf = "";
    const timer = setTimeout(() => reject(new Error("chrome did not report DevTools endpoint")), 15000);
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
  await cdp.send("Network.enable");

  const consoleLogs = [];
  cdp.on("Runtime.consoleAPICalled", (p) => {
    consoleLogs.push(`[${p.type}] ` + p.args.map((a) => a.value ?? a.description ?? "").join(" "));
  });

  console.log("Navigating to https://knotgate.labs.jb.gg/travel/ ...");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/travel/" });

  // Poll for the page state
  console.log("Waiting for page and cosmos to initialize...");
  let state = null;
  const start = Date.now();
  while (Date.now() - start < 35000) {
    await new Promise((res) => setTimeout(res, 1000));
    try {
      const evalRes = await cdp.send("Runtime.evaluate", {
        expression: `({
          title: document.title,
          badge: document.querySelector('.srclabel')?.textContent ?? '',
          hasErrorCard: !!document.querySelector('.cosmos-error-card'),
          bodySnippet: document.body.innerText.slice(0, 500)
        })`,
        returnByValue: true,
      });
      state = evalRes.result?.value;
      if (
        state &&
        state.badge &&
        !state.badge.includes("connecting") &&
        !state.badge.includes("поиск")
      ) {
        break;
      }
    } catch (e) {
      // Continue polling
    }
  }

  console.log("\n--- Verification Results ---");
  console.log("Document title:", state?.title);
  console.log("Source badge:", state?.badge);
  console.log("Has error card:", state?.hasErrorCard);
  console.log("Body snippet:", state?.bodySnippet?.replace(/\n+/g, " "));

  if (consoleLogs.length > 0) {
    console.log("\nConsole logs (first 10):", consoleLogs.slice(0, 10));
  }

  chrome.kill("SIGKILL");

  const isConnected = state?.badge?.includes("живые данные") || state?.badge?.includes("live data") || state?.badge?.includes("live:");
  const hasError = state?.hasErrorCard || state?.bodySnippet?.includes("Not connected") || state?.bodySnippet?.includes("Не подключено");

  if (isConnected && !hasError) {
    console.log("\nSUCCESS: Page loaded and verified as CONNECTED to live graph!");
    process.exit(0);
  } else {
    console.error("\nFAIL: Page did not connect properly!");
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
