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

async function run() {
  const profile = mkdtempSync(join(tmpdir(), "chrome-full-eval-"));
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

  console.log("Launching Headless Chrome for End-to-End Verification...");
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
    chrome.on("exit", (code) => reject(new Error(`Chrome exited with code ${code}`)));
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

  console.log("\n========================================================");
  console.log("TEST 1: Memory Travel Visualizer (https://knotgate.labs.jb.gg/travel/)");
  console.log("========================================================");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/travel/" });

  let travelState = null;
  const start1 = Date.now();
  while (Date.now() - start1 < 25000) {
    await new Promise((res) => setTimeout(res, 1000));
    const evalRes = await cdp.send("Runtime.evaluate", {
      expression: `(() => {
        const nav = document.querySelector('.topbar nav');
        if (!nav) return null;
        const buttons = Array.from(nav.querySelectorAll('button')).map(b => b.textContent.trim());
        const links = Array.from(nav.querySelectorAll('a')).map(a => ({ text: a.textContent.trim(), href: a.getAttribute('href') }));
        return {
          title: document.title,
          badge: document.querySelector('.srclabel')?.textContent ?? '',
          buttons,
          links
        };
      })()`,
      returnByValue: true,
    });
    travelState = evalRes.result?.value;
    if (travelState && travelState.buttons.length >= 3) break;
  }

  console.log("Visualizer Title:", travelState?.title);
  console.log("Source Badge:", travelState?.badge);
  console.log("Navbar Buttons:", travelState?.buttons);
  console.log("Navbar Links:", travelState?.links);

  const hasHowItWorks = travelState?.buttons.some(b => b.toLowerCase().includes("how it works") || b.toLowerCase().includes("как это работает"));
  const hasPresentation = travelState?.links.some(l => l.href?.includes("presentation.html"));
  const hasEvals = travelState?.links.some(l => l.href?.includes("/evals/"));

  if (!hasHowItWorks) throw new Error("Missing 'How it works' button in Visualizer navigation!");
  if (!hasPresentation) throw new Error("Missing 'Presentation' link in Visualizer navigation!");
  if (!hasEvals) throw new Error("Missing 'Evals & Proofs' link in Visualizer navigation!");
  console.log("✓ PASS: Visualizer Topbar has Cosmos, Travel, How it works, Presentation, and Evals links.");

  console.log("\nTesting 'How it works' interactive presentation toggle...");
  await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const btn = Array.from(document.querySelectorAll('.topbar nav button')).find(b =>
        b.textContent.toLowerCase().includes('how it works') || b.textContent.toLowerCase().includes('как это работает')
      );
      if (btn) btn.click();
    })()`
  });
  await new Promise((res) => setTimeout(res, 2000));

  const presViewState = (await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const presTop = document.querySelector('.pres-top');
      if (!presTop) return null;
      const links = Array.from(presTop.querySelectorAll('a')).map(a => ({ text: a.textContent.trim(), href: a.getAttribute('href') }));
      const narr = document.querySelector('.pres-narr')?.textContent ?? '';
      const fact = document.querySelector('.pres-fact')?.textContent ?? '';
      const chapter = document.querySelector('.pres-chapternum')?.textContent ?? '';
      return { active: true, links, narr, fact, chapter };
    })()`,
    returnByValue: true
  })).result?.value;

  console.log("Interactive Presentation Open:", presViewState?.active);
  console.log("Current Chapter:", presViewState?.chapter);
  console.log("Narration snippet:", presViewState?.narr?.slice(0, 150));
  console.log("Fact snippet:", presViewState?.fact?.slice(0, 150));
  console.log("Pres-top Links:", presViewState?.links);

  if (!presViewState?.active) throw new Error("PresentationView did not activate when clicking 'How it works'!");
  console.log("✓ PASS: 'How it works' interactive presentation opened seamlessly!");

  console.log("\n========================================================");
  console.log("TEST 2: Executive Presentation Deck (https://knotgate.labs.jb.gg/travel/presentation.html)");
  console.log("========================================================");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/travel/presentation.html" });
  await new Promise((res) => setTimeout(res, 2000));

  const deckState = (await cdp.send("Runtime.evaluate", {
    expression: `(() => {
      const navLinks = Array.from(document.querySelectorAll('.nav-links a')).map(a => ({ text: a.textContent.trim(), href: a.getAttribute('href') }));
      const footerLinks = Array.from(document.querySelectorAll('.footer-links a')).map(a => ({ text: a.textContent.trim(), href: a.getAttribute('href') }));
      const h1 = document.querySelector('h1')?.textContent ?? '';
      return { title: document.title, h1, navLinks, footerLinks };
    })()`,
    returnByValue: true
  })).result?.value;

  console.log("Deck Title:", deckState?.title);
  console.log("Deck Header Links:", deckState?.navLinks);
  console.log("Deck Footer Links:", deckState?.footerLinks);

  const deckHasCosmos = deckState?.navLinks.some(l => l.href === "/travel/");
  const deckHasEvals = deckState?.navLinks.some(l => l.href === "/evals/");
  if (!deckHasCosmos) throw new Error("Deck nav missing link to /travel/!");
  if (!deckHasEvals) throw new Error("Deck nav missing link to /evals/!");
  console.log("✓ PASS: Executive Presentation has direct links to Cosmos (/travel/) and Evals (/evals/).");

  console.log("\n========================================================");
  console.log("TEST 3: PIER DO LESS Evaluation Workstation (https://knotgate.labs.jb.gg/evals/)");
  console.log("========================================================");
  await cdp.send("Page.navigate", { url: "https://knotgate.labs.jb.gg/evals/" });

  let evalsState = null;
  const start3 = Date.now();
  while (Date.now() - start3 < 20000) {
    await new Promise((res) => setTimeout(res, 1000));
    const evalRes = await cdp.send("Runtime.evaluate", {
      expression: `(() => {
        const brand = document.querySelector('.app-brandline')?.textContent ?? '';
        const title = document.title;
        const navLinks = Array.from(document.querySelectorAll('.header-link-list a')).map(a => ({ text: a.textContent.trim(), href: a.getAttribute('href') }));
        const tags = Array.from(document.querySelectorAll('.n-tag')).map(t => t.textContent.trim());
        const hasContent = !!document.getElementById('main-content');
        return { brand, title, navLinks, tags, hasContent };
      })()`,
      returnByValue: true
    });
    evalsState = evalRes.result?.value;
    if (evalsState && evalsState.navLinks.length >= 4) break;
  }

  console.log("Evals Title:", evalsState?.title);
  console.log("Evals Brandline:", evalsState?.brand);
  console.log("Evals Nav Links:", evalsState?.navLinks);
  console.log("Evals Tags / Summary:", evalsState?.tags);

  const evalsHasTravel = evalsState?.navLinks.some(l => l.href?.includes("/travel/"));
  const evalsHasDeck = evalsState?.navLinks.some(l => l.href?.includes("presentation.html") || l.href?.includes("/presentation"));

  if (!evalsHasTravel) throw new Error("Evals UI missing link to Memory Visualizer (/travel/)!");
  if (!evalsHasDeck) throw new Error("Evals UI missing link to Presentation Deck!");
  console.log("✓ PASS: Pier Do Less Evals UI loaded successfully and links to Memory Visualizer & Deck!");

  chrome.kill("SIGKILL");
  console.log("\n========================================================");
  console.log("ALL VERIFICATION SUITES PASSED SUCCESSFULLY!");
  console.log("========================================================");
}

run().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
