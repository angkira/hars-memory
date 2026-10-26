#!/usr/bin/env node
// Guard: the how-it-works deck must be LIVE-only.
//
// The deck runs a real `memory_recall` for the question the presenter types and
// replays that trace. It must never read the baked snapshot (app/public/data:
// ingest_fixture.json, traces/*) — otherwise a deck could silently present
// someone else's pre-recorded data again. This check fails the build if any
// presentation module imports the fixture loaders or names those files.
//
// Run standalone:  node scripts/check-deck-is-live.mjs
import { readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(ROOT, "app", "src");

const FILES = [
  "components/PresentationView.tsx",
  "components/PresDataPanel.tsx",
  "components/PresNarrationPanel.tsx",
  "lib/liveRecall.ts",
  "lib/presentation.ts",
];

const FORBIDDEN = [
  { re: /from\s+["'][^"']*\/data["']/, why: "imports the fixture loader module (lib/data)" },
  { re: /\bloadTrace\s*\(/, why: "calls loadTrace (public/data/traces)" },
  { re: /\bloadIngestFixture\b/, why: "reads ingest_fixture.json" },
  { re: /ingest_fixture/, why: "references ingest_fixture.json" },
  { re: /data\/traces/, why: "references public/data/traces" },
  { re: /\bDEMO_INDEX_ID\b/, why: "references the synthetic demo identity" },
];

let bad = 0;
for (const rel of FILES) {
  const path = join(SRC, rel);
  const text = readFileSync(path, "utf8");
  const lines = text.split("\n");
  for (const { re, why } of FORBIDDEN) {
    lines.forEach((line, i) => {
      if (!re.test(line)) return;
      // the guard may name the patterns itself; skip comments that explain the rule
      if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;
      bad++;
      console.error(`deck-is-live: ${relative(ROOT, path)}:${i + 1} ${why}`);
      console.error(`  ${line.trim()}`);
    });
  }
}

if (bad) {
  console.error(`\ndeck-is-live: ${bad} violation(s) — the deck must run only on a live recall.`);
  process.exit(1);
}
console.log(`deck-is-live: clean (${FILES.length} presentation modules, no fixture path)`);
