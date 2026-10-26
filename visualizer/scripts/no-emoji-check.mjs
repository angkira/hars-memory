// no-emoji-check — the project renders ZERO emoji / pictographs.
//
// Every glyph in the UI is a monochrome inline SVG (app/src/components/icons.tsx,
// currentColor, no colour font, no dependency). This guard makes that
// non-negotiable: `npm run build` (prebuild) and `scripts/smoke.mjs` both call
// it, so a pasted emoji cannot reach the bundle — in TSX, in CSS content, in
// the RU/EN i18n strings, in chapter narration copy, or in the data fixtures
// the presentation renders verbatim.
//
// What counts as a hit (ranges unicodedata calls Emoji with the
// Emoji_Presentation property, plus the Dingbats/Misc-Symbols/Geometric-Shapes
// blocks that font-fallback to colour emoji in Chrome and Safari):
//
//   U+00A9 U+00AE U+203C U+2049 U+2122 U+2139      legacy emoji presentation
//   U+2194–U+21AA                                  emoji arrows (↔ ↕ ⇄ …)
//   U+231A–U+23FF                                  watches, hourglass, ⏸ ⏹ ⏺
//   U+2460–U+24FF                                  enclosed alphanumerics (① …)
//   U+25A0–U+25FF                                  geometric shapes (▶ ◀ ● ◆ …)
//   U+2600–U+27BF                                  Misc Symbols + Dingbats (⚙ ✕ ✔ ★ …)
//   U+2934 U+2935 U+2B00–U+2BFF U+3030 U+303D     arrows/stars that render colour
//   U+3297 U+3299                                  CJK "congratulation"/"secret"
//   U+FE0F                                         emoji variation selector
//   U+1F000–U+1FAFF, U+1F1E6–U+1F1FF               emoji, pictographs, flags
//
// DELIBERATELY NOT a hit — typographic glyphs the copy legitimately uses, all of
// which render as plain text glyphs on every platform:
//
//   ← → ↑ ↓ (U+2190–U+2193, prose/keys), · (U+00B7), — – … « » „ " ' ’ ›
//   × (U+00D7), ± ≤ ≥ ≈ α, § (U+00A7), and box-drawing/ASCII art (U+2500–U+257F)
//   inside real document text in app/public/data (verbatim source notes).
//
// Usage:  node scripts/no-emoji-check.mjs [--quiet]
// Exit 1 on any hit, 0 when clean. Stdlib only (node:fs / node:path).

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const EMOJI = /[\u00A9\u00AE\u203C\u2049\u2122\u2139\u2194-\u21AA\u231A-\u23FF\u2460-\u24FF\u25A0-\u25FF\u2600-\u27BF\u2934\u2935\u2B00-\u2BFF\u3030\u303D\u3297\u3299\uFE0F\u{1F000}-\u{1FAFF}\u{1F1E6}-\u{1F1FF}]/u;
// same class, global, for counting several hits on one line
const EMOJI_G = new RegExp(EMOJI.source, "gu");

// Data fixtures are the one place a flagged codepoint can be legitimate: the
// full text of a real note carries markdown ASCII-art diagrams whose arrowheads
// and rules are geometric shapes (U+25A0–U+25FF: ▼ ├ ─ …). Those are verbatim
// document bytes, not UI glyphs — so the data sweep uses the emoji class
// WITHOUT the geometric-shapes block. UI sources (app/src, index.html) are
// scanned with the full class, an emoji pasted into a component still fails.
const EMOJI_DATA = new RegExp(EMOJI.source.replace("\\u25A0-\\u25FF", ""), "gu");

// Text we ship: source, styles, entry HTML, and the data the UI prints verbatim.
const TARGETS = [
  { dir: join(ROOT, "app", "src"), ext: [".ts", ".tsx", ".css", ".html"], re: EMOJI_G },
  { dir: join(ROOT, "app", "public", "data"), ext: [".json"], re: EMOJI_DATA },
];
const SINGLE_FILES = [join(ROOT, "app", "index.html")];

function walk(dir, ext, out = []) {
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    if (name === "node_modules" || name === "dist" || name.startsWith(".")) continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, ext, out);
    else if (ext.some((e) => name.endsWith(e))) out.push(p);
  }
  return out;
}

const files = [
  ...TARGETS.flatMap(({ dir, ext }) => walk(dir, ext).map((f) => ({ f, re: TARGETS.find((t) => f.startsWith(t.dir)).re }))),
  ...SINGLE_FILES.map((f) => ({ f, re: EMOJI_G })),
];

const hits = [];
for (const { f: file, re } of files) {
  const lines = readFileSync(file, "utf8").split("\n");
  lines.forEach((line, i) => {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(line)) !== null) {
      const ch = m[0];
      hits.push({
        file: relative(ROOT, file),
        line: i + 1,
        col: m.index + 1,
        ch,
        cp: `U+${ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}`,
      });
    }
  });
}

if (!process.argv.includes("--quiet")) {
  console.log(`no-emoji: scanned ${files.length} files (app/src, app/index.html, app/public/data)`);
}

if (hits.length) {
  for (const h of hits) {
    console.error(`  ${h.file}:${h.line}:${h.col}  ${h.cp} ${JSON.stringify(h.ch)}`);
  }
  console.error(`\nno-emoji: ${hits.length} HIT(S) — replace with a monochrome icon from app/src/components/icons.tsx`);
  process.exit(1);
}
console.log("no-emoji: clean (0 hits)");
