// Entity-type colour system for Memory Travel.
//
// The index ships 44 distinct `t` strings (concept, tool, jobrun,
// wrong_episode, …). A 44-hue legend is unreadable, so types are grouped
// into 12 semantic families, each with one stable hue. Both renderers
// (WebGPU + Canvas2D fallback) and the legend consume the same two
// functions, so a colour in the legend is the colour on the canvas.

export const FAMILY_COLORS: Record<string, string> = {
  // Desaturated deep tones (hsl sat 0.22–0.48, L≈0.60–0.70): the WebGL2
  // starfield draws with AdditiveBlending, which brightens overlaps
  // naturally — vivid bases turned the old map into eye-searing neon.
  docs: "#86a4d5", // documents, specs
  knowledge: "#a292d3", // concepts, findings, decisions, hypotheses
  engineering: "#72c1ca", // components, tools, encoders, agents
  models: "#cb86a8", // models, backbones, adapters
  data: "#74be86", // datasets, artifacts, checkpoints, content
  work: "#cfaf77", // tasks, plans, pipelines, job runs
  results: "#7fbd8c", // metrics, reports, analyses, tests
  experiments: "#ca7f68", // experiments
  config: "#b2c384", // configs, architecture, programs
  people: "#cd8484", // people, organizations, locations
  events: "#8d95ce", // events
  other: "#899ab4", // everything else (incl. UNKNOWN)
};

/** type string -> family key. Explicit for all 44 types seen in the index. */
const TYPE_FAMILY: Record<string, string> = {
  document: "docs",
  concept: "knowledge",
  hypothesis: "knowledge",
  finding: "knowledge",
  decision: "knowledge",
  component: "engineering",
  tool: "engineering",
  encoder: "engineering",
  adapter: "engineering",
  agent: "engineering",
  process: "engineering",
  model: "models",
  backbone: "models",
  data: "data",
  dataset: "data",
  artifact: "data",
  content: "data",
  checkpoint: "data",
  parameter: "data",
  task: "work",
  plan: "work",
  phase: "work",
  pipeline: "work",
  jobrun: "work",
  processrun: "work",
  monitor: "work",
  metric: "results",
  report: "results",
  analysis: "results",
  test: "results",
  risk: "results",
  experiment: "experiments",
  config: "config",
  architecture: "config",
  program: "config",
  person: "people",
  organization: "people",
  location: "people",
  event: "events",
  method: "engineering",
  other: "other",
  unknown: "other",
  wrong_episode: "other",
  naturalobject: "data",
};

export function familyOf(type: string): string {
  // "" guard: draw can run while a load() is still filling the types array
  return TYPE_FAMILY[(type ?? "").toLowerCase()] ?? "other";
}

export function colorForType(type: string): string {
  return FAMILY_COLORS[familyOf(type)] ?? FAMILY_COLORS.other;
}

/** "#rrggbb" -> [r,g,b,a] floats 0..1 (WebGPU path). */
export function hexToRGBA(hex: string, a = 1): [number, number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, a];
}

export const LEGEND_ORDER = Object.keys(FAMILY_COLORS);

/** Legend label per family (short, human). */
export const FAMILY_LABEL: Record<string, string> = {
  docs: "documents",
  knowledge: "knowledge",
  engineering: "engineering",
  models: "models",
  data: "data",
  work: "work & runs",
  results: "results & tests",
  experiments: "experiments",
  config: "config",
  people: "people & orgs",
  events: "events",
  other: "other",
};
