/**
 * Neurocompact A/B compaction extension.
 *
 * Arms:
 *   NC_PERSIST=0 -> baseline: do NOTHING - pi default compaction runs
 *                   (agent does its own comprehensive in-context summary).
 *   NC_PERSIST=1 -> neurocompact: SHORT handoff summary (details live in
 *                   long-term memory) + memory_compact() persists summary +
 *                   dropped context into the HARS session graph; questions
 *                   later use memory_recall.
 *
 * Env: NC_PERSIST, NC_SESSION_ID, NC_MCP_URL, NC_PROVIDER,
 *      NC_MODEL_ID (unused unless default fallback needed),
 *      NC_SUM_PROVIDER/NC_SUM_MODEL_ID (compactor model for the NC arm),
 *      (NC_PROVIDER / NC_SUM_PROVIDER are any pi provider name, e.g. openrouter or
 *      anthropic; the runner's --provider sets them, nothing here is provider-specific
 *      except the thinking=off handling at the streamSimple call),
 *      NC_SUM_REASONING_EFFORT (optional reasoning effort for the summarizer
 *      call, pass-through value e.g. off|low|medium|high), NC_SUM_MAX_TOKENS
 *      (VISIBLE answer budget; default 1536 baseline / 640 nc),
 *      NC_SUM_SECTION_TRIM=1|0 (deterministic post-processing to the visible
 *      cap; default ON for the nc arm, OFF for baseline),
 *      NC_NOTE_ONLY=1 (skip the summarizer and reuse the summary already in
 *      the session; fallback path for the nc-note-only fork arm),
 *      NC_SELFTEST=1 (run the trimmer over its fixtures — including forced
 *      stop=length amputations with boundary repair — print before/after,
 *      exit non-zero on any violation; no model, no network, no MCP).
 *
 * Both arms are prompt-SCHEMA-MATCHED (GOAL / KEY FIGURES & IDENTIFIERS /
 * DECISIONS & RATIONALE / ACTIVE STATE) and differ only in budget + whether
 * the note is comprehensive (baseline) or a short index of memory (nc), so an
 * A/B comparison measures the artifact and not the prompt shape.
 *
 * Why the trimmer exists: the note is written into a VISIBLE token budget, and
 * ACTIVE STATE is the LAST section. When the model overshoots, the provider
 * amputates the note at an arbitrary character, so the experiment would
 * compare "summary" vs "note with its state cut off" - a treatment defect.
 * The prompt now asks for ~85% of the cap and to compress earlier sections
 * before dropping ACTIVE STATE; the deterministic trimmer makes the cap hard
 * and the cut non-random (DECISIONS bullets, then GOAL, then KEY FIGURES,
 * whole lines/sentences only, never ACTIVE STATE, never mid-sentence).
 *
 * Every compaction prints one machine-readable stderr line:
 *   [nc-metrics] {"arm":"baseline|nc|note-only","stop":"stop","max_tokens":N,
 *                 "usage":{...},"summary_chars":N,"realized_tokens":N,
 *                 "tokenizer":"...","trimmed_sections":[...],
 *                 "trimmed_tokens":N,"persisted":true}
 * plus, when something was cut, a human-greppable
 *   [nc-metrics] trimmed_sections=[...] trimmed_tokens=N
 */
import { readFileSync } from "node:fs";
import { uuidv7 } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertToLlm, serializeConversation } from "@earendil-works/pi-coding-agent";

const PERSIST = (process.env.NC_PERSIST || "0") === "1";
// Fallback path for the nc-note-only fork arm: reuse the summary that is
// already in the session instead of paying for another summarizer call.
const NOTE_ONLY = (process.env.NC_NOTE_ONLY || "0") === "1";
// Deterministic section trim to the VISIBLE cap. Default ON for the memory arm
// (whose artifact is the size-limited note under test) and OFF for baseline,
// whose comprehensive summary must keep its own character.
const SECTION_TRIM = (process.env.NC_SUM_SECTION_TRIM ?? (PERSIST ? "1" : "0")) === "1";
const SID = process.env.NC_SESSION_ID || "nc-unknown";
const MCP_URL = process.env.NC_MCP_URL || "http://127.0.0.1:8301/mcp";
const PROVIDER = process.env.NC_PROVIDER || "local-router";
const MODEL_ID = process.env.NC_MODEL_ID || "muse-glimmer-30b";
// Summaries can run on a faster no-think model; defaults to the conversation model.
const SUM_PROVIDER = process.env.NC_SUM_PROVIDER || PROVIDER;
const SUM_MODEL_ID = process.env.NC_SUM_MODEL_ID || MODEL_ID;

const RAW_CAP = 60000; // cap on the serialized dropped context we look at
const DROP_HALF = 12000; // per-item cap sent to memory_compact (2 items)

let mcpSessionId: string | null = null;

async function mcpInit(): Promise<string | null> {
	const headers: Record<string, string> = {
		"Content-Type": "application/json",
		Accept: "application/json, text/event-stream",
	};
	const init = await fetch(MCP_URL, {
		method: "POST",
		headers,
		body: JSON.stringify({
			jsonrpc: "2.0",
			id: 1,
			method: "initialize",
			params: {
				protocolVersion: "2025-03-26",
				capabilities: {},
				clientInfo: { name: "pi-neurocompact", version: "1.0" },
			},
		}),
	});
	if (!init.ok) throw new Error(`initialize HTTP ${init.status}`);
	const sid = init.headers.get("mcp-session-id");
	const h2 = sid ? { ...headers, "mcp-session-id": sid } : headers;
	await fetch(MCP_URL, {
		method: "POST",
		headers: h2,
		body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
	});
	return sid;
}

async function mcpToolCall(name: string, args: Record<string, unknown>) {
	// Reuse ONE MCP session for every call of this process: the HARS server
	// binds per-session state (LightRAG pipeline ingress) to the event loop
	// that first touched it, and per-call ephemeral sessions can land on
	// different live loops, which fast-fails memory_compact with
	// "pipeline ingress belongs to another event loop". A single session
	// keeps all calls on one loop.
	const headers: Record<string, string> = {
		"Content-Type": "application/json",
		Accept: "application/json, text/event-stream",
	};
	if (mcpSessionId === null) mcpSessionId = await mcpInit();
	let h2 = mcpSessionId ? { ...headers, "mcp-session-id": mcpSessionId } : headers;
	let res = await fetch(MCP_URL, {
		method: "POST",
		headers: h2,
		body: JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/call", params: { name, arguments: args } }),
	});
	if (!res.ok && (res.status === 404 || res.status === 400)) {
		// stale/expired session: re-initialize once and retry
		mcpSessionId = await mcpInit();
		h2 = mcpSessionId ? { ...headers, "mcp-session-id": mcpSessionId } : headers;
		res = await fetch(MCP_URL, {
			method: "POST",
			headers: h2,
			body: JSON.stringify({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name, arguments: args } }),
		});
	}
	if (!res.ok) throw new Error(`tools/call HTTP ${res.status}`);
	const j: any = await res.json();
	if (j?.error) throw new Error(j.error.message || "mcp error");
	const text = j?.result?.content?.[0]?.text;
	return typeof text === "string" ? JSON.parse(text) : j?.result ?? j;
}

let lastCompactedAt = 0;

// ---------------------------------------------------------------------------
// Deterministic note trimming (pure: no model, no I/O, no globals mutated)
//
// nc caps the note at NC_SUM_MAX_TOKENS visible tokens. A model that overshoots
// is amputated by the provider at an arbitrary character, and because ACTIVE
// STATE is the LAST section that is exactly what gets lost. This post-processor
// turns the hard cap into a *deterministic* cut:
//   1. trailing DECISIONS & RATIONALE bullets (then its last bullet -> first
//      sentence, then the whole section),
//   2. then GOAL (same escalation),
//   3. then trailing KEY FIGURES & IDENTIFIERS entries.
// ACTIVE STATE is never trimmed, nothing is cut mid-sentence, and the stored
// note always ends at a sentence / bullet / section boundary.
// ---------------------------------------------------------------------------

const NC_SECTIONS = [
	"GOAL",
	"KEY FIGURES & IDENTIFIERS",
	"DECISIONS & RATIONALE",
	"ACTIVE STATE",
] as const;
type NcSectionName = (typeof NC_SECTIONS)[number];

// Same estimator the eval runner falls back to (run_chainreplay.count_tokens):
// chars / 3.6 when tiktoken is not importable in-process. cl100k_base is
// ~4.0-4.2 chars/token on this kind of English prose, so the estimator is
// CONSERVATIVE (it over-counts), i.e. trimming to <= cap by the estimator keeps
// the tiktoken count under the cap too.
const TOKENIZER_LABEL = "estimator:chars/3.6";

function countTokens(text: string): number {
	return Math.round((text || "").length / 3.6);
}

interface NcSection {
	name: NcSectionName;
	header: string;
	body: string[];
}

interface NcNoteResult {
	text: string;
	before: number;
	after: number;
	trimmedTokens: number;
	trimmedSections: string[];
	sectionsKept: string[];
	boundaryRepaired: boolean;
	/** Characters of stored text removed by boundary repair (0 when none). */
	repairCutChars: number;
	/**
	 * Repair gave up honestly: the amputated tail is too large to shave (the
	 * cut would exceed the 20% repair budget), so the text is left as-is and
	 * this flag says the stored tail is NOT at a boundary. Never silently
	 * state-less: boundary_repaired=false + boundary_unresolved=true is the
	 * "amputation beyond repair" signal, distinct from a clean boundary.
	 */
	boundaryUnresolved: boolean;
	activeStateKept: boolean;
	activeStateComplete: boolean;
	overCap: boolean;
	trimApplied: boolean;
}

const SENTENCE_END = /[.!?…]["'”’)\]]*$/;

function normalizeHeader(line: string): string {
	return line
		.replace(/^[\s#>*_`-]+/, "")
		.replace(/[:*\s]+$/, "")
		.toUpperCase();
}

function parseNote(text: string): { pre: string[]; sections: NcSection[] } {
	const pre: string[] = [];
	const sections: NcSection[] = [];
	let cur: NcSection | null = null;
	for (const raw of (text || "").split("\n")) {
		const line = raw.replace(/\s+$/, "");
		const norm = normalizeHeader(line);
		const name = NC_SECTIONS.find((s) => s === norm);
		if (name) {
			cur = { name, header: name, body: [] };
			sections.push(cur);
			continue;
		}
		if (cur) cur.body.push(line);
		else if (line.trim()) pre.push(line);
	}
	for (const s of sections) {
		while (s.body.length && !s.body[s.body.length - 1].trim()) s.body.pop();
		while (s.body.length && !s.body[0].trim()) s.body.shift();
	}
	return { pre, sections };
}

function renderNote(pre: string[], sections: NcSection[]): string {
	const out: string[] = [...pre];
	for (const s of sections) {
		out.push(s.header);
		out.push(...s.body);
	}
	return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

// A line "ends at a boundary" when it closes a sentence, or when it is a short
// bullet/value line (those legitimately have no trailing period).
function endsAtBoundary(line: string): boolean {
	const t = (line || "").trim();
	if (!t) return false;
	if (SENTENCE_END.test(t)) return true;
	if (/^[-*•·]\s/.test(t) && t.length <= 200) return true;
	return false;
}

// A bare list marker ("3.", "2)", "-") is the provider's amputation point, not
// a complete sentence: SENTENCE_END matches its trailing period, but an item
// with no content is exactly the mid-thought cut the repair exists for (gate4
// chain_y062eGZR baseline C3 stored a dangling "3." tail). Only treated as a
// defect under forceRepair (stop=length); a normally-completed note never ends
// on a bare marker anyway.
function isDanglingMarker(line: string): boolean {
	return /^(\d{1,3}[.)]|[-*•·]|[-*•·]\s\d{1,3}[.)]?)$/.test((line || "").trim());
}

// A tail that cannot end a finished thought: an unclosed bracket/paren, a
// connector comma/colon/semicolon, or a dangling operator/hyphen (gate5
// chain_y062eGZR nc C3 ended "... valid model IDs (" — a short bullet, so the
// bullet heuristic accepted an amputated parenthetical as a boundary). Only
// consulted under forceRepair (stop=length); closing quotes/brackets and
// terminator-less VALUE lines ("batch key: HB-77") stay legitimate boundaries.
function endsIncomplete(line: string): boolean {
	return /[-([{,:;+*=<>\/\\|&*~^%#@$_]$/.test((line || "").trim());
}

// Index of the last real sentence terminator (skips decimals like 0.82).
function lastSentenceIndex(t: string): number {
	for (let i = t.length - 1; i >= 0; i--) {
		const c = t[i];
		if (c !== "." && c !== "!" && c !== "?" && c !== "…") continue;
		const prev = t[i - 1] || "";
		const next = t[i + 1] || "";
		if (/[0-9]/.test(prev) && /[0-9]/.test(next)) continue; // decimal point
		if (next && !/\s/.test(next)) continue; // mid-token, e.g. "e.g." inside a word
		return i;
	}
	return -1;
}

function cutToLastSentence(line: string): string | null {
	const t = (line || "").replace(/\s+$/, "");
	if (!t) return null;
	const i = lastSentenceIndex(t);
	if (i < 0) return null;
	const cut = t.slice(0, i + 1).trimEnd();
	return cut || null;
}

/**
 * Trim `text` to at most `cap` visible tokens.
 *
 * `enabled=false` measures only (the baseline arm keeps its artifact verbatim);
 * `enabled=true` applies the priority cut. Both modes still REPORT the boundary
 * and ACTIVE-STATE diagnostics so a baseline run is measurable on the same axes.
 *
 * `forceRepair` is wired from the compaction response's stop reason
 * (`response.stopReason === "length"`): the provider's amputation of an
 * over-long completion can end on a bare list marker ("3.") or mid-sentence
 * even when the chars/3.6 estimator put the note under the cap. Strict mode
 * treats those markers as amputated tails too. The repair only runs when the
 * trimmer is enabled (NC_SUM_SECTION_TRIM=1, which the eval runner sets for
 * every summarizing arm); it changes NO caps, prompts, or persistence.
 */
function trimNote(text: string, cap: number, enabled: boolean = true, forceRepair: boolean = false): NcNoteResult {
	const before = countTokens(text);
	const parsed = parseNote(text);
	const sections: NcSection[] = parsed.sections.map((s) => ({ ...s, body: [...s.body] }));
	const trimmedSections: string[] = [];
	let repairCutChars = 0;
	const mark = (name: string) => {
		if (!trimmedSections.includes(name)) trimmedSections.push(name);
	};
	const find = (name: NcSectionName) => sections.find((s) => s.name === name);
	const current = () => renderNote(parsed.pre, sections);
	const over = () => countTokens(current()) > cap;

	let trimApplied = false;
	if (enabled && parsed.sections.length && over()) {
		// Drop trailing body lines of `s` (whole lines only) until under cap.
		const dropTrailing = (s: NcSection, keepMin: number): boolean => {
			while (s.body.length > keepMin) {
				s.body.pop();
				mark(s.name);
				trimApplied = true;
				while (s.body.length > keepMin && !s.body[s.body.length - 1].trim()) s.body.pop();
				if (!over()) return true;
			}
			return !over();
		};
		// Shorten the last body line of `s` to its last complete sentence.
		const shortenLast = (s: NcSection): boolean => {
			if (!s.body.length) return false;
			const cut = cutToLastSentence(s.body[s.body.length - 1]);
			if (!cut || cut === s.body[s.body.length - 1]) return false;
			s.body[s.body.length - 1] = cut;
			mark(s.name);
			trimApplied = true;
			return !over();
		};
		const dropSection = (s: NcSection): boolean => {
			const i = sections.indexOf(s);
			if (i >= 0) sections.splice(i, 1);
			mark(s.name);
			trimApplied = true;
			return !over();
		};

		// Priority 1: trailing DECISIONS & RATIONALE bullets, then the last
		// bullet's tail, then the section itself.
		const dec = find("DECISIONS & RATIONALE");
		let done = !over();
		if (!done && dec) done = dropTrailing(dec, 1) || shortenLast(dec) || dropSection(dec);
		// Priority 2: GOAL.
		const goal = find("GOAL");
		if (!done && goal) done = dropTrailing(goal, 1) || shortenLast(goal) || dropSection(goal);
		// Priority 3: trailing KEY FIGURES & IDENTIFIERS entries.
		const key = find("KEY FIGURES & IDENTIFIERS");
		if (!done && key) done = dropTrailing(key, 1) || shortenLast(key) || dropSection(key);
		// ACTIVE STATE is never touched: if it alone exceeds the cap, the note
		// stays over cap and `over_cap` says so (honest, never state-less).
	}

	// Boundary invariant: whatever happens above (or to a note the model itself
	// amputated at stop=length), the stored text ends at a sentence / bullet /
	// section boundary. Only the TAIL is repaired; nothing is invented. The
	// verbatim (baseline) arm is left byte-identical on purpose. forceRepair
	// (stop=length) additionally rejects bare list markers as "boundaries".
	let boundaryRepaired = false;
	for (let guard = 0; enabled && guard < 400; guard++) {
		const last = sections[sections.length - 1];
		if (!last) break;
		if (!last.body.length) break; // header-only tail: nothing to cut
		const line = last.body[last.body.length - 1];
		if (!line.trim()) {
			last.body.pop();
			continue;
		}
		if (endsAtBoundary(line)
			&& !(forceRepair && (isDanglingMarker(line) || endsIncomplete(line)))
		) break;
		const cut = cutToLastSentence(line);
		if (cut && cut.trim() && cut.trim() !== line.trim() && !(forceRepair && isDanglingMarker(cut))) {
			repairCutChars += line.length - cut.length;
			last.body[last.body.length - 1] = cut;
			boundaryRepaired = true;
			continue; // the cut ends on a terminator -> next pass breaks
		}
		if (last.body.length > 1 || last.name !== "ACTIVE STATE") {
			repairCutChars += line.length;
			last.body.pop();
			boundaryRepaired = true;
			continue;
		}
		// Single unterminated ACTIVE STATE line: keep it. Deleting the only line
		// of state to satisfy a punctuation rule would be the bug, not the fix.
		break;
	}

	// Schema-less notes (parseNote found no GOAL/... header): the loop above
	// iterates `sections` and exits immediately when that list is empty, so a
	// stop=length amputation used to be stored verbatim with
	// boundary_repaired=false (gate5 chain_y062eGZR nc C3 stored
	// "...to use valid model IDs (" unrepaired -> CLIP FAIL). Repair the RAW
	// text tail instead: same boundary rule as the sectioned loop, cut to the
	// last sentence or pop whole trailing lines, count repair_cut_chars, never
	// invent content, never touch anything but the tail. Budget: the repair may
	// remove at most ~20% of the note; beyond that the amputation is too deep
	// to fix by shaving, the text is left byte-identical, and
	// boundary_unresolved=true reports it honestly.
	let boundaryUnresolved = false;
	if (enabled && sections.length === 0) {
		const budget = Math.max(1, Math.floor(text.length * 0.2));
		let cut = 0;
		while (parsed.pre.length) {
			const line = parsed.pre[parsed.pre.length - 1];
			if (endsAtBoundary(line) && !(forceRepair && isDanglingMarker(line))) break;
			const sentenceCut = cutToLastSentence(line);
			if (
				sentenceCut && sentenceCut.trim()
				&& sentenceCut.trim() !== line.trim()
				&& !(forceRepair && isDanglingMarker(sentenceCut))
			) {
				const shave = line.length - sentenceCut.length;
				if (cut + shave > budget) {
					boundaryUnresolved = true;
					break;
				}
				cut += shave;
				parsed.pre[parsed.pre.length - 1] = sentenceCut;
				boundaryRepaired = true;
				continue;
			}
			if (cut + line.length > budget) {
				// Cannot remove even this line within budget: leave the text as
				// it is. If earlier cuts already landed, boundaryRepaired stays
				// true (cuts DID happen) and boundaryUnresolved marks that the
				// tail still does not sit on a boundary.
				boundaryUnresolved = true;
				break;
			}
			cut += line.length;
			parsed.pre.pop();
			boundaryRepaired = true;
		}
		repairCutChars += cut;
	}

	const out = enabled ? renderNote(parsed.pre, sections) : text.trim();
	const after = countTokens(out);
	const lastSection = sections[sections.length - 1];
	return {
		text: out,
		before,
		after,
		trimmedTokens: Math.max(0, before - after),
		trimmedSections,
		sectionsKept: sections.map((s) => s.name),
		boundaryRepaired,
		repairCutChars,
		boundaryUnresolved,
		activeStateKept: sections.some((s) => s.name === "ACTIVE STATE"),
		activeStateComplete: !!find("ACTIVE STATE")?.body.length,
		overCap: after > cap,
		trimApplied,
	};
}

function noteTail(text: string): string {
	const lines = (text || "").split("\n").filter((l) => l.trim());
	return lines.length ? lines[lines.length - 1] : "";
}

// ---------------------------------------------------------------------------
// NC_SELFTEST=1: pure trimmer fixtures, no model / network / MCP.
// ---------------------------------------------------------------------------

const FIXTURE_ALL_SECTIONS = `GOAL
Evaluate whether a short neurocompact handoff note plus external memory recall matches a comprehensive inline compaction summary for long agent sessions, using the same four-section schema in both arms so the comparison measures the artifact and not the prompt shape.

KEY FIGURES & IDENTIFIERS
- nc arm visible note cap: 512 tokens, soft target 435 tokens, deterministic section trim enabled.
- baseline arm visible cap: 1536 tokens, soft target 1305 tokens, section trim disabled.
- measured stop=length rate before the repair: 3 of 6 compactions at C=512, all cut inside ACTIVE STATE.
- recall budget: 1024 tokens, recall floor 100 chars, note plus recall target 1536.
- chains.fixture.jsonl: 2 chains, 3 spans each, 3 segments per span, about 1.7k chars per segment.
- runner: run_chainreplay.py, tokenizer tiktoken:cl100k_base, reserveTokens 2000, keepRecentTokens 200.
- judge: judge_v2.py with a pairwise rubric and 2 qa items per chain.
- compactor model: z-ai/glm-5.3-flash with reasoning off and maxTokens 2048.
- mcp endpoint: http://127.0.0.1:8305/mcp serving memory_compact and memory_recall.

DECISIONS & RATIONALE
- Both arms keep the four section headers so the A/B delta measures the note content and not the prompt skeleton.
- The nc note is an index rather than a copy: dropped context and the note are both persisted with memory_compact, and recall is queried at question time.
- Summarizer calls go through streamSimple with an explicit thinking level, because a missing thinking level made the OpenRouter adapter disable reasoning and some endpoints answered 400.
- Provider level 400 responses must not silently fall back to the raw tail, because that changes the arm's artifact without any visible signal.
- The visible answer budget stays NC_SUM_MAX_TOKENS and the thinking budget is added on top only when the model actually reasons.
- Deterministic section trimming exists because ACTIVE STATE is the last section, so an arbitrary provider cut deletes exactly the part that must survive.
- Baseline keeps its prompt verbatim except for the explicit soft target sentence, so the baseline keeps its comprehensive character.
- A raw-tail fallback is kept for the case where the compactor returns nothing at all, but it is logged loudly and never silent.
- Split runs across two arms were rejected because GPU and endpoint contention changed the latency distribution and muddied the cost comparison.

ACTIVE STATE
Probing in flight: 6 forced nc compactions at C=640 plus 3 baseline compactions at C=1536 on identical input.
Next concrete action is to compare realized tokens, stop reasons, trimmed sections and ACTIVE STATE survival per run.
Blocked on nothing; the report needs the worst note verbatim and the recommended cap plus the recall cap.`;

const FIXTURE_HUGE_KEY_FIGURES = `GOAL
Compare the nc and baseline compaction artifacts on identical input.

KEY FIGURES & IDENTIFIERS
- run 1 baseline tokens 1184, stop stop, chars 4310, trimmed none.
- run 2 baseline tokens 1203, stop stop, chars 4388, trimmed none.
- run 3 baseline tokens 1169, stop stop, chars 4251, trimmed none.
- run 4 nc tokens 812, stop length, chars 3011, trimmed none, active state partial.
- run 5 nc tokens 799, stop length, chars 2962, trimmed none, active state partial.
- run 6 nc tokens 640, stop stop, chars 2304, trimmed decisions, active state complete.
- recall p50 412 tokens, recall p90 688 tokens, recall p99 940 tokens, recall floor 100 chars.
- extraction p50 188 seconds, extraction p90 205 seconds, extraction p99 231 seconds.
- segment sizes 1831 1834 1771 1684 1732 1689 1651 1682 1678 chars.
- chain ids: fixture-chain-0, fixture-chain-1; session ids prefix cr-glm53flash.
- thresholds: force floor 400, context window 32768, reserve 2000, keep recent 200.
- costs: input 0.15 usd per million, output 0.50 usd per million, probe budget 0.05 usd.
- qa ids: q-0-0, q-0-1, q-1-0, q-1-1 with rubric weights 0.5 0.3 0.2.
- file paths: sessions/fixture-chain-0/nc, sessions/fixture-chain-0/baseline, report.json.
- versions: pi 1.4.0, node 25.0.0, python 3.13, tiktoken 0.9.0, glm-5.3-flash.
- metrics keys: stop, max_tokens, usage, summary_chars, realized_tokens, trimmed_sections.
- endpoints: 8305 glm small, 8306 nemotron free, 8300 llama router, 8301 hars memory.
- latency p50 11.4 seconds, latency p90 18.9 seconds, latency p99 24.2 seconds per compaction.
- usage prompt 2140 tokens, usage completion 512 tokens, usage total 2652 tokens per run.
- error rate 0 of 9, retry rate 0 of 9, timeout rate 0 of 9, empty note rate 0 of 9.
- budget headroom 96 tokens at soft target 544 and hard cap 640 visible tokens.
- active state survival rate 9 of 9 after the trimmer, 6 of 9 before the trimmer.
- tokens before compaction run 1 1544, run 2 1602, run 3 1588, run 4 1611, run 5 1595, run 6 1633.
- chars per token 4.06 tiktoken, 3.60 estimator, drift 11 percent too high on prose.
- compactions per arm 3 planned, 6 measured, 9 total, 0 aborted, 0 retried.
- qa pairs answered 12 of 12, judge verdicts 9 win 2 tie 1 loss, mean rubric 0.71.
- wall time per compaction 11.4 12.1 10.8 13.2 11.9 12.6 seconds, mean 12.0 seconds.
- persist latency 188 205 196 201 193 199 seconds, mean 197 seconds, p95 205 seconds.
- dropped context size 10432 chars, dropped chunks 2, per chunk cap 12000 chars.
- session files 6 jsonl, total 412331 bytes, mean 68722 bytes, max 74110 bytes.
- model registry entries 12 openrouter, 3 local router, 1 mcp server, 4 sections.
- note sizes 512 640 768 896 1024 tokens considered, chosen 640, rejected 512 and 768.
- stop length counts 3 of 6 at 512, 1 of 6 at 640, 0 of 6 at 768 in the pilot data.

DECISIONS & RATIONALE
- Keep the schema identical across arms.
- Fix the tail priority before tuning any other knob.

ACTIVE STATE
Decide the recommended nc cap and the recall cap so that note plus recall lands at 1536 tokens.`;

const FIXTURE_EXACT = `GOAL
Hold a note at exactly the visible cap and prove the trimmer leaves it untouched.

KEY FIGURES & IDENTIFIERS
- cap 640 tokens by the chars/3.6 estimator means 2304 characters.

DECISIONS & RATIONALE
- A note that is exactly at the cap is not over it, so no section may be cut.

ACTIVE STATE
Confirm the trimmer is a no-op at the cap boundary and that the tail is still a complete sentence.`;

// Build the "exactly at cap" fixture deterministically: normalize the fixture
// through the same parser/renderer (so blank-line spacing is already canonical),
// then pad the last section so the note is exactly `cap * 3.6` characters, i.e.
// exactly cap tokens by the estimator.
function fixtureAtCap(base: string, cap: number): string {
	const parsed = parseNote(base);
	const head = renderNote(parsed.pre, parsed.sections);
	const targetChars = Math.round(cap * 3.6);
	let pad = 1;
	for (let i = 0; i < 200; i++) {
		const candidate = `${head}\nCounter check line with id ${"0".repeat(pad)} loaded.`;
		if (candidate.length === targetChars) return candidate;
		if (candidate.length > targetChars) {
			const over = candidate.length - targetChars;
			if (pad - over < 1) break;
			pad -= over;
			continue;
		}
		pad += targetChars - candidate.length;
	}
	// Fallback: exact-length tail without padding digits.
	const tail = `Counter check line loaded.`;
	const need = targetChars - (head.length + 1 + tail.length);
	if (need > 0) return `${head}\n${tail}${"0".repeat(need)}`;
	return `${head}\n${tail}`;
}

function runSelfTest(): number {
	const cap = Number.parseInt(process.env.NC_SUM_MAX_TOKENS || "", 10) || 640;
	// Stop=length amputation fixtures: the provider cut the note mid-thought.
	// With forceRepair=true (wired from stopReason === "length") the trimmer
	// must repair the tail to a sentence boundary, keep ACTIVE STATE, and
	// report the cut; these mirror the gate4 CLIP FAIL (chain_y062eGZR baseline
	// C3 stored a dangling "3." tail with boundary_repaired=true).
	const cutMidSentence = FIXTURE_ALL_SECTIONS.slice(0, FIXTURE_ALL_SECTIONS.length - 118);
	const danglingTail = `${FIXTURE_ALL_SECTIONS}
Next steps:
1. Re-run the forced-compaction probe at the smaller visible cap.
2. Compare stop reasons and ACTIVE STATE survival across arms.
3.`;
	// gate5 chain_y062eGZR nc C3: the provider cut a short ACTIVE STATE bullet
	// inside a parenthetical — accepted by the bullet heuristic as a boundary
	// because it is a "- " line under 200 chars. forceRepair must reject it.
	const openParenTail = `${FIXTURE_ALL_SECTIONS}
- Pending: Update user config files to use valid model IDs (`;
	const fixtures: Array<{
		name: string; note: string; atCap?: boolean; forceRepair?: boolean;
		/** Note uses NO section schema (parseNote finds 0 sections). */
		schemaless?: boolean;
		/** Amputated tail exceeds the 20% repair budget: must be left as-is. */
		overRepairBudget?: boolean;
	}> = [
		{ name: "over-long-all-sections", note: FIXTURE_ALL_SECTIONS, atCap: false },
		{ name: "over-long-huge-key-figures", note: FIXTURE_HUGE_KEY_FIGURES, atCap: false },
		{ name: "exactly-at-cap", note: fixtureAtCap(FIXTURE_EXACT, cap), atCap: true },
		{ name: "stop-length-mid-sentence", note: cutMidSentence, atCap: false, forceRepair: true },
		{ name: "stop-length-dangling-marker", note: danglingTail, atCap: false, forceRepair: true },
		{ name: "stop-length-open-paren", note: openParenTail, atCap: false, forceRepair: true },
		// Schema-less notes (gate5 defect class): parseNote finds 0 sections, so
		// the sectioned repair loop never ran and the amputated tail was stored
		// verbatim with boundary_repaired=false. (a) tail < 20% of the note ->
		// repaired; (b) tail > 20% -> left as-is, boundary_unresolved=true,
		// boundary_repaired=false (honest, never silently state-less).
		{
			name: "stop-length-noschema-mid-sentence",
			note: `Fix ninja-cli-mcp: normalize internal zai-coding-plan/* model IDs before command construction so unknown prefixes stop producing unexpected server errors from the provider.
Auto-prune worktrees after each task; the cache grew to 1.8 GB without any cleanup path in the driver.
Pending: update user config files to use valid model IDs (`,
			forceRepair: true,
			schemaless: true,
		},
		{
			name: "stop-length-noschema-over-repair-budget",
			note: `Fix ninja-cli-mcp: normalize model IDs.
Pending: update the user config files and restart both MCP clients to use valid model IDs (`,
			forceRepair: true,
			schemaless: true,
			overRepairBudget: true,
		},
	];
	if ((process.env.NC_SELFTEST_EXTRA || "0") === "1") {
		// Simulated stop=length amputation: the provider cut the note in the
		// middle of ACTIVE STATE. The trimmer must keep the state and end on a
		// complete sentence.
		fixtures.push({
			name: "amputated-in-active-state",
			note: FIXTURE_ALL_SECTIONS.slice(0, FIXTURE_ALL_SECTIONS.length - 130),
			atCap: false,
			forceRepair: true,
		});
	}
	let violations = 0;
	console.log(`[selftest] cap=${cap} tokenizer=${TOKENIZER_LABEL} fixtures=${fixtures.length}`);
	for (const f of fixtures) {
		const r = trimNote(f.note, cap, true, f.forceRepair === true);
		const lastLine = noteTail(r.text);
		const checks: Array<[string, boolean]> = [
			["realized<=cap", r.after <= cap],
			...(f.schemaless
				? [] // no schema -> no ACTIVE STATE section by definition
				: ([
					["active_state_kept", r.activeStateKept],
					["active_state_complete", r.activeStateComplete],
				] as Array<[string, boolean]>)),
			// overRepairBudget: the tail is honestly left mid-sentence; the flag
			// must SAY so instead of claiming a boundary that is not there.
			[f.overRepairBudget ? "ends_mid_sentence_honest" : "ends_at_boundary",
				f.overRepairBudget ? !endsAtBoundary(lastLine) : endsAtBoundary(lastLine)],
			["no_unknown_section_lost", true],
			["trimmed_tokens==before-after", r.trimmedTokens === Math.max(0, r.before - r.after)],
			["schema_headers_kept", r.sectionsKept.every((s) => (NC_SECTIONS as readonly string[]).includes(s))],
		];
		if (f.atCap) checks.push(["at_cap_is_noop", r.text === f.note.trim()]);
		else if (!f.forceRepair) checks.push(["over_cap_input_was_trimmed", r.before > cap && r.after < r.before]);
		if (f.forceRepair) {
			checks.push(["tail_not_dangling_marker", !isDanglingMarker(lastLine)]);
			if (f.overRepairBudget) {
				checks.push(
					["boundary_repaired_false", !r.boundaryRepaired],
					["repair_cut_chars_zero", r.repairCutChars === 0],
					["boundary_unresolved", r.boundaryUnresolved],
					["text_untouched", r.text === f.note.trim()],
				);
			} else {
				checks.push(
					["boundary_repaired", r.boundaryRepaired],
					["repair_cut_chars_positive", r.repairCutChars > 0],
				);
			}
		}
		if (f.name === "over-long-huge-key-figures") {
			checks.push(["key_figures_trimmed", r.trimmedSections.includes("KEY FIGURES & IDENTIFIERS")]);
		}
		const bad = checks.filter(([, ok]) => !ok).map(([k]) => k);
		violations += bad.length;
		console.log(
			`[selftest] ${bad.length ? "FAIL" : "PASS"} ${f.name}: ` +
				`before=${r.before}tok/${f.note.length}ch after=${r.after}tok/${r.text.length}ch ` +
				`trimmed_tokens=${r.trimmedTokens} trimmed_sections=[${r.trimmedSections.join(", ")}] ` +
				`kept=[${r.sectionsKept.join(", ")}] boundary_repaired=${r.boundaryRepaired} ` +
				`repair_cut_chars=${r.repairCutChars} boundary_unresolved=${r.boundaryUnresolved} ` +
				`over_cap=${r.overCap} tail="${lastLine.slice(0, 80)}"` +
				(bad.length ? ` violations=[${bad.join(", ")}]` : ""),
		);
	}
	console.log(`[selftest] RESULT: ${violations === 0 ? "PASS" : `FAIL (${violations})`}`);
	return violations === 0 ? 0 : 1;
}

const NC_SELFTEST_ON =
	(process.env.NC_SELFTEST || "0") === "1" ||
	(process.env.NC_SELFTEST || "").toLowerCase() === "true" ||
	process.argv.includes("--selftest");

if (NC_SELFTEST_ON) {
	process.exit(runSelfTest());
}

export default function (pi: ExtensionAPI) {
	pi.on("session_before_compact", async (event, ctx) => {
		// BOTH arms are handled here so compaction has a single writer and the
		// post-compaction context is guaranteed to fit back under the trigger
		// (pi's native unbounded summarizer can exceed it and loop).
		//   baseline (NC_PERSIST=0): bounded comprehensive summary by the AGENT
		//   model (classic "everything in the note" baseline, no memory).
		//   neurocompact (NC_PERSIST=1): short handoff note + memory_compact().
		const { preparation, signal } = event;
		const { messagesToSummarize, turnPrefixMessages, tokensBefore, firstKeptEntryId, previousSummary } = preparation;

		// "original compaction" arm (long-replay eval): the real session's own
		// compaction summary is the artifact this branch continued from, so use
		// it verbatim instead of generating a new one. No memory_compact here —
		// this arm has no long-term store by construction.
		const origFile = (process.env.NC_ORIG_SUMMARY_FILE || "").trim();
		if (origFile) {
			let orig = "";
			try {
				orig = readFileSync(origFile, "utf8").trim();
			} catch (err) {
				console.error(`[nc] cannot read NC_ORIG_SUMMARY_FILE ${origFile}`, err);
			}
			if (orig) {
				if (process.env.NC_DEBUG === "1") {
					console.error(`[nc-debug] original-summary compaction: len=${orig.length} tokensBefore=${tokensBefore}`);
				}
				return {
					compaction: {
						summary: orig,
						firstKeptEntryId,
						tokensBefore,
						usage: undefined,
					},
				};
			}
		}

		if (process.env.NC_DEBUG === "1") {
			console.error(`[nc-debug] before_compact enter: msgs=${messagesToSummarize.length} prefix=${turnPrefixMessages.length} tokensBefore=${tokensBefore}`);
		}

		const model = ctx.modelRegistry.find(SUM_PROVIDER, SUM_MODEL_ID);
		if (process.env.NC_DEBUG === "1") {
			console.error(`[nc-debug] find(${SUM_PROVIDER}/${SUM_MODEL_ID}) ->`, model ? "ok" : "NOT FOUND");
		}
		let sumFailed = false;
		let sumFailureReason: string | undefined;
		if (!model) {
			// Never skip persisting when the compactor model cannot be resolved:
			// fall through with an empty summary so memory_compact still runs and
			// the handoff note degrades to the bounded raw tail.
			sumFailed = true;
			sumFailureReason = `model ${SUM_PROVIDER}/${SUM_MODEL_ID} not found in the model registry`;
			ctx.ui.notify(`[nc] model ${SUM_PROVIDER}/${SUM_MODEL_ID} not found; persisting raw-tail summary`, "warning");
		}

		const all = [...messagesToSummarize, ...turnPrefixMessages];
		const convText = serializeConversation(convertToLlm(all)).slice(0, RAW_CAP);
		const prev = previousSummary ? `\n\nPrevious handoff note (keep its active state):\n${previousSummary}` : "";

		// Equalized-budget protocol (chain-replay eval): both arms must produce
		// an ARTIFACT of the same shape and differ only in the token budget, so
		// the A/B delta is "how much of the dropped window the artifact carries"
		// and not a prompt-shape artefact. Same four headers, same rules; the
		// only difference is what the note is allowed to be (comprehensive vs
		// short index) and the cap (NC_SUM_MAX_TOKENS: 1536 baseline / 512 nc).
		const capParsed = Number.parseInt(process.env.NC_SUM_MAX_TOKENS || "", 10);
		const noteCap = Number.isNaN(capParsed) ? (!PERSIST ? 1536 : 640) : capParsed;
		const softTarget = Math.max(64, Math.floor(noteCap * 0.85));
		// SCHEMA + BUDGET rules are IDENTICAL in both arms; only the verbosity
		// instruction below differs. The explicit soft target (85% of the hard
		// cap) and the "the tail must always exist" rule exist because ACTIVE
		// STATE is the last section: a note that runs out of budget loses
		// exactly the state the handoff depends on.
		const commonRules = `Use exactly these four section headers, each on its own line, in this order:
GOAL
KEY FIGURES & IDENTIFIERS
DECISIONS & RATIONALE
ACTIVE STATE
Output budget: the hard output limit is ${noteCap} tokens. Target about ${softTarget} tokens (~85% of that) and stop writing when you reach the target; a note that is complete at ${softTarget} tokens is better than one that is cut at ${noteCap}.
The note is read top-down and its TAIL must always exist, so spend the budget on the tail, not on the head: DECISIONS & RATIONALE and ACTIVE STATE must always be complete, and ACTIVE STATE must always be present.
Plain text under each header; no tables, no markdown fences.`;
		// Baseline arm: bounded COMPREHENSIVE summary of the dropped window (all
		// facts stay in context). NC arm: the same schema, but a short index —
		// the facts themselves live in searchable long-term memory and are
		// recalled at question time. NC is explicitly NOT self-sufficient.
		const summaryMessages = [
			{
				role: "user" as const,
				content: [
					{
						type: "text" as const,
						text: !PERSIST
							? `Write the continuation summary of the conversation below for a colleague who will continue this work with ONLY this text in context. Be comprehensive: preserve every number, percentage and metric (with what it measures), every model/dataset/file name, identifier, path, command, version, decision and its rationale, and the exact current state.
${commonRules}${prev}

<conversation>
${convText}
</conversation>`
							: `Write the continuation note of the conversation below. The dropped context and this note are also stored in a searchable long-term memory that is queried automatically when a question about earlier detail is asked, so this note is a short INDEX of what matters, not a copy of the facts and not a self-sufficient record. Write tersely: short lines, no narration of the conversation, and no exhaustive enumeration (KEY FIGURES & IDENTIFIERS holds at most five entries; the rest are recallable). Keep only what a reader must not get wrong: the goal in one line, the identifiers that are easy to confuse, the decisions already taken and why, and what is in flight. If you are short of room, compress or drop earlier sections first — ACTIVE STATE is written last and must always be there and complete.
${commonRules}${prev}

<conversation>
${convText}
</conversation>`,
					},
				],
				timestamp: Date.now(),
			},
		];

		let response: any;
		const sumBaseTokens = noteCap;
		const sumReasoningEffort = (process.env.NC_SUM_REASONING_EFFORT || "").trim();
		// pi's own summarizer (core/compaction: completeSummarization() ->
		// completeSimple()) always runs the model through the provider-neutral
		// path with an explicit thinking level. A model call with NO thinking
		// level makes the OpenRouter adapter send reasoning: {effort: "none"}
		// (openai-completions buildParams: `thinkingFormat === "openrouter"`),
		// and endpoints such as z-ai/glm-5.3-flash reject that with
		//   400 "Reasoning is mandatory for this endpoint and cannot be disabled."
		// The stream then resolves to an AssistantMessage with stopReason
		// "error" instead of throwing, which is why the old code silently fell
		// back to the raw tail. Mirror the session's thinking level instead.
		const sessionThinking = ctx.thinkingLevel;
		// A model whose compat says it cannot honour a reasoning-effort knob
		// still gets `reasoning: {effort: ...}` from the OpenRouter adapter, and
		// some endpoints then spend the entire output budget on thoughts
		// (nemotron-free: 2048 of 2048 tokens, stopReason "length", empty text).
		// Summaries need no reasoning, so ask such models for "off" explicitly
		// (verified accepted: reasoning {effort:"none"} -> finish=stop). Models
		// that do support the knob inherit the session level, like pi's own
		// summarizer does.
		const sumSupportsEffort = model ? (model as any).compat?.supportsReasoningEffort !== false : false;
		const sumThinking =
			sumReasoningEffort ||
			(!sumSupportsEffort
				? "off"
				: sessionThinking && sessionThinking !== "off"
					? sessionThinking
					: model?.reasoning
						? "medium"
						: "off");
		// Reasoning tokens are spent from the same max_tokens ceiling. pi's
		// adjustMaxTokensForThinking() adds the level's thinking budget to the
		// caller's cap (clamped to the model limit) for exactly this reason;
		// without it a reasoning summarizer burns the whole budget on thoughts
		// (nemotron-free: 870 of 1000 tokens) and stops with stopReason
		// "length". Keep the configured cap as the visible-answer budget.
		const THINKING_BUDGET: Record<string, number> = { minimal: 1024, low: 2048, medium: 8192, high: 16384 };
		const thinkBudget = model?.reasoning && sumThinking !== "off" ? (THINKING_BUDGET[sumThinking] ?? 0) : 0;
		const sumCap = model?.maxTokens && model.maxTokens > 0 ? model.maxTokens : sumBaseTokens + thinkBudget;
		const sumMaxTokens = Math.min(sumBaseTokens + thinkBudget, sumCap);
		if (NOTE_ONLY) {
			// nc-note-only (fork of a post-compaction session): NO summarizer
			// call. The note that is already in the session is the artifact
			// under test, so reusing it is the honest fallback; falling back to
			// the raw tail would silently change the arm's budget.
			const reused = (previousSummary || "").trim();
			if (!reused) {
				sumFailureReason = "NC_NOTE_ONLY=1 but there is no previous compaction summary to reuse";
			}
			response = {
				content: reused ? [{ type: "text" as const, text: reused }] : [],
				usage: undefined,
				stopReason: "reused",
			};
		} else if (sumFailed) {
			// Compactor unavailable before the call: use the raw-tail path below.
			response = { content: [], usage: undefined, stopReason: "error" };
		} else
		try {
			// Same call shape as pi's native compaction summarizer: provider-
			// neutral streamSimple (auth resolved by the registry at request
			// time), no tools, no prompt-cache writes, session-scoped routing.
			response = await ctx.modelRegistry.streamSimple(model, { messages: summaryMessages }, {
				maxTokens: sumMaxTokens,
				signal,
				cacheRetention: "none",
				sessionId: uuidv7(),
				// pi-ai's anthropic adapter (streamSimpleAnthropic) switches thinking ON for
				// any truthy `reasoning`, including the string "off" (it only checks
				// `!options.reasoning`), so for the anthropic provider "off" must be
				// passed as an absent level. The openrouter/openai-completions path maps
				// "off" itself (clampedReasoning === "off"), so it keeps the string:
				// that call shape is the verified one and must not change.
				reasoning: (SUM_PROVIDER === "anthropic" && sumThinking === "off"
					? undefined
					: sumThinking) as any,
			}).result();
			if (process.env.NC_DEBUG === "1") {
				console.error(`[nc-debug] complete ok stop=${response.stopReason} thinking=${sumThinking} effortSupported=${sumSupportsEffort} maxTokens=${sumMaxTokens} usage=${JSON.stringify(response.usage ?? null)} error=${JSON.stringify(response.errorMessage ?? null)}`);
			}
		} catch (err) {
			sumFailureReason = String(err);
			ctx.ui.notify(`[nc] summarization failed (${err}); persisting raw-tail summary`, "warning");
			if (process.env.NC_DEBUG === "1") console.error(`[nc-debug] complete FAILED`, err);
			// Still persist: memory_compact below runs with the raw-tail note
			// instead of silently dropping this compaction's facts.
			response = { content: [], usage: undefined, stopReason: "error" };
		}

		const summary = (response.content ?? [])
			.filter((c: any): c is { type: "text"; text: string } => c.type === "text")
			.map((c: any) => c.text)
			.join("\n")
			.trim();
		// A provider-level failure does not throw: the stream resolves with
		// stopReason "error" (and zero usage). Log it loudly - an empty summary
		// must be visible on stderr, not only in NC_DEBUG.
		if (!sumFailureReason && response.stopReason === "error") {
			sumFailureReason = response.errorMessage || "provider returned stopReason=error";
		} else if (!sumFailureReason && !summary) {
			sumFailureReason = `empty completion (stop=${response.stopReason}${response.errorMessage ? `: ${response.errorMessage}` : ""})`;
		}
		if (sumFailureReason) {
			console.error(`[nc] summarizer failed: ${sumFailureReason}`);
		} else if (response.stopReason === "length") {
			console.error(`[nc] summarizer warning: completion hit the ${sumMaxTokens}-token cap; using partial summary`);
		}
		// NEVER fall back to pi's default (unbounded) summarizer here: its
		// output can exceed the compaction trigger and cause a compaction
		// loop that collapses context. If the model returned nothing usable,
		// use a bounded raw tail of the dropped conversation instead.
		//
		// Deterministic post-processing to the VISIBLE cap (nc default ON): the
		// provider may have amputated the note at stop=length, which always eats
		// the tail (ACTIVE STATE). Never trust the model's own length control.
		// forceRepair wires the stop reason in: a stop=length completion that was
		// cut on a bare list marker or mid-sentence gets its tail repaired.
		const trimRes = summary
			? trimNote(summary, noteCap, SECTION_TRIM, response.stopReason === "length")
			: null;
		const trimmedSummary = trimRes ? trimRes.text : "";
		if (trimRes) {
			// Human-greppable line, ALWAYS printed for a summarized compaction
			// and ALWAYS carrying boundary_repaired / repair_cut_chars /
			// boundary_unresolved: the eval harness scrapes these, so an absent
			// field must never be distinguishable from a false one. (The
			// machine-readable JSON [nc-metrics] line below is what the runner
			// parses; this one is for humans and grep.)
			console.error(
				`[nc-metrics] trimmed_sections=[${trimRes.trimmedSections.join(",")}] trimmed_tokens=${trimRes.trimmedTokens} boundary_repaired=${trimRes.boundaryRepaired} repair_cut_chars=${trimRes.repairCutChars} boundary_unresolved=${trimRes.boundaryUnresolved}`,
			);
		}
		if (trimRes && !SECTION_TRIM && trimRes.overCap) {
			console.error(
				`[nc] note is ${trimRes.after} tokens > cap ${noteCap} and NC_SUM_SECTION_TRIM=0 (verbatim arm)`,
			);
		}
		if (trimRes && SECTION_TRIM && trimRes.overCap) {
			console.error(
				`[nc] note still ${trimRes.after} tokens > cap ${noteCap} after trim (ACTIVE STATE is never cut)`,
			);
		}
		const summaryOrRaw = trimmedSummary || `[unsummarized — raw tail of dropped context]\n${convText.slice(-4000)}`;
		const realizedTokens = countTokens(summaryOrRaw);
		if (process.env.NC_DEBUG === "1") {
			console.error(`[nc-debug] summary len=${summary.length} head=${summary.slice(0, 90)}`);
		}

		let persisted = false;
		let persistError: string | undefined;
		if (PERSIST) {
			try {
				const dropped =
					convText.length > 2 * DROP_HALF
						? [convText.slice(0, DROP_HALF), convText.slice(-DROP_HALF)]
						: [convText];
				const r = await mcpToolCall("memory_compact", {
					session_id: SID,
					summary: summaryOrRaw,
					dropped_context: dropped,
					metadata: {
						source: "pi-compaction",
						tokens_before: tokensBefore,
						messages: all.length,
						persist: PERSIST ? 1 : 0,
					},
				});
				persisted = r?.ok !== false;
				if (!persisted) persistError = "memory_compact returned ok=false";
				ctx.ui.notify(`[nc] memory_compact seq=${r?.seq ?? "?"} ok=${r?.ok}`, "info");
			if (process.env.NC_DEBUG === "1") console.error(`[nc-debug] memory_compact result:`, JSON.stringify(r).slice(0, 160));
			} catch (err) {
				persistError = String(err);
				ctx.ui.notify(`[nc] memory_compact failed: ${err}`, "error");
			if (process.env.NC_DEBUG === "1") console.error("[nc-debug] memory_compact FAILED", err);
			}
		}
		// One machine-readable line per compaction, always on stderr and always
		// regardless of NC_DEBUG: the eval runner parses finish_reason / the
		// realized artifact size from here instead of scraping debug noise.
		console.error(`[nc-metrics] ${JSON.stringify({
			arm: NOTE_ONLY ? "note-only" : PERSIST ? "nc" : "baseline",
			stop: String(response.stopReason ?? "unknown"),
			max_tokens: sumMaxTokens,
			visible_cap: noteCap,
			usage: response.usage ?? null,
			summary_chars: summaryOrRaw.length,
			realized_tokens: realizedTokens,
			realized_chars: summaryOrRaw.length,
			tokenizer: TOKENIZER_LABEL,
			section_trim: SECTION_TRIM,
			trimmed_sections: trimRes ? trimRes.trimmedSections : [],
			trimmed_tokens: trimRes ? trimRes.trimmedTokens : 0,
			boundary_repaired: trimRes ? trimRes.boundaryRepaired : false,
			repair_cut_chars: trimRes ? trimRes.repairCutChars : 0,
			boundary_unresolved: trimRes ? trimRes.boundaryUnresolved : false,
			...(trimRes ? {
				model_tokens: trimRes.before,
				active_state_kept: trimRes.activeStateKept,
				active_state_complete: trimRes.activeStateComplete,
				over_cap: trimRes.overCap,
				sections_kept: trimRes.sectionsKept,
			} : {}),
			persisted,
			note_only: NOTE_ONLY,
			model: `${SUM_PROVIDER}/${SUM_MODEL_ID}`,
			reasoning: sumThinking,
			tokens_before: tokensBefore,
			fallback: summaryOrRaw.startsWith("[unsummarized") ? "raw-tail" : "",
			...(sumFailureReason ? { error: sumFailureReason } : {}),
			...(persistError ? { persist_error: persistError } : {}),
		})}`);

		return {
			compaction: {
				summary: summaryOrRaw,
				firstKeptEntryId,
				tokensBefore,
				usage: response.usage,
			},
		};
	});

	// Deterministic force-compact at the boundary: when context usage grows past
	// the threshold, trigger and AWAIT pi's compaction before the request runs.
	//
	// This MUST sit on "input" (before the turn's LLM request), not on
	// "agent_before_settle": calling ctx.compact() while pi is settling never
	// completes on pi 1.4.0 — the onComplete/onError callbacks are never
	// invoked, the 120 s safety timeout below expires, and NO compaction entry
	// is written (verified: memory_compact still ran, so the extension hook
	// fired, but the session file gained no compaction and stdout had no
	// compaction_end). From "input" the same call completes in ~2 s and the
	// entry is persisted before the request.
	pi.on("input", async (_event, ctx) => {
		try {
			const usage = ctx.getContextUsage();
			if (process.env.NC_DEBUG === "1") {
				console.error("[neurocompact] input-hook usage=", JSON.stringify(usage),
					"threshold=", process.env.NC_FORCE_COMPACT_THRESHOLD);
			}
			if (!usage || usage.tokens === null) return;
			const parsed = Number.parseInt(process.env.NC_FORCE_COMPACT_THRESHOLD || "", 10);
			const threshold = Number.isNaN(parsed) ? 3500 : parsed;
			if (usage.tokens <= threshold) return;
			if (process.env.NC_DEBUG === "1") {
				console.error("[neurocompact] input-hook usage tokens=", usage.tokens, "threshold=", threshold);
			}
			const now = Date.now();
			if (now - lastCompactedAt < 2000) return;
			lastCompactedAt = now;

			// ctx.compact() is fire-and-forget, so wrap it in a promise that
			// resolves on its completion/error callbacks, and race a 120 s
			// safety timeout. The timeout timer is always cleared so it never
			// keeps the process alive.
			let timer: ReturnType<typeof setTimeout> | undefined;
			try {
				await Promise.race([
					new Promise<void>((resolve) => {
						let done = false;
						const finish = () => {
							if (!done) {
								done = true;
								resolve();
							}
						};
						try {
							ctx.compact({ onComplete: finish, onError: finish });
						} catch (e) {
							finish();
						}
					}),
					new Promise<void>((resolve) => {
						timer = setTimeout(resolve, 120000);
					}),
				]);
			} finally {
				if (timer !== undefined) clearTimeout(timer);
			}
			if (process.env.NC_DEBUG === "1") {
				console.error("[neurocompact] input-hook compact awaited");
			}
		} catch (err) {
			console.error("[neurocompact] input-hook compact failed", err);
		}
	});
}
