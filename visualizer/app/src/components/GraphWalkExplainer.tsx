import { useMemo, useState } from "react";
import type { ChunkHit, TravelTrace } from "../lib/types";
import type { Dict } from "../i18n/strings";
import { fmt } from "../i18n/strings";
import { buildGraphWalk, type HiWStep } from "../lib/graphWalk";
import {
  AlertIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  LayersIcon,
  LinkIcon,
  NodesIcon,
  SearchIcon,
  SparkIcon,
} from "./icons";

interface Props {
  trace: TravelTrace;
  t: Dict;
  onFocusNode: (nodeIndex: number) => void;
  onFocusChunk: (chunk: ChunkHit) => void;
}

const STEPS: { id: HiWStep; key: "hiwStepSeeds" | "hiwStepWalk" | "hiwStepChunks" | "hiwStepRank" | "hiwStepContext" }[] = [
  { id: 1, key: "hiwStepSeeds" },
  { id: 2, key: "hiwStepWalk" },
  { id: 3, key: "hiwStepChunks" },
  { id: 4, key: "hiwStepRank" },
  { id: 5, key: "hiwStepContext" },
];

function shortId(id: string): string {
  const tail = id.split("-").pop() ?? id;
  return tail.slice(0, 6);
}

function nodeTitle(n: { name: string; type: string; deg: number; desc: string }): string {
  return `${n.name} · ${n.type} · ${n.deg}\n${n.desc}`;
}

/**
 * The five-step "how the graph walk works" illustration for the graph stage.
 * One step body at a time (progressive disclosure), a clickable rail that
 * carries the live counts, and a three-verb legend for the motion vocabulary.
 */
export default function GraphWalkExplainer({ trace, t, onFocusNode, onFocusChunk }: Props) {
  const model = useMemo(() => buildGraphWalk(trace), [trace]);
  const [step, setStep] = useState<HiWStep>(1);
  const [selectedSeed, setSelectedSeed] = useState<number | null>(null);
  const [showDropped, setShowDropped] = useState(false);
  const [showAllSeeds, setShowAllSeeds] = useState(false);

  const counts: Record<HiWStep, number> = {
    1: model.seeds.length,
    2: model.neighbours.length,
    3: model.linkedChunkCount,
    4: model.kept.length,
    5: model.contextDocs.length,
  };

  const activeSeed =
    model.seedGroups.find((g) => g.seed.i === selectedSeed) ??
    model.seedGroups.find((g) => g.children.length > 0) ??
    model.seedGroups[0] ??
    null;

  const go = (next: HiWStep) => {
    setStep(next);
    setShowDropped(false);
  };

  const titles: Record<HiWStep, string> = {
    1: t.hiwTitleSeeds,
    2: t.hiwTitleWalk,
    3: t.hiwTitleChunks,
    4: t.hiwTitleRank,
    5: t.hiwTitleContext,
  };
  const captions: Record<HiWStep, string> = {
    1: fmt(t.hiwCapSeeds, {
      seeds: model.seeds.length,
      matched: model.matchedTokenCount,
      tokens: model.tokens.length,
    }),
    2: fmt(t.hiwCapWalk, {
      neighbors: model.neighbours.length,
      hops: model.hop2.length > 0 ? "2" : "1",
      edges: model.tracedEdges,
    }),
    3: fmt(t.hiwCapChunks, {
      chunks: model.linkedChunkCount,
      entities: model.chunkLinks.length,
      docs: model.docs.length,
    }),
    4: fmt(t.hiwCapRank, {
      kept: model.kept.length,
      k: model.topK ?? model.kept.length,
      dropped: model.droppedCount,
    }),
    5: fmt(t.hiwCapContext, {
      sources: model.contextSourceCount ?? model.contextDocSources.reduce((n, d) => n + d.n, 0),
      docs: model.contextDocs.length,
    }),
  };

  return (
    <div className="hiw" role="region" aria-label={t.hiwAria}>
      <div className="hiw-rail" role="tablist" aria-label={t.hiwAria}>
        {STEPS.map((s, i) => (
          <div className="hiw-rail-item" key={s.id}>
            {i > 0 && <span className={`hiw-rail-line ${step >= s.id ? "on" : ""}`} aria-hidden="true" />}
            <button
              type="button"
              role="tab"
              aria-selected={step === s.id}
              className={`hiw-tab ${step === s.id ? "on" : ""} ${step > s.id ? "past" : ""}`}
              onClick={() => go(s.id)}
              title={`${t.hiwStepOf.replace("{n}", String(s.id))} · ${t[s.key]}`}
            >
              <span className="hiw-tab-num">{s.id}</span>
              <span className="hiw-tab-label">{t[s.key]}</span>
              <span className="hiw-tab-count">{counts[s.id]}</span>
            </button>
          </div>
        ))}
      </div>

      <div className="hiw-head">
        <span className="hiw-kicker">
          {t.hiwStepOf.replace("{n}", String(step))} <span className="dim">·</span> {titles[step]}
        </span>
        <span className="hiw-caption">{captions[step]}</span>
      </div>

      <div className="hiw-body" key={step}>
        {step === 1 && (
          <div className="hiw-pane">
            <div className="hiw-token-block">
              <div className="hiw-block-title">
                <SearchIcon size={10} />
                <span>{t.hiwQuestionTokens}</span>
                <span className="hiw-src-tag">{model.tokensFrom === "trace" ? t.hiwSrcTrace : t.hiwSrcQuestion}</span>
              </div>
              <div className="hiw-tokens">
                {model.tokens.map((tok, i) => {
                  const matched = !model.unmatchedTokens.includes(tok);
                  return (
                    <span
                      key={tok}
                      className={`hiw-token ${matched ? "matched" : "unmatched"}`}
                      style={{ animationDelay: `${i * 28}ms` }}
                      title={matched ? t.hiwMatched : t.hiwNoMatch}
                    >
                      {tok}
                    </span>
                  );
                })}
                {model.tokens.length === 0 && <span className="hiw-missing-inline">{t.hiwNoTokens}</span>}
              </div>
              {model.tokens.length > 0 && (
                <p className="hiw-note">
                  {fmt(t.hiwTokenNote, { matched: model.matchedTokenCount, total: model.tokens.length })}
                </p>
              )}
            </div>

            <div className="hiw-block-title">
              <NodesIcon size={10} />
              <span>{t.hiwSeeds}</span>
              <span className="hiw-src-tag">{t.hiwSrcTrace}</span>
            </div>
            {model.seeds.length === 0 ? (
              <MissingBox text={t.hiwNoSeeds} />
            ) : (
              <div className={`hiw-seed-list ${showAllSeeds ? "open" : ""}`}>
                {model.seeds.map((s, i) => (
                  <button
                    key={s.node.i}
                    type="button"
                    className="hiw-seed"
                    style={{ animationDelay: `${i * 22}ms` }}
                    onClick={() => {
                      onFocusNode(s.node.i);
                      setSelectedSeed(s.node.i);
                      go(2);
                    }}
                    title={`${nodeTitle(s.node)}\n${t.hiwClickFocus}`}
                  >
                    <span className="hiw-seed-dot" aria-hidden="true" />
                    <span className="hiw-seed-name">{s.node.name}</span>
                    <span className="hiw-type">{s.node.type || t.hiwMissing}</span>
                    <span className="hiw-seed-deg" title={t.hiwDegree}>{s.node.deg}</span>
                    {s.matchedTokens.length > 0 ? (
                      <span className="hiw-seed-match">{s.matchedTokens.join(" · ")}</span>
                    ) : (
                      <span className="hiw-seed-nomatch" title={t.hiwSeedNoToken}>
                        {t.hiwNoToken}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
            {model.seeds.length > 7 && (
              <button type="button" className="hiw-more" onClick={() => setShowAllSeeds((v) => !v)}>
                {showAllSeeds ? <ChevronUpIcon size={10} /> : <ChevronDownIcon size={10} />}
                {showAllSeeds ? t.hiwShowLess : fmt(t.hiwShowMore, { n: model.seeds.length - 7 })}
              </button>
            )}
          </div>
        )}

        {step === 2 && (
          <div className="hiw-pane">
            <div className="hiw-seed-chips" role="group" aria-label={t.hiwSeeds}>
              {model.seedGroups.map((g) => (
                <button
                  key={g.seed.i}
                  type="button"
                  className={`hiw-seed-chip ${activeSeed?.seed.i === g.seed.i ? "on" : ""} ${g.children.length === 0 ? "empty" : ""}`}
                  onClick={() => setSelectedSeed(g.seed.i)}
                  onDoubleClick={() => onFocusNode(g.seed.i)}
                  title={`${nodeTitle(g.seed)}\n${fmt(t.hiwSeedLinks, { n: g.children.length })}`}
                >
                  <span className="hiw-seed-dot" aria-hidden="true" />
                  {g.seed.name}
                  <span className="hiw-chip-count">{g.children.length}</span>
                </button>
              ))}
              {model.seedGroups.length === 0 && <span className="hiw-missing-inline">{t.hiwNoSeeds}</span>}
            </div>

            {model.walkEdgeCount > 0 && (
              <p className={`hiw-note ${model.labelledEdgeCount === 0 ? "warn" : ""}`}>
                {fmt(t.hiwRelationCoverage, {
                  labelled: model.labelledEdgeCount,
                  total: model.walkEdgeCount,
                })}
              </p>
            )}
            {model.walkEdgeCount > 0 && model.labelledEdgeCount < model.walkEdgeCount && (
              <p className="hiw-note dim">{t.hiwRelationWhy}</p>
            )}

            {activeSeed && (
              <div className="hiw-walk" key={activeSeed.seed.i}>
                <div className="hiw-walk-head">
                  <button
                    type="button"
                    className="hiw-walk-root"
                    onClick={() => onFocusNode(activeSeed.seed.i)}
                    title={nodeTitle(activeSeed.seed)}
                  >
                    <span className="hiw-hop hop0">{t.hiwHop0}</span>
                    <span className="hiw-walk-name">{activeSeed.seed.name}</span>
                    <span className="hiw-type">{activeSeed.seed.type}</span>
                  </button>
                  <span className="hiw-walk-facts">
                    {fmt(t.hiwSeedLinks, { n: activeSeed.children.length })}
                  </span>
                </div>

                {activeSeed.children.length === 0 ? (
                  <MissingBox text={t.hiwNoNeighbours} />
                ) : (
                  <ul className="hiw-edges">
                    {activeSeed.children.map((c, i) => (
                      <li
                        key={`${activeSeed.seed.i}-${c.node.i}`}
                        className="hiw-edge"
                        style={{ animationDelay: `${i * 30}ms` }}
                      >
                        <span className="hiw-edge-line" aria-hidden="true" />
                        <span
                          className={`hiw-rel ${c.relation ? "ok" : "missing"}`}
                          title={c.relation ? c.relation : t.hiwRelationMissingNote}
                        >
                          {c.relation ?? (model.labelledEdgeCount === 0 ? "—" : t.hiwRelationMissing)}
                        </span>
                        <button
                          type="button"
                          className="hiw-node"
                          onClick={() => onFocusNode(c.node.i)}
                          title={`${nodeTitle(c.node)}\n${t.hiwClickFocus}`}
                        >
                          {c.node.name}
                        </button>
                        <span className={`hiw-hop ${c.hop === 1 ? "hop1" : c.hop === 2 ? "hop2" : "hopna"}`}>
                          {c.hop === 1 ? t.hiwHop1 : c.hop === 2 ? t.hiwHop2 : t.hiwMissing}
                        </span>
                        <span className="hiw-type">{c.node.type || t.hiwMissing}</span>
                      </li>
                    ))}
                  </ul>
                )}

                <p className="hiw-note">
                  {activeSeed.outsideLinks > 0
                    ? fmt(t.hiwOutsideLinks, {
                        n: activeSeed.outsideLinks,
                        total: activeSeed.seed.deg,
                        shown: activeSeed.tracedLinks,
                      })
                    : t.hiwAllLinksTraced}
                </p>
              </div>
            )}

            <div className="hiw-prune-box">
              <span className="hiw-prune-title">
                <AlertIcon size={10} />
                {t.hiwPruned}
              </span>
              {model.prunedHubs != null ? (
                <span className="hiw-prune-line">
                  {model.hubCutoffFanout != null
                    ? fmt(t.hiwHubCutoffFanout, { n: model.prunedHubs, fanout: model.hubCutoffFanout })
                    : model.hubCutoffDegree != null
                      ? fmt(t.hiwHubCutoff, { n: model.prunedHubs, deg: model.hubCutoffDegree })
                      : fmt(t.hiwHubCutoffPlain, { n: model.prunedHubs })}
                </span>
              ) : (
                <span className="hiw-prune-line missing">{t.hiwHubMissing}</span>
              )}
              {model.fullNeighbourCount != null && model.fullNeighbourCount > model.neighbours.length && (
                <span className="hiw-prune-line">
                  {fmt(t.hiwDisplayCap, {
                    shown: model.neighbours.length,
                    total: model.fullNeighbourCount,
                    cap: model.nodeCap ?? model.neighbours.length,
                  })}
                </span>
              )}
              {model.fullNeighbourCount != null && model.fullNeighbourCount === model.neighbours.length && (
                <span className="hiw-prune-line">
                  {fmt(t.hiwAllNeighboursCarried, { n: model.neighbours.length })}
                </span>
              )}
              {model.extraEdges > 0 && (
                <span className="hiw-prune-line dim">{fmt(t.hiwExtraEdges, { n: model.extraEdges })}</span>
              )}
              {model.statsMissing && (
                <span className="hiw-prune-line missing">{t.hiwStatsMissing}</span>
              )}
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="hiw-pane">
            <div className="hiw-block-title">
              <LinkIcon size={10} />
              <span>{t.hiwChunkProvenance}</span>
              <span className="hiw-src-tag">{t.hiwSrcTrace}</span>
            </div>
            {model.chunkLinks.length === 0 ? (
              <MissingBox text={t.hiwNoChunkLinks} />
            ) : (
              <div className="hiw-prov-list">
                {model.chunkLinks.map((l, i) => (
                  <div className="hiw-prov" key={l.node.i} style={{ animationDelay: `${i * 24}ms` }}>
                    <div className="hiw-prov-head">
                      <button
                        type="button"
                        className={`hiw-prov-ent ${l.node.seed ? "seed" : ""}`}
                        onClick={() => onFocusNode(l.node.i)}
                        title={nodeTitle(l.node)}
                      >
                        <span className="hiw-seed-dot" aria-hidden="true" />
                        {l.node.name}
                      </button>
                      <span className="hiw-prov-count">{fmt(t.hiwChunkCount, { n: l.chunks.length })}</span>
                    </div>
                    <div className="hiw-prov-chunks">
                      {l.chunks.slice(0, 3).map((h) => (
                        <button
                          key={h.chunk_id}
                          type="button"
                          className="hiw-chunk-mini"
                          onClick={() => onFocusChunk(h)}
                          title={`${h.doc} · ${h.chunk_id}`}
                        >
                          {h.doc}
                          <span className="hiw-chunk-id">{shortId(h.chunk_id)}</span>
                        </button>
                      ))}
                      {l.chunks.length > 3 && (
                        <span className="hiw-chunk-more">+{l.chunks.length - 3}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {model.entitiesWithoutChunks.length > 0 && (
              <p className="hiw-note">
                {fmt(t.hiwChunkNoChunks, {
                  n: model.entitiesWithoutChunks.length,
                  k: model.topK ?? model.kept.length,
                })}
              </p>
            )}

            <div className="hiw-block-title hiw-doc-title">
              <LayersIcon size={10} />
              <span>{t.hiwDocsContribution}</span>
            </div>
            <div className="hiw-doc-list">
              {model.docs.slice(0, 7).map((d) => {
                const max = Math.max(1, ...model.docs.map((x) => x.kept));
                return (
                  <div className="hiw-doc-row" key={d.doc}>
                    <span className="hiw-doc-name">{d.doc}</span>
                    <span className="hiw-doc-bar" aria-hidden="true">
                      <span style={{ width: `${Math.round((d.kept / max) * 100)}%` }} />
                    </span>
                    <span className="hiw-doc-count" title={fmt(t.hiwDocCandidates, { n: d.total })}>
                      {d.kept}
                    </span>
                  </div>
                );
              })}
              {model.docs.length > 7 && (
                <p className="hiw-note">{fmt(t.hiwMoreDocs, { n: model.docs.length - 7 })}</p>
              )}
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="hiw-pane">
            <div className="hiw-split" role="img" aria-label={captions[4]}>
              <span
                className="hiw-split-kept"
                style={{ width: `${Math.round((model.kept.length / Math.max(1, model.kept.length + model.droppedCount)) * 100)}%` }}
              />
              <span className="hiw-split-dropped" />
            </div>
            <div className="hiw-split-legend">
              <span className="hiw-kept-label">
                <CheckIcon size={10} /> {fmt(t.hiwRankKept, { k: model.topK ?? model.kept.length })} · {model.kept.length}
              </span>
              <span className="hiw-dropped-label">
                {t.hiwRankDropped} · {model.droppedCount}
              </span>
            </div>

            {model.channels ? (
              <div className="hiw-chan-line">
                <span className="hiw-chan-title">{t.hiwRecallChannels}</span>
                <span className="hiw-chan dense">
                  {t.hiwChannelDense} {model.channels.dense ?? "—"}
                </span>
                <span className="hiw-chan bm25">
                  {t.hiwChannelBm25} {model.channels.bm25 ?? "—"}
                </span>
                {model.channels.ripgrep && (
                  <span
                    className="hiw-chan ripgrep"
                    title={`${t.hiwChannelRipgrep}: ${model.channels.ripgrep.enabled ? t.hiwOn : t.hiwOff}`}
                  >
                    {t.hiwChannelRipgrep} {model.channels.ripgrep.matched ?? model.channels.ripgrep.hits ?? "—"}
                  </span>
                )}
                {model.channels.graphBridge && (
                  <span
                    className="hiw-chan bridge"
                    title={model.channels.graphBridge.reason ?? t.hiwChannelBridge}
                  >
                    {t.hiwChannelBridge} {model.channels.graphBridge.enabled ? t.hiwOn : t.hiwOff}
                  </span>
                )}
              </div>
            ) : (
              <p className="hiw-note">{t.hiwChannelsMissing}</p>
            )}

            <div className="hiw-rank-list">
              {model.kept.map((k, i) => (
                <button
                  key={k.hit.chunk_id}
                  type="button"
                  className={`hiw-rank-row kept ${k.promotedBy > 0 ? "promoted" : k.demotedBy > 0 ? "demoted" : ""}`}
                  style={{ animationDelay: `${i * 26}ms` }}
                  onClick={() => onFocusChunk(k.hit)}
                  title={`${k.hit.chunk_id}\n${t.hiwClickChunk}`}
                >
                  <span className="hiw-rank-num">{k.rank}</span>
                  <span className="hiw-rank-doc">{k.hit.doc}</span>
                  <span className="hiw-channels">
                    {k.channels.map((ch) => (
                      <span key={ch} className={`hiw-chan ${ch}`}>
                        {ch === "dense" ? t.hiwChannelDense : ch === "bm25" ? t.hiwChannelBm25 : t.hiwChannelGraph}
                      </span>
                    ))}
                  </span>
                  {k.promotedBy > 0 && (
                    <span className="hiw-move up" title={fmt(t.hiwPromoted, { n: k.promotedBy })}>
                      +{k.promotedBy}
                    </span>
                  )}
                  {k.demotedBy > 0 && (
                    <span className="hiw-move down" title={fmt(t.hiwDemoted, { n: k.demotedBy })}>
                      -{k.demotedBy}
                    </span>
                  )}
                  <span className="hiw-rank-score">{k.hit.fused.toFixed(3)}</span>
                </button>
              ))}
            </div>

            <div className="hiw-dropped-block">
              {model.droppedCount > 0 && (
                <button type="button" className="hiw-more" onClick={() => setShowDropped((v) => !v)}>
                  {showDropped ? <ChevronUpIcon size={10} /> : <ChevronDownIcon size={10} />}
                  {showDropped
                    ? t.hiwDroppedHide
                    : fmt(t.hiwDroppedShow, { n: model.droppedCount })}
                </button>
              )}
              {showDropped && (
                <div className="hiw-dropped-list">
                  {model.droppedFromFused.map((h) => (
                    <span className="hiw-dropped-chip" key={h.chunk_id} title={h.chunk_id}>
                      {h.doc}
                    </span>
                  ))}
                  {model.poolOnly.slice(0, 10).map((h) => (
                    <span className="hiw-dropped-chip dim" key={h.chunk_id} title={h.chunk_id}>
                      {h.doc}
                    </span>
                  ))}
                  {model.droppedCount === 0 && <span className="hiw-missing-inline">{t.hiwNoDropped}</span>}
                </div>
              )}
              <p className="hiw-note">
                {fmt(t.hiwDroppedNote, {
                  fused: model.droppedFromFused.length,
                  k: model.topK ?? model.kept.length,
                  pool: model.poolOnly.length,
                })}
              </p>
            </div>

            <p className="hiw-note hiw-channel-note">{t.hiwChannelNote}</p>
            {model.topKSource === "fused-length" && (
              <p className="hiw-note">{t.hiwTopKMissing}</p>
            )}
          </div>
        )}

        {step === 5 && (
          <div className="hiw-pane">
            <div className="hiw-stats">
              <div className="hiw-stat">
                <span className="hiw-stat-val">{model.kept.length}</span>
                <span className="hiw-stat-lbl">{t.hiwContextChunks}</span>
              </div>
              <div className="hiw-stat">
                <span className="hiw-stat-val">
                  {model.contextSourceCount ??
                    model.contextDocSources.reduce((n, d) => n + d.n, 0)}
                </span>
                <span className="hiw-stat-lbl">{t.hiwContextSources}</span>
              </div>
              <div className="hiw-stat">
                <span className="hiw-stat-val">{model.contextDocs.length}</span>
                <span className="hiw-stat-lbl">{t.hiwContextDocs}</span>
              </div>
              <div className="hiw-stat">
                <span className="hiw-stat-val">{model.contextEntities}</span>
                <span className="hiw-stat-lbl">{t.hiwContextFacts}</span>
              </div>
            </div>

            <p className="hiw-sowhat">
              <SparkIcon size={11} />
              <span>
                {fmt(t.hiwSoWhat, {
                  sources:
                    model.contextSourceCount ??
                    model.contextDocSources.reduce((n, d) => n + d.n, 0),
                  docs: model.contextDocs.length,
                })}
              </span>
            </p>

            <div className="hiw-block-title">
              <LayersIcon size={10} />
              <span>{t.hiwContextDocsTitle}</span>
            </div>
            {model.contextMissing ? (
              <MissingBox text={t.hiwContextEmpty} />
            ) : (
              <div className="hiw-final-docs">
                {model.contextDocSources.map(({ doc, n }) => (
                  <div className="hiw-final-doc" key={doc}>
                    <span className="hiw-final-doc-name">{doc}</span>
                    <span className="hiw-final-doc-count">{fmt(t.hiwSourceCount, { n })}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="hiw-legend" aria-label={t.hiwLegend}>
        <span className="hiw-legend-title">{t.hiwLegend}</span>
        <span className="hiw-legend-item">
          <span className="hiw-verb appear" aria-hidden="true" />
          {t.hiwLegendAppear}
        </span>
        <span className="hiw-legend-item">
          <span className="hiw-verb move" aria-hidden="true">
            <ChevronUpIcon size={9} />
          </span>
          {t.hiwLegendMove}
        </span>
        <span className="hiw-legend-item">
          <span className="hiw-verb fade" aria-hidden="true" />
          {t.hiwLegendFade}
        </span>
      </div>
    </div>
  );
}

function MissingBox({ text }: { text: string }) {
  return (
    <div className="hiw-missing">
      <AlertIcon size={11} />
      <span>{text}</span>
    </div>
  );
}
