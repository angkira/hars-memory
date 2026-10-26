import { useEffect } from "react";
import { BRIDGE_DIRECT_URL, type ConnState } from "../lib/live";
import type { GraphStats } from "../lib/cosmosSource";
import { fieldOwnsKey } from "../lib/keys";
import { fmt } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import { SparkIcon, XIcon } from "./icons";

interface Props {
  onClose: () => void;
  /** Live connection identity (memory_status + trace_start), never hardcoded. */
  conn: ConnState;
  /** Refetch memory_status — called when the card opens so the facts are fresh. */
  onRefreshConn: () => void;
  /** Which dataset the starfield is actually rendering, from /api/stats. */
  identity: GraphStats | null;
  sourcePhase: "probing" | "live" | "unreachable" | "demo";
}

/** What-this-is card: provenance, render modes, controls. Replaces the dead
 *  external docs link — everything shown is true of this repo. Bilingual
 *  (RU/EN) through the shared dictionary, so it follows the language toggle.
 *  The Connected memory block shows what the bridge ACTUALLY serves (index
 *  id/dir, live counts, last ingest, models) — the app carries no hardcoded
 *  deployment identity. */
export default function AboutModal({ onClose, conn, onRefreshConn, identity, sourcePhase }: Props) {
  const { t, lang } = useI18n();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (fieldOwnsKey(e)) return;
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // (re)read the bridge identity every time the card opens
  useEffect(() => {
    onRefreshConn();
  }, [onRefreshConn]);

  // the bridge URL is rendered as a selectable <code> in the middle of the copy
  const [bridgeBefore, bridgeAfter] = t.aboutBridge.split("{url}");

  const nf = (n: number) => n.toLocaleString(lang === "ru" ? "ru-RU" : "en-US");
  const info = conn.kind === "connected" ? conn.info : null;
  const lastIngest = info?.lastIngest
    ? new Date(info.lastIngest).toLocaleString(lang === "ru" ? "ru-RU" : "en-US")
    : null;
  const models = [info?.embedder, info?.extractor, info?.queryModel].filter((m): m is string => !!m);

  return (
    <div className="errwrap" role="dialog" aria-label={t.aboutAria} onClick={onClose}>
      <div className="errcard about" onClick={(e) => e.stopPropagation()}>
        <h2>
          <SparkIcon size={16} /> Memory Travel
          <button className="ghostbtn" onClick={onClose} title={t.aboutClose}>
            <XIcon size={13} />
          </button>
        </h2>
        <p>{t.aboutIntro}</p>

        <section className="connblock">
          <h3>{t.aboutConnTitle}</h3>
          {info ? (
            <table className="conntable">
              <tbody>
                <tr>
                  <th>{t.aboutConnBridge}</th>
                  <td><code>{info.endpoint ?? BRIDGE_DIRECT_URL}</code></td>
                </tr>
                {info.indexId && (
                  <tr>
                    <th>{t.aboutConnIndex}</th>
                    <td><code>{info.indexId}</code></td>
                  </tr>
                )}
                {info.indexDir && (
                  <tr>
                    <th>{t.aboutConnDir}</th>
                    <td className="dim">{info.indexDir}</td>
                  </tr>
                )}
                {(info.nodeCount != null || info.edgeCount != null) && (
                  <tr>
                    <th>{t.aboutConnEntities} / {t.aboutConnRelations}</th>
                    <td>
                      {info.nodeCount != null ? nf(info.nodeCount) : "…"} / {info.edgeCount != null ? nf(info.edgeCount) : "…"}
                    </td>
                  </tr>
                )}
                {lastIngest && (
                  <tr>
                    <th>{t.aboutConnIngest}</th>
                    <td>
                      {lastIngest}
                      {info.staleDays != null && (
                        <span className="dim"> · {fmt(t.aboutStaleDays, { n: info.staleDays })}</span>
                      )}
                    </td>
                  </tr>
                )}
                {models.length > 0 && (
                  <tr>
                    <th>{t.aboutConnModels}</th>
                    <td className="dim">{models.join(" · ")}</td>
                  </tr>
                )}
              </tbody>
            </table>
          ) : conn.kind === "connecting" ? (
            <p className="dim">{t.connConnecting}</p>
          ) : conn.kind === "offline" ? (
            <p className="dim">
              {t.aboutConnNone}
              <br />
              <span className="connerr">{conn.error}</span>
            </p>
          ) : null}
        </section>

        <p>{t.aboutCosmos}</p>
        <p>{t.aboutTravel}</p>
        <p className="dim">
          {sourcePhase === "demo" ? fmt(t.aboutSourceDemo, { date: "2026-10-17" })
            : sourcePhase === "live" && identity
              ? fmt(t.aboutSourceLive, {
                  id: identity.index_id,
                  n: identity.nodes != null ? identity.nodes.toLocaleString("en-US") : "?",
                  m: identity.edges != null ? identity.edges.toLocaleString("en-US") : "?",
                })
              : t.aboutSourceNone}
        </p>
        <p className="dim">{t.aboutProvenance}</p>
        <p className="dim">{t.aboutRendering}</p>
        <p className="dim">
          {bridgeBefore}
          <code>{BRIDGE_DIRECT_URL}</code>
          {bridgeAfter}
        </p>
        <p className="dim">{t.aboutRepo}</p>
      </div>
    </div>
  );
}
