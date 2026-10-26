// AnimatedCard — THE reusable animated card used everywhere (selection
// tooltip, presentation fact/narration cards, stage inspector header, ingest
// document preview). Two-phase entrance per the design language:
//   phase 1 (~300ms): the gradient-edged frame draws itself on
//                     (border-gradient sweep + frame fade/scale-in)
//   phase 2 (~220ms): content slides/fades in (small translate, no bounce)
// Exit: quick fade + scale-down (180ms) via the `open` prop; unmounts itself.
//
// Monochrome palette with ONE accent gradient (teal) — matches the design
// system, no rainbow. `prefers-reduced-motion` collapses all phases (global
// media query in index.css + local guards here).
import { useEffect, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

export type CardVariant = "accent" | "gold" | "plain";
export type CardSize = "sm" | "md" | "lg";

interface Props {
  open: boolean;
  /** unmount-on-exit handled internally: keep rendering while exiting */
  children: ReactNode;
  title?: ReactNode;
  /** small line under the title (meta/kicker) */
  meta?: ReactNode;
  variant?: CardVariant;
  size?: CardSize;
  className?: string;
  style?: CSSProperties;
  /** stagger the two-phase entrance (left→right card rows); phase 2 waits 260ms */
  delayMs?: number;
  onClose?: () => void;
  closeTitle?: string;
}

const EXIT_MS = 180;

export function AnimatedCard({
  open,
  children,
  title,
  meta,
  variant = "accent",
  size = "md",
  className = "",
  style,
  delayMs = 0,
  onClose,
  closeTitle,
}: Props) {
  // `shown` stays true through the exit animation so the fade can play.
  const [shown, setShown] = useState(open);
  useEffect(() => {
    if (open) setShown(true);
    else if (shown) {
      const id = window.setTimeout(() => setShown(false), EXIT_MS);
      return () => window.clearTimeout(id);
    }
  }, [open, shown]);
  if (!shown) return null;
  const cls = [
    "acard",
    `acard-${variant}`,
    `acard-${size}`,
    open ? "acard-in" : "acard-out",
    className,
  ].join(" ");
  return (
    <div
      className={cls}
      style={{ ...(delayMs ? { "--acard-delay": `${delayMs}ms` } : null), ...style } as CSSProperties}
      role={onClose ? "dialog" : undefined}
    >
      <i className="acard-frame" aria-hidden="true" />
      <div className="acard-body">
        {(title || onClose) && (
          <div className="acard-head">
            <div className="acard-titlewrap">
              {title && <h4 className="acard-title">{title}</h4>}
              {meta && <span className="acard-meta">{meta}</span>}
            </div>
            {onClose && (
              <button className="ghostbtn" onClick={onClose} title={closeTitle ?? "close"}>
                <XGlyph />
              </button>
            )}
          </div>
        )}
        <div className="acard-content">{children}</div>
      </div>
    </div>
  );
}

function XGlyph() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" className="icon" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" aria-hidden="true">
      <line x1="5" y1="5" x2="19" y2="19" />
      <line x1="19" y1="5" x2="5" y2="19" />
    </svg>
  );
}
