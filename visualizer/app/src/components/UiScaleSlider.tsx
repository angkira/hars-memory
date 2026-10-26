import { useEffect, useState } from "react";
import { useI18n } from "../i18n/useI18n";
import {
  autoScaleForViewport,
  getUiScale,
  isAutoScale,
  resetUiScale,
  setUiScale,
  subscribeUiScale,
  UI_SCALE_MAX,
  UI_SCALE_MIN,
  UI_SCALE_STEP,
} from "../lib/uiScale";

/**
 * Local {en, ru} labels — kept inside this component on purpose: the shared
 * i18n dictionaries (src/i18n/strings.ts) are owned by another agent, so the
 * scale control reads the language from the existing hook and carries its own
 * two strings instead of adding keys there.
 */
const L: Record<"en" | "ru", {
  group: string;
  slider: string;
  percent: string;
  reset: string;
  resetHint: string;
  autoNote: string;
}> = {
  en: {
    group: "UI scale",
    slider: "Interface scale",
    percent: "Interface scale, {p} percent",
    reset: "auto",
    resetHint: "Back to the automatic scale for this screen ({p}%)",
    autoNote: "auto — the default for this screen size; use the slider to override",
  },
  ru: {
    group: "Масштаб интерфейса",
    slider: "Масштаб интерфейса",
    percent: "Масштаб интерфейса, {p} процентов",
    reset: "авто",
    resetHint: "Вернуть автоматический масштаб для этого экрана ({p}%)",
    autoNote: "авто — значение по умолчанию для такого экрана; ползунок задаёт своё",
  },
};

/**
 * Top-bar UI-scale slider: 0.75–1.60 in 0.05 steps, live percentage readout, and
 * a reset-to-auto affordance (the "auto" button, or a double-click on the
 * readout). Explicit values persist in localStorage and win over the auto
 * default; the range input is natively keyboard operable (arrows / Home / End).
 */
export default function UiScaleSlider() {
  const { lang } = useI18n();
  const [scale, setScale] = useState(getUiScale);
  const [auto, setAuto] = useState(isAutoScale);
  useEffect(() => subscribeUiScale((s, a) => { setScale(s); setAuto(a); }), []);

  const t = L[lang] ?? L.en;
  const pct = Math.round(scale * 100);
  const autoHint = t.resetHint.replace("{p}", String(Math.round(autoScaleForViewport() * 100)));

  return (
    <div className="uiscale" role="group" aria-label={t.group}>
      <span className="uiscale-key" aria-hidden="true">UI</span>
      <input
        className="uiscale-range"
        type="range"
        min={UI_SCALE_MIN}
        max={UI_SCALE_MAX}
        step={UI_SCALE_STEP}
        value={scale}
        aria-label={t.slider}
        aria-valuetext={t.percent.replace("{p}", String(pct))}
        onChange={(e) => setUiScale(Number(e.target.value))}
      />
      <span
        className={auto ? "uiscale-pct is-auto" : "uiscale-pct"}
        title={auto ? t.autoNote : autoHint}
        onDoubleClick={() => resetUiScale()}
      >
        {pct}%
      </span>
      <button
        type="button"
        className={auto ? "uiscale-reset is-auto" : "uiscale-reset"}
        onClick={() => resetUiScale()}
        title={autoHint}
        aria-label={`${t.reset} — ${autoHint}`}
      >
        {t.reset}
      </button>
    </div>
  );
}
