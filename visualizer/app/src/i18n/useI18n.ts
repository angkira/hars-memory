import { useEffect, useState } from "react";
import { getLang, setLang, subscribeLang, STRINGS, type Dict, type Lang } from "./strings";

/** Reactive i18n hook: `t` is the current dictionary, `lang` + `setLangL10n` switch it. */
export function useI18n(): { t: Dict; lang: Lang; setLangL10n: (l: Lang) => void } {
  const [, bump] = useState(0);
  useEffect(() => subscribeLang(() => bump((x) => x + 1)), []);
  return {
    t: STRINGS[getLang()],
    lang: getLang(),
    setLangL10n: setLang,
  };
}
