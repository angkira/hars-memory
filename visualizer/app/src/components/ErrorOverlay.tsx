import { useI18n } from "../i18n/useI18n";
import { AlertIcon, XIcon } from "./icons";

interface Props {
  title: string;
  body: string;
  detail?: string | null;
  onDismiss?: () => void;
  dismissLabel?: string;
  /** Extra action buttons (e.g. "Open demo data") rendered under the body. */
  actions?: React.ReactNode;
}

/** Full-view readable failure card. Replaces the silent black canvas: any
 *  init / data-load / render failure funnels here with stage + detail.
 *  The default dismiss label is localised; callers may still override it. */
export default function ErrorOverlay({ title, body, detail, onDismiss, dismissLabel, actions }: Props) {
  const { t } = useI18n();
  return (
    <div className="errwrap" role="alert">
      <div className="errcard">
        <h2>
          <AlertIcon size={18} /> {title}
        </h2>
        <p>{body}</p>
        {detail && <pre className="errdetail">{detail}</pre>}
        {actions && <div className="overlay-actions">{actions}</div>}
        {onDismiss && (
          <button onClick={onDismiss}>
            <XIcon size={12} /> {dismissLabel ?? t.errDismiss}
          </button>
        )}
      </div>
    </div>
  );
}
