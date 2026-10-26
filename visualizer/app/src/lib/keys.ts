// Global keyboard shortcuts vs. text entry.
//
// The app installs several `window`-level keydown listeners (presenter
// shortcuts, Esc to dismiss overlays). Those listeners see every keystroke in
// the document, including the ones aimed at a focused field. Without a guard a
// bare " " shortcut toggles playback instead of typing a space, "f" opens
// fullscreen instead of inserting the letter, and the arrow keys jump chapters
// instead of moving the caret.
//
// `isTextEntryTarget` answers "is this keystroke meant for a field?" so every
// global handler can bail out before it steals the key.

/** Elements that consume printable keys and caret movement themselves. */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!target || typeof (target as Element).closest !== "function") return false;
  const el = target as HTMLElement;
  const tag = el.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "OPTION") return true;
  if (el.isContentEditable) return true;
  return el.closest('[contenteditable=""], [contenteditable="true"], [role="textbox"]') !== null;
}

/**
 * True when the focused field owns this keystroke and no global shortcut may
 * act on it. Escape is handled as "leave the field": it blurs the field rather
 * than reaching the overlay's own Escape action, so a second Esc still exits.
 */
export function fieldOwnsKey(e: KeyboardEvent): boolean {
  if (!isTextEntryTarget(e.target)) return false;
  if (e.key === "Escape") (e.target as HTMLElement).blur();
  return true;
}
