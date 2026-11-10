"""Textual CSS (kept as a Python string so it ships in the wheel without package-data rules)."""

CSS = """
Screen { background: $surface; }
.hidden { display: none; }

/* health */
#health-banner { height: auto; padding: 0 1; color: $text-muted; }
#health-table { height: 1fr; }
#health-detail { height: 5; border-top: solid $primary; padding: 0 1; }

/* recall */
#recall-form { height: auto; padding: 0 1; }
.frow { height: 1; margin-bottom: 0; }
.frow Label { width: auto; margin: 0 1 0 0; color: $text-muted; }
.frow Input { width: 1fr; margin-right: 1; }
.frow Select { width: 14; margin-right: 1; }
.frow Checkbox { width: auto; margin-right: 2; }
#recall-status { height: auto; padding: 0 1; background: $boost; }
#recall-main { height: 1fr; }
#stages-pane { width: 46%; }
#stages { height: 1fr; }
#stage-detail-scroll { height: 14; border-top: solid $primary; }
#stage-detail, #hit-detail { padding: 0 1; }
#recall-tabs { width: 1fr; }
.sect { background: $boost; width: 100%; padding: 0 1; }
#kg-ent, #kg-rel, #chunks { height: auto; max-height: 12; }
#hits { height: 1fr; }
#hit-detail-scroll { height: 12; border-top: solid $primary; }

/* sessions */
#sess-left { width: 62; }
#sessions { height: 1fr; }
#sess-source { height: 3; color: $text-muted; padding: 0 1; }
#sess-right { width: 1fr; }
#sess-title { height: auto; padding: 0 1; background: $boost; }
#comps { height: 40%; }
#graph { height: 1fr; }

/* projects */
#projects { width: 55%; }
#project-detail-scroll { width: 1fr; }
#project-detail { padding: 0 1; }

/* eval */
#eval-left { width: 44; }
#runs { height: 1fr; }
#eval-status { height: 3; color: $text-muted; padding: 0 1; }
#eval-tabs { width: 1fr; }
#eval-summary { padding: 0 1; }
#arms { height: 50%; }
#deltas { height: 1fr; }

/* modals */
ConfirmWriteModal, RememberModal, CompactModal { align: center middle; }
#confirm-dialog { width: 92; height: auto; max-height: 90%; border: thick $warning; background: $surface; padding: 1 2; }
#confirm-title { text-style: bold; color: $warning; }
#confirm-body { height: auto; max-height: 20; }
#confirm-guard { color: $error; text-style: bold; height: auto; margin-top: 1; }
#confirm-buttons, #form-buttons { height: 3; align: right middle; }
#form-dialog { width: 90; height: 90%; border: thick $primary; background: $surface; padding: 1 2; }
#form-dialog TextArea { height: 1fr; min-height: 5; }
"""
