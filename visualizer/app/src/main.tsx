import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// live-query sheet — imported AFTER index.css so its equal-specificity
// overrides (query-bar width rule + pipeline step list) win the cascade.
import './query-steps.css'
// presentation three-column layout — must come after index.css so its
// deliberate overrides of the old floating presentation chrome win.
import './presentation-layout.css'
// ui-scale sheet: #root{zoom} + the .canvaswrap counter-zoom (see ui-scale.css).
import './ui-scale.css'
import App from './App.tsx'
import { initUiScale } from './lib/uiScale'

// Resolve and apply the UI scale (saved preference, else the big-screen default
// for this viewport) BEFORE the first paint, so a 4K screen never flashes the
// unscaled layout. Sets CSS custom properties only — no React state involved.
initUiScale()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
