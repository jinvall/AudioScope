# CHANGELOG

## 1.1.0 - 2026-09-14

- Added spectral gradient with 6 color stops: dark navy → violet → indigo → blue-purple → green → slate
- New CSS custom properties: `--srp-spectral-0` through `--srp-spectral-5` in both tokens and main CSS
- New utility classes: `.srp-spectral-bg`, `.srp-spectral-bar`, `.srp-spectral-banner`, `.srp-spectral-text`
- Direction variants: `.srp-spectral-bg-vertical`, `.srp-spectral-bg-diagonal`, `.srp-spectral-bg-radial`
- Individual stop utilities: `.spectral-stop-{0-5}` (text color) and `.bg-spectral-stop-{0-5}` (background)
- Added spectral gradient demo section to `index.html`

## 1.0.0 - 2026-07-19

- Created `srp-css-theme-pack` from source branding assets.
- Added drop-in CSS theme file `css/srp-theme.css`.
- Added token-only file `css/srp-theme-tokens.css`.
- Added JS helper `js/srp-theme.js` for dark/light mode persistence.
- Added preview page `index.html`.
- Copied and normalized key brand assets into `assets/`.
- Added documentation: `README.md`, `MANUAL.md`, `OVERVIEW.md`, `TODO.md`, `KAKI.md`.
