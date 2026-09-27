# SRP CSS Theme Pack

Drop-in branding theme generated from your source assets in `/home/jason/Pictures/srp_theme_pack`.

## Quick Start

1. Copy the `srp-css-theme-pack` folder into your project.
2. Add this to your HTML:

```html
<link rel="stylesheet" href="/path/to/srp-css-theme-pack/css/srp-theme.css" />
<script src="/path/to/srp-css-theme-pack/js/srp-theme.js"></script>
```

3. Initialize theme mode:

```html
<script>
  SRPTheme.initTheme("srp-dark");
</script>
```

## Files

- `css/srp-theme.css` — full theme (tokens + ready component styles)
- `css/srp-theme-tokens.css` — tokens only
- `js/srp-theme.js` — optional dark/light toggle helper with debug logging
- `assets/*` — copied brand assets
- `index.html` — preview/demo page
- `MANUAL.md` — practical integration manual
- `OVERVIEW.md` — architectural summary
- `CHANGELOG.md` — change history
- `TODO.md` — next improvements
- `KAKI.md` — notes/suggestions

## Notes

- Theme defaults to dark mode (`data-theme="srp-dark"`).
- Light mode is available via `data-theme="srp-light"`.
- `srp-theme.js` persists mode in `localStorage` key `srp-theme-mode`.
