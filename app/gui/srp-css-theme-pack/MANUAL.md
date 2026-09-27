# MANUAL — SRP Theme Pack

## 1) Minimal Integration

```html
<html data-theme="srp-dark">
  <head>
    <link rel="stylesheet" href="./srp-css-theme-pack/css/srp-theme.css" />
  </head>
  <body>
    <button class="srp-btn srp-btn-primary">Go</button>
  </body>
</html>
```

## 2) Utility Classes

- `.srp-container`
- `.srp-card`
- `.srp-btn .srp-btn-primary .srp-btn-secondary`
- `.srp-input .srp-select .srp-textarea`
- `.srp-topbar .srp-brand .srp-brand-logo .srp-badge`

## 3) Theme Switching

Use either:
- attribute switch: `document.documentElement.setAttribute('data-theme', 'srp-light')`
- helper: `SRPTheme.toggleTheme()`

## 4) Background Options

- Add `class="srp-background"` on `<body>` to use packaged background image.
- Override image with:

```css
:root {
  --srp-bg-image: url('/my/background.svg');
}
```

## 5) Debugging

- JS logs are prefixed with `[SRP_THEME]`.
- For layout debugging, add `.srp-debug-outline` to a wrapper.
