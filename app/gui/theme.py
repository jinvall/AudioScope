"""Theme-pack integration: SRP tokens in, Qt styling out.

The design system lives at ``app/gui/srp-css-theme-pack/`` as CSS custom
properties. Qt does not read CSS variables, so this module parses the token
files and translates the palette into a Qt stylesheet plus a small accessor the
widgets use for colours Qt has no stylesheet hook for (waveform strokes,
spectrogram ramps, plot backgrounds).

The tokens are read from the files at runtime rather than copied into Python.
That is the point: the theme pack stays the single source of truth, so editing
``srp-theme.css`` changes the application without touching this file.

Two flavours are supported, matching the pack's ``srp-dark`` (default) and
``srp-light`` modes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

#: Location of the design system, relative to this file.
THEME_PACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "srp-css-theme-pack")
TOKENS_FILE = os.path.join(THEME_PACK_DIR, "css", "srp-theme-tokens.css")
THEME_FILE = os.path.join(THEME_PACK_DIR, "css", "srp-theme.css")

_TOKEN_RE = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
_BLOCK_RE = re.compile(r"([^{}]+)\{([^}]*)\}", re.DOTALL)

#: Spectral ramp from the theme pack, dark flavour.  Used for the spectrogram so
#: it matches the pack's intended look rather than an arbitrary colormap.
SPECTRAL_FALLBACK = (
    "#060119", "#2F076E", "#370150", "#5633C7", "#134E1A", "#65637E",
)


@dataclass(frozen=True)
class Theme:
    """A resolved set of tokens plus the paths they came from."""

    mode: str
    tokens: dict
    sources: tuple

    # -- token access -------------------------------------------------
    #: The pack names the same concept two ways across its files: the theme
    #: file uses ``--srp-bg`` and the token file uses ``--color-bg``.  Lookups
    #: are by the short name and resolve through these prefixes, so widgets ask
    #: for ``bg`` rather than knowing which file a token happens to live in.
    PREFIXES = ("srp-", "color-", "")

    def get(self, name: str, default: Optional[str] = None) -> str:
        if name in self.tokens:
            return self.tokens[name]
        for prefix in self.PREFIXES:
            key = f"--{prefix}{name}"
            if key in self.tokens:
                return self.tokens[key]
        return default if default is not None else ""

    def color(self, name: str, default: str = "#888888") -> str:
        """A colour token, resolved to ``#rrggbb`` for Qt."""
        raw = self.get(name)
        return normalise_color(raw, default) if raw else default

    def px(self, name: str, default: int = 8) -> int:
        """A spacing or radius token, in pixels."""
        raw = self.get(name)
        if not raw:
            return default
        match = re.search(r"(-?\d+(?:\.\d+)?)", raw)
        if not match:
            return default
        # rem is 16px at the pack's root sizing.
        value = float(match.group(1))
        if "rem" in raw:
            value *= 16.0
        return int(round(value))

    def spectral_ramp(self) -> list:
        """Spectral gradient stops, dark flavour first."""
        stops = []
        for index in range(7):
            value = self.get(f"spectral-{index}")
            if value:
                stops.append(normalise_color(value, SPECTRAL_FALLBACK[index]))
        return stops or [normalise_color(c, "#888888")
                         for c in SPECTRAL_FALLBACK]

    @property
    def available(self) -> bool:
        return bool(self.tokens)


# ----------------------------------------------------------------------
def normalise_color(value: str, default: str = "#888888") -> str:
    """Turn a CSS colour into ``#rrggbb`` (or ``#rrggbbaa``).

    Qt's stylesheet parser is unreliable with 8-digit hex and with ``rgba()``,
    so colours are resolved here instead.
    """
    if not value:
        return default
    text = value.strip().lower()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        if len(digits) in (6, 8) and all(
            c in "0123456789abcdef" for c in digits
        ):
            return "#" + digits[:6]
        return default
    match = re.match(r"rgba?\(([^)]+)\)", text)
    if match:
        parts = [p.strip() for p in match.group(1).split(",")[:3]]
        try:
            r, g, b = (int(round(float(p))) for p in parts)
        except (ValueError, TypeError):
            return default
        return "#%02x%02x%02x" % (
            max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        )
    return default


def _parse_blocks(text: str) -> list:
    """Yield ``(selector_text, body)`` for each CSS block."""
    for match in _BLOCK_RE.finditer(text):
        yield match.group(1), match.group(2)


def load_theme(mode: str = "dark", theme_dir: Optional[str] = None) -> Theme:
    """Parse the theme pack and return a resolved :class:`Theme`.

    Falls back to an empty token set if the pack is missing, so a missing
    design system degrades to unstyled Qt rather than a crash.
    """
    mode = "light" if str(mode).lower() in ("light", "srp-light") else "dark"
    tokens_path = os.path.join(theme_dir or THEME_PACK_DIR, "css",
                              "srp-theme-tokens.css")
    theme_path = os.path.join(theme_dir or THEME_PACK_DIR, "css",
                              "srp-theme.css")
    tokens: dict = {}
    sources = []

    for path in (theme_path, tokens_path):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        sources.append(path)

        # The pack's cascade is: the base block defines every token, and the
        # optional light flavour overrides *colours* only.  Mirroring that
        # matters - parsing only the requested flavour leaves light mode
        # without radii or spacing, because the light block never declares them.
        def collect(want_light: bool) -> dict:
            found: dict = {}
            for selector, body in _parse_blocks(text):
                selector_l = selector.lower()
                is_light = "srp-light" in selector_l
                is_dark = "srp-dark" in selector_l or ":root" in selector_l
                if not (is_light or is_dark):
                    continue
                if is_light != want_light:
                    continue
                for name, value in _TOKEN_RE.findall(body):
                    found[name.strip()] = value.strip()
            return found

        tokens.update(collect(want_light=False))
        if mode == "light":
            tokens.update(collect(want_light=True))


    return Theme(mode=mode, tokens=tokens, sources=tuple(sources))


# ----------------------------------------------------------------------
#: Base text size, in pixels, before scaling.  Every other text size in the
#: stylesheet is derived from this one, so text can be made larger for a
#: particular screen or pair of eyes without editing the stylesheet.
BASE_FONT_PX = 15

#: The smallest multiplier accepted.  Below 1 the window stops being readable,
#: which is worse than having no size control at all.
MIN_FONT_SCALE = 0.8

#: The largest accepted, chosen to stay inside a 1280-wide window before the
#: two-column inspector has to be scrolled.
MAX_FONT_SCALE = 2.0


def build_stylesheet(theme: Theme, extra: str = "",
                     font_scale: float = 1.0) -> str:
    """Translate tokens into a Qt stylesheet.

    Qt's QSS is a subset of CSS: no custom properties, no flexbox, and the
    selectors differ. Every colour and metric comes from the token set, so the
    theme pack's palette, radii and spacing drive the whole application.

    ``font_scale`` multiplies every text size, so one setting makes the whole
    application larger or smaller together.  Derived from a single base rather
    than written out per rule, because sizes that are individually correct but
    collectively too small are exactly the failure this exists to fix.
    """
    if not theme.available:
        return extra

    try:
        scale = float(font_scale)
    except (TypeError, ValueError):
        scale = 1.0
    scale = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, scale))

    def font(px: int) -> int:
        return max(1, int(round(px * scale)))

    base_px = font(BASE_FONT_PX)
    heading_px = font(BASE_FONT_PX + 3)
    title_px = font(BASE_FONT_PX + 6)

    c = theme.color
    radius = theme.px("radius-md", 12)
    radius_sm = theme.px("radius-sm", 8)
    pad = theme.px("space-2", 8)
    pad_lg = theme.px("space-4", 16)
    border = theme.color("border", "#3d2d61")

    parts = [f"""
    QWidget {{
        background-color: {c('bg', '#0f0a18')};
        color: {c('text', '#f3f3f7')};
        font-size: {base_px}px;
    }}
    QLabel#Muted, QLabel#Subtle {{ color: {c('text-muted', '#c8c9d4')}; }}
    QLabel#Subtle {{ color: {c('text-subtle', '#9698ab')}; }}
    QLabel#Heading {{ font-size: {heading_px}px; font-weight: 600; }}
    QLabel#Title {{ font-size: {title_px}px; font-weight: 600; }}
    QFrame#Card {{
        background-color: {c('surface', '#1a1230')};
        border: 1px solid {border};
        border-radius: {radius}px;
    }}
    QFrame#Header {{
        background-color: {c('bg-elev-1', '#161028')};
        border-bottom: 1px solid {border};
    }}
    QListWidget, QTreeWidget, QTableWidget, QTextEdit, QPlainTextEdit, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {c('bg-elev-1', '#161028')};
        border: 1px solid {border};
        border-radius: {radius_sm}px;
        padding: {pad - 2}px;
        selection-background-color: {c('primary', '#12f012')};
        selection-color: {c('primary-contrast', '#041105')};
    }}
    QListWidget::item {{
        padding: {pad}px {pad_lg}px {pad}px {pad_lg}px;
        border-bottom: 1px solid {border};
    }}
    /* Narrower padding on the table: the row layout already has cell padding
       of its own, and the generous list padding elided the column values. */
    QTreeWidget::item {{
        padding: {pad - 2}px {pad}px {pad - 2}px {pad}px;
        border-bottom: 1px solid {border};
    }}
    QTreeView::item {{ border: none; }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {c('primary', '#12f012')};
        color: {c('primary-contrast', '#041105')};
    }}
    QPushButton {{
        background-color: {c('surface-2', '#231743')};
        border: 1px solid {border};
        border-radius: {radius_sm}px;
        padding: {pad}px {pad_lg}px;
        color: {c('text', '#f3f3f7')};
    }}
    QPushButton:hover {{ border-color: {c('primary', '#12f012')}; }}
    QPushButton:checked {{
        background-color: {c('primary', '#12f012')};
        color: {c('primary-contrast', '#041105')};
        font-weight: 600;
    }}
    QPushButton#Danger:checked {{
        background-color: {c('danger', '#ff4d6d')};
        color: {c('bg', '#0f0a18')};
        font-weight: 600;
    }}
    QPushButton:disabled {{ color: {c('text-subtle', '#9698ab')}; }}
    QPushButton#Primary {{
        background-color: {c('primary', '#12f012')};
        color: {c('primary-contrast', '#041105')};
        font-weight: 600;
        border-color: {c('primary', '#12f012')};
    }}
    QPushButton#Danger {{
        border-color: {c('danger', '#ff4d6d')};
        color: {c('danger', '#ff4d6d')};
    }}
    QPushButton#Danger:checked {{
        background-color: {c('danger', '#ff4d6d')};
        color: {c('bg', '#0f0a18')};
    }}
    QSlider::groove:horizontal {{
        height: 4px; background: {c('surface-2', '#231743')};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c('primary', '#12f012')};
        width: 12px; margin: -5px 0; border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{ background: {c('primary', '#12f012')}; }}
    QTabWidget::pane {{
        border: 1px solid {border}; border-radius: {radius}px;
        background-color: {c('surface', '#1a1230')};
    }}
    QTabBar::tab {{
        background: {c('bg-elev-1', '#161028')};
        border: 1px solid {border};
        padding: {pad}px {pad_lg}px;
        border-top-left-radius: {radius_sm}px;
        border-top-right-radius: {radius_sm}px;
    }}
    QTabBar::tab:selected {{ background: {c('surface', '#1a1230')}; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: {c('bg', '#0f0a18')}; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c('border', '#3d2d61')};
        border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QStatusBar {{
        background: {c('bg-elev-1', '#161028')};
        border-top: 1px solid {border};
    }}
    QToolTip {{
        background: {c('surface-2', '#231743')};
        color: {c('text', '#f3f3f7')};
        border: 1px solid {border};
        padding: 4px;
    }}
    QSplitter::handle {{ background: {border}; }}
    """.strip()]

    if extra:
        parts.append(extra.strip())
    return "\n".join(parts)
